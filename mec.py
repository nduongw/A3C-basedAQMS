from utils.GnbSever import Network
from utils.Init import Init
from utils.Utils import *
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import yaml
import os
from models.model import Model
from utils.shared_adam import SharedAdam
import torch 
from utils.func import v_wrap, set_init, push_and_pull, record
import torch.multiprocessing as mp
import torch.nn.functional as F

os.environ["OMP_NUM_THREADS"] = "1"
UPDATE_GLOBAL_ITER = 5
GAMMA = 0.9
MAX_EP = 3000


with open('config/hyperparameter.yaml') as f:
    Config = yaml.safe_load(f)

init = Init(Config)
init.create_map()
print(init.map)
init.set_cover_radius()
x = []
y = []
sns.set_theme()
plt.figure(figsize=(10, 7))
plt.xlim([0, 70])
plt.ylim([0, 100])
# plt.margins(0)
plt.xticks(np.arange(0, 80, step=10))
plt.yticks(np.arange(0, 110, step=10))

# print(init.map)
# init.set_cover_radius()
# for i in range(2):
#     init.run_per_second()
# sns.heatmap(data=init.cover_map[-1], cmap='Blues')
# plt.show()

# plt.ion()

# for i in range(10):
#     x = []
#     y = []
#     for i in range(init.map.shape[0]):
#         for j in range(init.map.shape[1]):
#             if init.map[i, j] == 1:
#                 # print(i, j)
#                 x.append(i)
#                 y.append(j)      
#     init.run_per_second()
#     plt.scatter(y, x)
#     # map_a = init.get_cover_map()
#     # plt.draw()
#     plt.pause(0.5)
#     plt.cla()
#     x.clear()
#     y.clear()
# plt.show(block=True)

# sns.heatmap(data=init.cover_map[-1], cmap='Blues')
# plt.show()
action = torch.rand([5, 5])
network = Network(init.map, Config)
new_action = network.change_on_off(action)
print(new_action)

class Net(Model):
    def __init__(self, Config):
        super(Net, self).__init__(Config)
        self.distribution = torch.distributions.Categorical

    def forward(self, x):
        logits, values = super(Net, self).forward(x)
        return logits, values

    def choose_action(self, s):
        self.eval()
        logits, _ = self.forward(s)
        prob = F.softmax(logits, dim=1).data
        m = self.distribution(prob)
        return m.sample().numpy()[0]

    def loss_func(self, s, a, v_t):
        self.train()
        logits, values = self.forward(s)
        td = v_t - values
        c_loss = td.pow(2)
        
        probs = F.softmax(logits, dim=1)
        m = self.distribution(probs)
        exp_v = m.log_prob(a) * td.detach().squeeze()
        a_loss = -exp_v
        total_loss = (c_loss + a_loss).mean()
        return total_loss


class Worker(mp.Process):
    def __init__(self, gnet, opt, global_ep, global_ep_r, res_queue, name, Config):
        super(Worker, self).__init__()
        self.name = 'w%02i' % name
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.gnet, self.opt = gnet, opt
        self.lnet = Model(Config)           # local network
        # self.env = gym.make('CartPole-v0').unwrapped

    def run(self):
        total_step = 1
        while self.g_ep.value < MAX_EP:
            s = self.env.reset()
            buffer_s, buffer_a, buffer_r = [], [], []
            ep_r = 0.
            while True:
                if self.name == 'w00':
                    self.env.render()
                a = self.lnet.choose_action(v_wrap(s[None, :]))
                s_, r, done, _ = self.env.step(a)
                if done: r = -1
                ep_r += r
                buffer_a.append(a)
                buffer_s.append(s)
                buffer_r.append(r)

                if total_step % UPDATE_GLOBAL_ITER == 0 or done:  # update global and assign to local net
                    # sync
                    push_and_pull(self.opt, self.lnet, self.gnet, done, s_, buffer_s, buffer_a, buffer_r, GAMMA)
                    buffer_s, buffer_a, buffer_r = [], [], []

                    if done:  # done and print information
                        record(self.g_ep, self.g_ep_r, ep_r, self.res_queue, self.name)
                        break
                s = s_
                total_step += 1
        self.res_queue.put(None)



if __name__=='__main__':
    gnet = Net()
    gnet.share_memory()
    opt = SharedAdam(gnet.parameters(), lr=1e-4, betas=(0.92, 0.999))
    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()

    # parallel training
    workers = [Worker(gnet, opt, global_ep, global_ep_r, res_queue, i, Config) for i in range(mp.cpu_count())]
    [w.start() for w in workers]

    [w.join() for w in workers]

