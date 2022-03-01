from utils.GnbSever import Network
from utils.Map import Map
from utils.Utils import *
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import yaml
import os
from models.model import Model
import torch 
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.nn as nn
import gym

with open('config/hyperparameter.yaml') as f:
    Config = yaml.safe_load(f)

init = Map(Config)
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



n_train_processes = 3
learning_rate = 0.0002
update_interval = 5
gamma = 0.98
max_train_steps = 60000
PRINT_INTERVAL = update_interval * 100

class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.model = Model()

    def pi(self, x, softmax_dim=1):
        x = F.relu(self.fc1(x))
        x, _ = self.model(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob

    def v(self, x):
        _, v = self.model(x)
        return v

def worker(worker_id, master_end, worker_end):
    master_end.close()  # Forbid worker to use the master end for messaging
    env = gym.make('CartPole-v1')
    env.seed(worker_id)

    while True:
        cmd, action = worker_end.recv()
        if cmd == 'step':
            ob, reward, done, info = env.step(action)
            if done:
                ob = env.reset()
            worker_end.send((ob, reward, done, info))
        elif cmd == 'reset':
            ob = env.reset()
            worker_end.send(ob)
        elif cmd == 'reset_task':
            ob = env.reset_task()
            worker_end.send(ob)
        elif cmd == 'close':
            worker_end.close()
            break
        elif cmd == 'get_spaces':
            worker_end.send((env.observation_space, env.action_space))
        else:
            raise NotImplementedError

class ParallelEnv:
    def __init__(self, n_train_processes):
        self.nenvs = n_train_processes
        self.waiting = False
        self.closed = False
        self.workers = list()
        master_ends, worker_ends = zip(*[mp.Pipe() for _ in range(self.nenvs)])
        self.master_ends, self.worker_ends = master_ends, worker_ends

        for worker_id, (master_end, worker_end) in enumerate(zip(master_ends, worker_ends)):
            p = mp.Process(target=worker,
                           args=(worker_id, master_end, worker_end))
            p.daemon = True
            p.start()
            self.workers.append(p)

        # Forbid master to use the worker end for messaging
        for worker_end in worker_ends:
            worker_end.close()

    def step_async(self, actions):
        for master_end, action in zip(self.master_ends, actions):
            master_end.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [master_end.recv() for master_end in self.master_ends]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        for master_end in self.master_ends:
            master_end.send(('reset', None))
        return np.stack([master_end.recv() for master_end in self.master_ends])

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def close(self):  # For clean up resources
        if self.closed:
            return
        if self.waiting:
            [master_end.recv() for master_end in self.master_ends]
        for master_end in self.master_ends:
            master_end.send(('close', None))
        for worker in self.workers:
            worker.join()
            self.closed = True

def test(step_idx, model):
    env = gym.make('CartPole-v1')
    score = 0.0
    done = False
    num_test = 10

    for _ in range(num_test):
        s = env.reset()
        while not done:
            prob = model.pi(torch.from_numpy(s).float(), softmax_dim=0)
            a = Categorical(prob).sample().numpy()
            s_prime, r, done, info = env.step(a)
            s = s_prime
            score += r
        done = False
    print(f"Step # :{step_idx}, avg score : {score/num_test:.1f}")

    env.close()

def compute_target(v_final, r_lst, mask_lst):
    G = v_final.reshape(-1)
    td_target = list()

    for r, mask in zip(r_lst[::-1], mask_lst[::-1]):
        G = r + gamma * G * mask
        td_target.append(G)

    return torch.tensor(td_target[::-1]).float()

if __name__ == '__main__':
    envs = ParallelEnv(n_train_processes)

    model = ActorCritic()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    step_idx = 0
    s = envs.reset()
    while step_idx < max_train_steps:
        s_lst, a_lst, r_lst, mask_lst = list(), list(), list(), list()
        for _ in range(update_interval):
            prob = model.pi(torch.from_numpy(s).float())
            a = Categorical(prob).sample().numpy()
            s_prime, r, done, info = envs.step(a)

            s_lst.append(s)
            a_lst.append(a)
            r_lst.append(r/100.0)
            mask_lst.append(1 - done)

            s = s_prime
            step_idx += 1

        s_final = torch.from_numpy(s_prime).float()
        v_final = model.v(s_final).detach().clone().numpy()
        td_target = compute_target(v_final, r_lst, mask_lst)

        td_target_vec = td_target.reshape(-1)
        s_vec = torch.tensor(s_lst).float().reshape(-1, 4)  # 4 == Dimension of state
        a_vec = torch.tensor(a_lst).reshape(-1).unsqueeze(1)
        advantage = td_target_vec - model.v(s_vec).reshape(-1)

        pi = model.pi(s_vec, softmax_dim=1)
        pi_a = pi.gather(1, a_vec).reshape(-1)
        loss = -(torch.log(pi_a) * advantage.detach()).mean() +\
            F.smooth_l1_loss(model.v(s_vec).reshape(-1), td_target_vec)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step_idx % PRINT_INTERVAL == 0:
            test(step_idx, model)

    envs.close()

