from utils.Utils import *
import matplotlib.pyplot as plt
import numpy as np
import yaml

import torch 
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.nn as nn

from utils.Env import Env
from utils.A2C import ActorCritic, ParallelEnv, compute_target

with open('config/hyperparameter.yaml') as f:
    Config = yaml.safe_load(f)

# init = Map(Config)
# init.create_map()
# print(init.map)
# init.set_cover_radius()
# x = []
# y = []
# sns.set_theme()
# plt.figure(figsize=(10, 7))
# plt.xlim([0, 70])
# plt.ylim([0, 100])
# plt.margins(0)
# plt.xticks(np.arange(0, 80, step=10))
# plt.yticks(np.arange(0, 110, step=10))

# print(init.map)
# init.set_cover_radius()
# for i in range(2):
#     init.run_per_second()
# sns.heatmap(data=init.cover_map[-1], cmap='Blues')
# plt.show()

# plt.ion()


# sns.heatmap(data=init.cover_map[-1], cmap='Blues')
# plt.show()
# action = torch.rand([5, 5])
# network = Network(init.map, Config)
# new_action = network.change_on_off(action)
# print(new_action)



n_train_processes = Config.get("n_train_processes")
learning_rate = Config.get("learning_rate")
update_interval = Config.get("update_interval")
max_train_steps = Config.get("max_train_steps")
PRINT_INTERVAL = update_interval * 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test(step_idx, model):
    env = Env(Config)
    score = 0.0
    done = 0
    num_test = 1

    for _ in range(num_test):
        s = env.reset()     #(num_frame, 2, h, w)
        while done <10:
            prob = model.pi(torch.unsqueeze(torch.from_numpy(np.array(s)), 0).float().to(device)).detach()   #(1, h, w)
            a = env.env.map_to_action(torch.squeeze(prob)) # ( h, w)
            s_prime, r = env.step(a)
            s = s_prime
            score += r
            done += 1
        
    print(f"Step # :{step_idx}, avg score : {score/num_test:.1f}")

    # env.close()


if __name__ == '__main__':
    mp.set_start_method('spawn')
    envs = ParallelEnv(n_train_processes)

    model = ActorCritic().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    step_idx = 0
    s = envs.reset()        #(n_env, num_frame, 2, h, w)
    while step_idx < max_train_steps:
        s_list, a_list, r_list = list(), list(), list()
        for _ in range(update_interval):
            prob = model.pi(torch.from_numpy(s).float().to(device)).detach() #(n_env, h, w)
            # a = Categorical(prob).sample().numpy()
            a = envs.choose_action(prob) # (n_env, h, w)
            s_prime, r = envs.step(a)   # (n_env, num_frame, 2, h, w) and (n_env, 1)

            s_list.append(s)
            a_list.append(a)
            r_list.append(r)
            # mask_list.append(1 - done)

            s = s_prime
            step_idx += 1
        
        s_final = torch.from_numpy(s_prime).float()     # (n_env, num_frame, 2, h, w)
        v_final = model.v(s_final.to(device)).to('cpu').detach().clone().numpy()     # (n_env, 1)
        td_target = compute_target(v_final, r_list)     # (update_interval, n_env)

        td_target_vec = td_target.reshape(-1).to(device)      # (update_interval*n_env) nối update_interval hàng thành 1 hàng
        # s_vec = torch.tensor(s_list).float().reshape(-1, 4)  # 4 == Dimension of state
        s_vec = torch.from_numpy(np.concatenate(s_list)).float()       #(n_env*len(s_list), num_frame, 2, h, w)
        # a_vec = torch.tensor(a_list).reshape(-1).unsqueeze(1)
        a_vec = torch.from_numpy(np.concatenate(a_list)).float().to(device)      #(n_env*len(s_list), h, w)
        advantage = td_target_vec - model.v(s_vec.to(device)).reshape(-1)      #(update_interval*n_env)

        pi = model.pi(s_vec.to(device))        #(n_env*len(s_list), h, w)
        # pi_a = pi.gather(1, a_vec).reshape(-1)
        pi_a = torch.mean(pi*a_vec, dim=(1, 2)) #(update_interval*n_env)
        loss = -(torch.log(pi_a) * advantage.detach()).mean() +\
            F.smooth_l1_loss(model.v(s_vec.to(device)).reshape(-1), td_target_vec)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step_idx % PRINT_INTERVAL == 0:
            test(step_idx, model)

    envs.close()

