import torch 
import torch.multiprocessing as mp
import torch.nn.functional as F
import numpy as np
import yaml

from utils.Env import Env
from utils.Utils import *
from utils.A2C import ActorCritic, ParallelEnv, compute_target

with open('config/hyperparameter.yaml') as f:
    Config = yaml.safe_load(f)

n_train_processes = Config.get("n_train_processes")
learning_rate = Config.get("learning_rate")
update_interval = Config.get("update_interval")
max_train_steps = Config.get("max_train_steps")

PRINT_INTERVAL = update_interval * 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test(step_idx, model, env):
    score = 0.0
    done = 0
    num_test = 1
    s = env.reset()

    for _ in range(num_test):
        while done < 500:
            prob = model.pi(torch.unsqueeze(torch.from_numpy(np.array(s)), 0).float().to(device)).detach()   #(1, h, w)
            a = env.env.map_to_action(torch.squeeze(prob)) # ( h, w)
            send_car = len(count_car(a))
            total_car = len(count_car(env.env.map))
            s_prime, r = env.step(a)
            cover_score = torch.count_nonzero(env.env.cover_map).item()
            total_score = Config.get('road_length') * Config.get('road_width')
            avg_a = (send_car / total_car) * 100
            print(f"Step :{done}, avg score : {r:.1f}, avg_cover_radius : {cover_score / total_score * 100 : .2f}, avg_sent_package: {avg_a}")
            s = s_prime
            score += r
            done += 1

    # print(f"Step :{step_idx}, avg score : {score/num_test:.1f}, avg_cover_radius : {cover_score / total_score * 100 : .2f}, avg_sent_package: {avg_a}")
    # cover_score = torch.count_nonzero(env.env.cover_map).item()
    # total_score = Config.get('road_length') * Config.get('road_width')
    # avg_a = (sum(a_lst) / sum(car_lst)) * 100


if __name__ == '__main__':
    env = Env(Config)
    mp.set_start_method('spawn')
    envs = ParallelEnv(n_train_processes)

    model = ActorCritic().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    step_idx = 0
    s = envs.reset()        #(n_env, num_frame, 2, h, w)
    loss_lst = []

    while step_idx < max_train_steps:
        s_list, a_list, r_list = list(), list(), list()
        for _ in range(update_interval):
            prob = model.pi(torch.from_numpy(s).float().to(device)).detach() #(n_env, h, w)
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

        loss_lst.append(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step_idx % PRINT_INTERVAL == 0:
            print(f'Done {step_idx} / {max_train_steps}, loss = {sum(loss_lst) / PRINT_INTERVAL}')
            loss_lst.clear()
            # test(step_idx, model, env)
    test(step_idx, model, env)

    envs.close()

