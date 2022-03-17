import torch 
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import yaml
import wandb

from utils.Env import Env
from utils.Utils import *
from utils.A2C import ActorCritic, ParallelEnv, compute_target

with open('config/hyperparameter.yaml') as f:
    Config = yaml.safe_load(f)

n_train_processes = Config.get("n_train_processes")
learning_rate = Config.get("learning_rate")
update_interval = Config.get("update_interval")
max_train_steps = Config.get("max_train_steps")

PRINT_INTERVAL = update_interval * 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test(step_idx, model, env):
    score = 0.0
    done = 0
    num_test = 1
    s = env.reset()
    cover_lst = []
    sent_lst = []

    for _ in range(num_test):
        while done < 150:
            prob = model.pi(torch.unsqueeze(torch.from_numpy(np.array(s)), 0).float().to(device)).detach()   #(1, h, w)
            a = env.env.map_to_action(torch.squeeze(prob)) # ( h, w)
            send_car = len(count_car(a))
            total_car = len(count_car(env.env.map))
            s_prime, r = env.step(a)
            cover_score = torch.count_nonzero(env.env.cover_map).item()
            total_score = Config.get('road_length') * Config.get('road_width')
            avg_a = (send_car / total_car) * 100
            cover_lst.append(cover_score / total_score * 100)
            sent_lst.append(avg_a)
            s = s_prime
            score += r
            done += 1

    print(f"Step :{step_idx}, avg score : {score/num_test:.1f}, avg_cover_radius : {sum(cover_lst) / len(cover_lst) : .2f}, avg_sent_package: {sum(sent_lst) / len(sent_lst) : .2f}")
    wandb.log({'avg_score': score/num_test, 'avg_cover_radius': sum(cover_lst) / len(cover_lst), 'avg_sent_package': sum(sent_lst) / len(sent_lst)})

if __name__ == '__main__':
    mp.set_start_method('spawn')
    wandb.init(project="MEC-project", entity="mec", name='no pooling, batchnorm')
    
    env = Env(Config)
    envs = ParallelEnv(n_train_processes)
    model = ActorCritic().to(device)
    optimizer = torch.optim.Adam(model.model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.Adam([{'params': model.model.parameters()}, {'params': model.model.fc_v.parameters(), 'lr': learning_rate * 5}], lr=learning_rate)
    
    step_idx = 0
    policy_loss = []
    value_loss = []
    s = envs.reset()        #(n_env, num_frame, 2, h, w)

    while step_idx < max_train_steps:
        s_list, a_list, r_list, v_list, pi_list = list(), list(), list(), list(), list()
        for _ in range(update_interval):
            prob = model.pi(torch.from_numpy(s).float().to(device)) #(n_env, h, w)
            v = model.v(torch.from_numpy(s).float().to(device)) #(n_env, 1)
            a = envs.choose_action(prob.detach()) # (n_env, h, w)
            s_prime, r = envs.step(a)   # (n_env, num_frame, 2, h, w) and (n_env, 1)

            s_list.append(s) #(update_interval, n_env, num_frame, 2, h, w)
            a_list.append(a) #(update_interval, n_env, h, w)
            r_list.append(r) #(update_interval, n_env)

            v_list.append(v) #(update_interval, n_env)
            pi_list.append(prob) #(update_interval, n_env, h, w)

            s = s_prime
            step_idx += 1
        s_final = torch.from_numpy(s_prime).float()     # (n_env, num_frame, 2, h, w)
        v_final = model.v(s_final.to(device)).to('cpu').detach().clone().numpy()     # (n_env, 1)
        td_target = compute_target(v_final, r_list)     # (update_interval, n_env)

        td_target_vec = td_target.reshape(-1).to(device)      # (update_interval*n_env) nối update_interval hàng thành 1 hàng
        # s_vec = torch.tensor(s_list).float().reshape(-1, 4)  # 4 == Dimension of state
        s_vec = torch.from_numpy(np.concatenate(s_list)).float()       #(n_env*len(s_list), num_frame, 2, h, w)
        map_car_vec = s_vec[:,-1, 0, :, :].detach().clone().to(device)   #(n_env*len(s_list), h, w)
        # a_vec = torch.tensor(a_list).reshape(-1).unsqueeze(1)
        a_vec = torch.from_numpy(np.concatenate(a_list)).float().to(device)      #(n_env*len(s_list), h, w)
        advantage = td_target_vec - torch.squeeze(torch.cat(v_list,dim=0))    #(update_interval*n_env)

        pi = torch.cat(pi_list, dim=0)
        # pi = model.pi(s_vec.to(device))        #(n_env*len(s_list), h, w)
        # pi_a = pi.gather(1, a_vec).reshape(-1)
        zeros_map = torch.zeros_like(pi)
        ones_map = torch.ones_like(pi)

        pi_on = torch.where(pi*a_vec == zeros_map, ones_map, pi*a_vec)
        pi_off = torch.where(1 - pi*(map_car_vec - a_vec) == zeros_map, ones_map, 1 - pi*(map_car_vec - a_vec))

        pi_a_on = torch.mean(torch.log(pi_on), dim=(1, 2)) #(update_interval*n_env)
        pi_a_off = torch.mean(torch.log(pi_off), dim=(1, 2)) #(update_interval*n_env)
        loss = -((pi_a_on + pi_a_off) * advantage.detach()).mean() + nn.MSELoss()(torch.squeeze(torch.cat(v_list,dim=0)), td_target_vec)

        policy_loss.append(((pi_a_on + pi_a_off) * advantage.detach()).mean())
        value_loss.append(nn.MSELoss()(torch.squeeze(torch.cat(v_list,dim=0)), td_target_vec))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step_idx % PRINT_INTERVAL == 0:
            # print(f'Done {step_idx} / {max_train_steps}, policy loss: {sum(policy_loss) / PRINT_INTERVAL}, value loss: {sum(value_loss) / PRINT_INTERVAL}')
            wandb.log({'Policy loss' : sum(policy_loss) / PRINT_INTERVAL, 'Value loss': sum(value_loss) / PRINT_INTERVAL, 'step': step_idx})
            policy_loss.clear()
            value_loss.clear()

            test(step_idx, model, env)

    envs.close()

