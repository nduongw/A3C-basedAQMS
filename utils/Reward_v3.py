from utils.Utils import *
import torch
import numpy as np

def calc_reward(action, cover_map, present_map, new_cover_map, Config):
    num_car_on = count_car(action)
    total_car = count_car(present_map)

    alpha = Config.get("alpha")
    beta = Config.get("beta")

    reward_car_on = (new_cover_map - cover_map).sum()#/torch.count_nonzero(new_cover_map - cover_map).item()
    on_reward_t = (len(total_car) / (1 + len(num_car_on))) * reward_car_on
    
    # reward_car_off = (1 - cover_map).sum()#/(cover_map.size()[0] * cover_map.size()[1])
    # off_reward_t = (1 / (1 + len(total_car) - len(num_car_on))) * reward_car_off
    off_map = present_map - action
    off_cover_map = set_cover_radius(off_map, count_car(off_map))
    reward_car_off = (off_cover_map * new_cover_map).sum()
    off_reward_t = (len(total_car) / (1 + len(total_car) - len(num_car_on))) * reward_car_off

    # old_cmap = torch.where(cover_map != 0, 1, 0)
    # new_cmap = torch.where(new_cover_map != 0, 1, 0)
    # overlap = old_cmap + new_cmap
    # overlap_r = torch.count_nonzero(torch.where(overlap == 2, 1, 0)).item()

    spatial_overlap = len(num_car_on) * ((Config.get('cover_radius') * 2 + 1) ** 2) - torch.count_nonzero(torch.where(new_cover_map == 1, 1, 0)).item() 
    # reward =  alpha * on_reward_t + beta * off_reward_t - (1 - alpha - beta) * spatial_overlap
    reward =  alpha * off_reward_t  - (1 - alpha - beta) * spatial_overlap

    return reward
