from utils.Utils import *
import torch
import numpy as np
import os
import time
import csv
from datetime import date

def calc_reward(action, cover_map, current_map, new_cover_map, Config):
    alpha = Config.get("alpha")
    beta = Config.get("beta")
    
    today = date.today()
    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)
    if not os.path.exists('log'):
        os.mkdir('log')
    if not os.path.exists(f'log/{today}'):
        os.mkdir(f'log/{today}')
    
    f = open(f'log/{today}/reward6.1.csv', 'a')
    writer = csv.writer(f)
    
    num_car_on = count_car(action)
    total_car = count_car(current_map)

    difference = new_cover_map - cover_map
    on_reward = torch.where(difference == 1, 1, 0)
    # on_reward_t = (1 / (1 + len(num_car_on))) * torch.count_nonzero(on_reward).item()
    on_reward_t = torch.count_nonzero(on_reward).item()
    
    off_map = current_map - action
    off_cover_map = set_cover_radius(off_map, count_car(off_map))
    clone = torch.clone(off_cover_map)
    reward_car_off = off_cover_map * new_cover_map
    d = torch.where(reward_car_off > 0, 1, 0)
    e = torch.where(clone > reward_car_off, -1, 1)
    off_reward_t = torch.sum(clone * e).item()

    # old_cmap = torch.where(cover_map != 0, 1, 0)
    # new_cmap = torch.where(new_cover_map != 0, 1, 0)
    # overlap = old_cmap + new_cmap
    # overlap_r = torch.count_nonzero(torch.where(overlap == 2, 1, 0)).item()
    
    spatial_overlap = len(num_car_on) * ((Config.get('cover_radius') * 2 + 1) ** 2) - torch.count_nonzero(torch.where(new_cover_map == 1, 1, 0)).item() 
    # spatial_overlap /= np.sqrt(len(total_car))
    reward = (1 / (1 + len(num_car_on))) * (alpha * on_reward_t - (1 - alpha) * spatial_overlap)

    writer.writerow((on_reward_t, off_reward_t, spatial_overlap))
    return reward
