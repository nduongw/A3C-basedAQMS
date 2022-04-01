from utils.Utils import *
import torch
import numpy as np
import os
import time
import csv
from datetime import date
from utils.Utils import calc_overlap

def calc_reward(action, cover_map, current_map, new_cover_map, Config, time):
    alpha = Config.get("alpha")
    theta = Config.get("theta")
    
    today = date.today()
    # t = time.localtime()
    # current_time = time.strftime("%H:%M:%S", t)
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
    on_reward_t =  torch.count_nonzero(on_reward).item()
    
    off_car = current_map - action
    off_cover_map = set_cover_radius(off_car, count_car(off_car))
    clone = torch.clone(off_cover_map)
    reward_car_off = off_cover_map * new_cover_map
    e = torch.where(reward_car_off > 0, 1, -1)
    off_reward_t = torch.sum(clone * e).item()

    # old_cmap = torch.where(cover_map != 0, 1, 0)
    # new_cmap = torch.where(new_cover_map != 0, 1, 0)
    # overlap = old_cmap + new_cmap
    # overlap_r = torch.count_nonzero(torch.where(overlap == 2, 1, 0)).item()
    
    spatial_overlap = calc_overlap(num_car_on) 
    # spatial_overlap /= np.sqrt(len(total_car))
    reward = (1 / (1 + len(num_car_on))) * (alpha * on_reward_t + theta*off_reward_t - (1 - alpha - theta) * spatial_overlap)

    writer.writerow((on_reward_t, off_reward_t, spatial_overlap))
    return reward/1000.0
