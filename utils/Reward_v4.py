from utils.Utils import *
import torch
import numpy as np
import os
import time
import csv
from datetime import date

def calc_reward(action, cover_map, current_map, new_cover_map, Config):
    today = date.today()
    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)
    if not os.path.exists('log'):
        os.mkdir('log')
    if not os.path.exists(f'log/{today}'):
        os.mkdir(f'log/{today}')
    
    f = open(f'log/{today}/reward4_2.csv', 'a')
    writer = csv.writer(f)
    
    num_car_on = count_car(action)
    total_car = count_car(current_map)

    alpha = Config.get("alpha")

    reward_car_on = (new_cover_map - cover_map).sum()#/torch.count_nonzero(new_cover_map - cover_map).item()
    on_reward_t = (1 / (1 + len(num_car_on))) * reward_car_on
    
    # reward_car_off = (1 - cover_map).sum()#/(cover_map.size()[0] * cover_map.size()[1])
    # off_reward_t = (1 / (1 + len(total_car) - len(num_car_on))) * reward_car_off
    off_map = current_map - action
    off_cover_map = set_cover_radius(off_map, count_car(off_map))
    reward_car_off = (off_cover_map * new_cover_map).sum()
    off_reward_t = (1 / (1 + len(total_car) - len(num_car_on))) * reward_car_off

    # old_cmap = torch.where(cover_map != 0, 1, 0)
    # new_cmap = torch.where(new_cover_map != 0, 1, 0)
    # overlap = old_cmap + new_cmap
    # overlap_r = torch.count_nonzero(torch.where(overlap == 2, 1, 0)).item()

    spatial_overlap = len(num_car_on) * ((Config.get('cover_radius') * 2 + 1) ** 2) - torch.count_nonzero(torch.where(new_cover_map == 1, 1, 0)).item() 
    spatial_overlap /= np.sqrt(len(total_car))
    reward =  alpha * on_reward_t - (1 - alpha) * spatial_overlap

    writer.writerow((on_reward_t.item(), off_reward_t.item(), spatial_overlap))
    return reward
