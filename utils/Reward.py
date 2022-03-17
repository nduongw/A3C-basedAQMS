from utils.Utils import *
import torch

def calc_reward(action, cover_map, current_map, new_cover_map, Config):
    num_car_on = count_car(action)
    total_car = count_car(current_map)

    alpha = Config.get("alpha")
    beta = Config.get("beta")
    theta = Config.get("theta")

    reward_time_on = (new_cover_map - cover_map).sum()/torch.count_nonzero(new_cover_map - cover_map).item()
    reward_time_off = (1 - cover_map).sum()/(cover_map.size()[0] * cover_map.size()[1])
    reward_packet = len(num_car_on) / len(total_car)
    reward_overlap = new_cover_map.sum()/action.sum()

    reward = alpha*reward_time_on - beta*reward_time_off - theta*reward_packet - (1 - alpha - beta - theta)*reward_overlap

    return reward
