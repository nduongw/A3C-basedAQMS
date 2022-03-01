from Utils import *
import torch

def calc_reward(action, cover_map, map, cover_map_prime):
    car_num = count_car(action)
    total_car = count_car(map)

    new_state = torch.rand([cover_map.shape[0], cover_map.shape[1]])
    for i in range(new_state.shape[0]):
        for j in range(new_state.shape[1]):
            new_state[i, j] = max(cover_map_prime[i, j], cover_map[i, j])

    total_cover = new_state - cover_map - (1 - cover_map)
    total_packet = car_num / total_car

    reward = total_cover - total_packet

    return reward
