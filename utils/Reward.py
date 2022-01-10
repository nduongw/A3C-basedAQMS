from Utils import *
import torch

def calc_reward(car_action, air_map, car_map):
    car_num = count_car(car_action)
    total_car = count_car(car_map)
    new_air_map = generate_air_quality_map(car_action)

    new_state = torch.rand([air_map.shape[0], air_map.shape[1]])
    for i in range(new_state.shape[0]):
        for j in range(new_state.shape[1]):
            new_state[i, j] = max(new_air_map[i, j], air_map[i, j])

    do_bao_phu = new_state - air_map - (1 - air_map)
    so_goi_tin = car_num / total_car

    reward = do_bao_phu - so_goi_tin

    return reward, new_state
