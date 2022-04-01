import random
import numpy as np

from utils.Utils import *
from utils.Reward_v6 import calc_reward

class Map:
    def __init__(self, Config):
        self.Config = Config
        self.map = torch.zeros(Config.get('road_length'), Config.get('road_width'))
        self.cover_map = torch.zeros(Config.get('road_length'), Config.get('road_width'))
        self.stime = 0

    def create_map(self):
        self.map = generate_map(self.Config)
        # self.cover_map = generate_air_quality_map(self.map, self.Config)

    def run_per_second(self):
        for i in range(self.map.shape[0] - 1, -1, -1):
            for j in range(self.map.shape[1] - 1, -1, -1):
                if self.map[i, j] == 1:
                    # print(f'car {i} {j}')
                    speed = random.randint(self.Config.get('car_speed_min'), self.Config.get('car_speed_max'))
                    if i + speed > self.map.shape[0] - 1:
                        self.map[i, j] = 0
                    else: 
                        self.map[i, j] = 0
                        self.map[i + speed, j] = 1

        for i in range(self.map.shape[0]):
            for j in range(self.map.shape[1]):
                if self.map[i, j] == 0 and i < self.Config.get('car_spawn_idx') and j not in get_coordinate_dis(self.Config):
                    a = random.uniform(0, 1)
                    if a > self.Config.get('car_spawn_prob'):
                        self.map[i, j] = 0
                    else:
                        self.map[i, j] = 1
        
        # for i in range(Config.get('road_number')):
        #     # from pdb import set_trace
        #     # set_trace()
        #     num_car = count_car_in_road(self.map, Config.get('x_min')[i], Config.get('x_max')[i])
        #     # print(f'current number of car: {num_car}')
        #     x_delta = Config.get('x_max')[i] - Config.get('x_min')[i]
        #     y_delta = Config.get('road_length')/10

        #     area_total = x_delta * y_delta
        
        #     num_points = scipy.stats.poisson(Config.get('lambda_list')[i] * area_total).rvs()
        #     # print(f'points: {num_points}')
            
        #     if num_car < num_points:
        #         xx = x_delta * scipy.stats.uniform.rvs(0, 1, (((num_points - num_car), 1))) + Config.get('x_min')[i]
        #         yy = Config.get('car_speed_max') * scipy.stats.uniform.rvs(0, 1, (((num_points - num_car), 1)))

        #         x_idx = xx.astype(int)
        #         y_idx = yy.astype(int)
        
        #         for x, y in zip(x_idx, y_idx):
        #             self.map[y.item(), x.item()] = 1
        
        

        previous_map = self.cover_map
        previous_map -= self.Config.get('air_discount')
        zeros_tensor = torch.zeros(previous_map.size())
        after_map = torch.where(previous_map > 0, previous_map, zeros_tensor)
        self.cover_map = after_map
        self.stime += 1

        return self.map, self.cover_map

    def step(self, action):
        car_list = count_car(action)
        new_cover_map = set_cover_radius(action, car_list)
        new_cover_map = torch.where(new_cover_map > self.cover_map, new_cover_map, self.cover_map)
        reward = calc_reward(action, self.cover_map, self.map, new_cover_map, self.Config, self.stime)
        self.cover_map = new_cover_map

        return reward

    def map_to_action(self, prob_map):
        on_off_map = torch.where(prob_map > self.Config.get('action_prob'), 1, 0)
        action = self.map * on_off_map.to('cpu')
        return action

    def seed(self, seed):
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
