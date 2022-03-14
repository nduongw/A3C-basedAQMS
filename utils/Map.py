import random

from utils.Utils import *
from utils.Reward_v3 import calc_reward

class Map:
    def __init__(self, Config):
        self.Config = Config
        self.map = torch.zeros(Config.get('road_length'), Config.get('road_width'))
        self.cover_map = torch.zeros(Config.get('road_length'), Config.get('road_width'))

    def create_map(self):
        self.map = generate_map(self.Config)
        self.cover_map = generate_air_quality_map(self.map, self.Config)

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
                    a = random.random()
                    # print(a)
                    if a > self.Config.get('car_spawn_prob'):
                        self.map[i, j] = 0
                    else:
                        self.map[i, j] = 1

        previous_map = self.cover_map
        previous_map -= self.Config.get('air_discount')
        zeros_tensor = torch.zeros(previous_map.size())
        after_map = torch.where(previous_map > 0, previous_map, zeros_tensor)
        self.cover_map = after_map

        return self.map, self.cover_map

    def step(self, action):
        car_list = count_car(action)
        new_cover_map = set_cover_radius(action, car_list)
        new_cover_map = torch.where(new_cover_map > self.cover_map, new_cover_map, self.cover_map)
        reward = calc_reward(action, self.cover_map, self.map, new_cover_map, self.Config)
        self.cover_map = new_cover_map

        return reward

    def map_to_action(self, prob_map):
        on_off_map = torch.where(prob_map > self.Config.get('action_prob'), 1, 0)
        action = self.map * on_off_map.to('cpu')
        return action

