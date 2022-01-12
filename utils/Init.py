from utils.Utils import *
import random

class Init:
    def __init__(self, Config):
        self.Config = Config
        self.map = torch.zeros(Config.get('roadLength'), Config.get('roadWidth'))
        self.cover_map = []
        self.time = 0
    
    def create_map(self):
        self.map = generate_map(self.Config)   
        cover_map = generate_air_quality_map(self.map, self.Config)
        self.cover_map.append(cover_map)
        
        return list(cover_map, self.map)

    def run_per_second(self):
        # cho xe di chuyen
        for i in range(self.map.shape[0] - 1, -1, -1):
            for j in range(self.map.shape[1] - 1, -1, -1):
                if self.map[i, j] == 1:
                    speed = random.randint(1, 2)
                    if i + speed > self.map.shape[0] - 1:
                        self.map[i, j] = 0
                    else: 
                        self.map[i, j] = 0
                        self.map[i + speed, j] = 1
        
        #tao xe ngau nhien 
        for i in range(self.map.shape[0]):
            for j in range(self.map.shape[1]):
                if self.map[i, j] == 0 and i < 2 and j not in get_coordinate_dis(self.Config):
                    a = random.random()
                    if a > 0.02:
                        self.map[i, j] = 0
                    else:
                        self.map[i, j] = 1

        previous_map = self.cover_map[-1]
        previous_map -= self.Config.get('air_discount') * previous_map
        zeros_tensor = torch.zeros(previous_map.size())
        previous_map = torch.where(previous_map > 0, previous_map, zeros_tensor)
        self.cover_map.append(previous_map)
        self.time += 1
    
    def step(self, prob_map):
        on_off_map = torch.where(prob_map > self.Config.get('action_prob'), 1, 0)
        num_car = count_car(on_off_map)
        action_map = set_cover_radius(on_off_map, self.Config.get('action_range'), num_car)
        action = torch.where(self.init_map == action_map, self.init_map, 0)
        return action
    
    def recalc_cover_map(self, prob_map):
        action = self.step(prob_map)
        new_map = torch.zeros(self.Config.get('roadLength'), self.Config.get('roadWidth'))
        cover_map = set_cover_radius(new_map, self.Config.get('cover_radius'), action)
        new_cover_map = torch.where(self.map > cover_map, self.map, cover_map)
        self.cover_map.append(new_cover_map)

        return list(new_cover_map, self.map)

