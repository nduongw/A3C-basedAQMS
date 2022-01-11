
from utils.Utils import *

class Init:
    def __init__(self, Config):
        self.Config = Config
        self.map = torch.zeros(Config.get('roadLength'), Config.get('roadWidth'))
        self.cover_map = []
        self.time = 0
    
    def create_map(self):
        self.map = generate_map(self.Config)   

    def set_cover_radius(self):
        cover_map = generate_air_quality_map(self.map, self.Config)
        self.cover_map.append(cover_map)

    def run_per_second(self):
        # if self.time == 0:
        #     cover_map = generate_air_quality_map(self.map)
        #     self.cover_map.append(cover_map)

        for i in range(self.map.shape[0] - 1, -1, -1):
            for j in range(self.map.shape[1] - 1, -1, -1):
                if self.map[i, j] == 1:
                    speed = random.randint(1, 2)
                    if i + speed > self.map.shape[0] - 1:
                        self.map[i, j] = 0
                    else: 
                        self.map[i, j] = 0
                        self.map[i + speed, j] = 1
        
        for i in range(self.map.shape[0]):
            for j in range(self.map.shape[1]):
                if self.map[i, j] == 0 and i < 2 and j not in get_coordinate_dis(self.Config):
                    a = random.random()
                    if a > 0.02:
                        self.map[i, j] = 0
                    else:
                        self.map[i, j] = 1

        # assert time > 0, 'chua co map nao duoc do'
        previous_map = self.cover_map[-1]
        previous_map -= self.Config.get('air_discount') * previous_map
        zeros_tensor = torch.zeros(previous_map.size())
        previous_map = torch.where(previous_map > 0, previous_map, zeros_tensor)
        self.cover_map.append(previous_map)
        self.time += 1
