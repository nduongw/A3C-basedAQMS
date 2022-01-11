import torch
from utils.Utils import count_car, set_cover_radius
# from Init import Init

class Network:
    def __init__(self, init_map, Config):
        self.init_map = init_map
        self.action = []
        self.Config = Config
    
    def change_on_off(self, prob_map):
        action_map = torch.where(prob_map > self.Config.get('action_prob'), 1, 0)
        num_action = count_car(action_map)
        set_cover_radius(action_map, self.Config.get('action_range'), num_action)
        action = torch.where(self.init_map == action_map, self.init_map, 0)
        self.action.append(action)
        return action

    def recalc_cover_map(self):
        new_map = torch.zeros(self.Config.get('roadLength'), self.Config.get('roadWidth'))
        cover_map = set_cover_radius(new_map, self.Config.get('cover_radius'), self.action[-1])
        new_cover_map = torch.where(self.init_map.map > cover_map, self.init_map.map, cover_map)
        self.init_map.cover_map.append(new_cover_map)

