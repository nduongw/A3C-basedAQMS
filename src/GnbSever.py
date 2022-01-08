import torch
from src.Utils import count_car, set_cover_radius
from script.Config import Config
# from Init import Init

class Network:
    def __init__(self, init_map):
        self.init_map = init_map
        self.action = []
    
    def change_on_off(self, prob_map):
        action_map = torch.where(prob_map > Config.action_prob, 1, 0)
        num_action = count_car(action_map)
        set_cover_radius(action_map, Config.action_range, num_action)
        action = torch.where(self.init_map == action_map, self.init_map, 0)
        self.action.append(action)
        return action

    def recalc_cover_map(self):
        new_map = torch.zeros(Config.roadLength, Config.roadWidth)
        cover_map = set_cover_radius(new_map, Config.cover_radius, self.action[-1])
        new_cover_map = torch.where(self.init_map.map > cover_map, self.init_map.map, cover_map)
        self.init_map.cover_map.append(new_cover_map)

