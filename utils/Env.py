import numpy as np
from utils.Map import Map
import torch


class Env:
    def __init__(self, config):
        self.env = Map(config)
        self.config = config

    def reset(self):
        self.env.create_map()
        obs = self.run()

        return obs

    def step(self, action):
        # action = self.env.map_to_action(prob)
        reward = self.env.step(action)
        obs = self.run()

        return obs, reward

    def run(self):
        obs = []
        obs.append(np.stack((self.env.map, self.env.cover_map)))
        for _ in range(self.config.get("num_frame") - 1):
            s, c = self.env.run_per_second()
            obs.append(np.stack((s, c)))

        return obs 
    
    def seed(self, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)

