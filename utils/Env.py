from hashlib import new
import torch
import numpy as np
from utils.Map import Map
from collections import deque
class Env:
    def __init__(self, config):
        self.env = Map(config)
        self.config = config
        self.queue = deque(maxlen=self.config.get('num_frame'))
        
    def reset(self):
        self.env.create_map()
        self.queue.clear()
        obs = self.run()
        
        return obs

    def step(self, action):
        reward = self.env.step(action)
        obs = self.run()    

        return obs, reward

    def run(self):
        if len(self.queue) != 0:
            s, c = self.env.run_per_second()
            self.queue.append(np.stack((s, c)))
        else :
            s, c = self.env.run_per_second()
            for _ in range(self.config.get('num_frame')):
                self.queue.append(np.stack((s, c)))

        obs = list(self.queue)
        return obs 
    
    def seed(self, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)

