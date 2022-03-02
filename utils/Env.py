import numpy as np
from utils.Map import Map

class Env:
    def __init__(self, config):
        self.env = Map(config)

    def reset(self):
        self.env.create_map()

    def step(self, prob):
        action = self.env.map_to_action(prob)
        reward = self.env.step(action)
        obs = self.run()

        return obs, reward

    def run(self):
        obs = []
        for _ in range(4):
            s, c = self.env.run_per_second()
            obs.append(np.stack(s, c))
        
        return obs