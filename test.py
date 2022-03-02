import yaml

from utils.Utils import *
from utils.Env import Env
from utils.Map import Map
from utils.Reward import calc_reward

with open('config/hyperparameter.yaml') as f:
    Config = yaml.safe_load(f)

demo = Map(Config)
s0, _ = demo.create_map()
s1, _ = demo.run_per_second()
print(s0)
print('\n')
# print(c0)
# print('\n')
print(s1)
# print('\n')
# print(c1)