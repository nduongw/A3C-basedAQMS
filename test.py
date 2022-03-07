import yaml

from utils.Utils import *
from utils.Env import Env
from utils.Map import Map
from utils.Reward import calc_reward
from utils.A2C import ParallelEnv

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

with open('config/hyperparameter.yaml') as f:
    Config = yaml.safe_load(f)


envs = ParallelEnv(4)
s = envs.reset()
print('enviroment 1')
print(s[0])
print('\n')
print('enviroment 2')
print(s[1])
print('\n')
print('enviroment 3')
print(s[2])
print('\n')
print('enviroment 4')
print(s[3])
print('\n')
# demo = Map(Config)
# demo.create_map()

# for i in range(100):
#     x = []
#     y = []
#     for i in range(demo.map.shape[0]):
#         for j in range(demo.map.shape[1]):
#             if demo.map[i, j] == 1:
#                 # print(i, j)
#                 x.append(i)
#                 y.append(j)      
#     demo.run_per_second()
#     print(x, y)
#     print(len(x))
#     plt.scatter(y, x)
#     # map_a = init.get_cover_map()
#     # sns.heatmap(data=init.cover_map[-1], cmap='Blues')
#     plt.draw()
#     plt.cla()
#     plt.pause(0.5)
#     x.clear()
#     y.clear()
# plt.show()
