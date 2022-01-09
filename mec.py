from src.GnbSever import Network
from src.Init import Init
from src.Utils import *
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

init = Init() 
init.create_map()

print(init.map)
init.set_cover_radius()
x = []
y = []
sns.set_theme()
plt.figure(figsize=(10, 7))
plt.xlim([0, 70])
plt.ylim([0, 100])
# plt.margins(0)
plt.xticks(np.arange(0, 80, step=10))
plt.yticks(np.arange(0, 110, step=10))

# print(init.map)
# init.set_cover_radius()
# for i in range(2):
#     init.run_per_second()
# sns.heatmap(data=init.cover_map[-1], cmap='Blues')
# plt.show()

# plt.ion()

for i in range(50):
    x = []
    y = []
    for i in range(init.map.shape[0]):
        for j in range(init.map.shape[1]):
            if init.map[i, j] == 1:
                # print(i, j)
                x.append(i)
                y.append(j)      
    init.run_per_second()
    plt.scatter(y, x)
    # map_a = init.get_cover_map()
    # plt.draw()
    plt.pause(0.5)
    plt.cla()
    x.clear()
    y.clear()
plt.show(block=True)

# sns.heatmap(data=init.cover_map[-1], cmap='Blues')
# plt.show()
# action = torch.rand([5, 5])
# network = Network(init.map)
# new_action = network.change_on_off(action)
# print(new_action)