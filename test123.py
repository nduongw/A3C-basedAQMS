import torch
import yaml

with open('config/hyperparameter.yaml') as f:
    Config = yaml.safe_load(f)
    
is_road = torch.zeros(Config.get('road_length'), Config.get('road_width'))
x_start = 0
x_end = Config.get('road_width_list')[0]
is_road[:, x_start:x_end] = 1

for i in range(len(Config.get('road_dis_list'))):
    x_start = x_end + Config.get('road_dis_list')[i]
    x_end = x_start + Config.get('road_width_list')[i+1]
    is_road[:, x_start:x_end] = 1

    
# print(Config.get('road_length'))
# print(Config.get('road_width'))
print(torch.count_nonzero(is_road))
print(is_road[:,80])

