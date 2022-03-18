import yaml
import torch
import random

from utils.Map import Map 
from utils.Utils import calc_overlap, count_car, set_cover_radius

with open('config/hyperparameter.yaml') as f:
    Config = yaml.safe_load(f)

demo_map = Map(Config)
demo_map.create_map()
cover_lst = []
sent_lst = []
for _ in range(150):
    car = count_car(demo_map.map)
    action_map = torch.zeros(Config.get('road_length'), Config.get('road_width'))
    for x, y in car:
        rand = random.uniform(0, 1)
        if rand > 0.4:
            action_map[x, y] = 1
        else :
            action_map[x, y] = 0
    demo_map.step(action_map)
    
    overlap = len(count_car(action_map)) * (Config.get('cover_radius') * 2 + 1) ** 2 - torch.count_nonzero(torch.where(demo_map.cover_map == 1, 1, 0)).item()
    demo_map.run_per_second()
    cover = torch.count_nonzero(demo_map.cover_map).item() / (Config.get('road_length') * Config.get('road_width'))
    overlap /= (Config.get('road_length') * Config.get('road_width'))
    sent = len(count_car(action_map))
    total = len(car)
    cover_lst.append(cover)
    sent_lst.append(sent / total)
    print(f'cover: {cover} - overlap: {overlap} - total_sent: {sent / total}')
# car = count_car(demo_map.map)
# action_map = torch.zeros(Config.get('road_length'), )
# for x, y in car:
#     pass
print('--------------------------')
print(f'Avg cover: {sum(cover_lst) / len(cover_lst)},  Avg sent: {sum(sent_lst) / len(sent_lst)}')