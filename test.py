import yaml
import torch
import numpy as np
from random import sample

from utils.Map import Map 
from utils.Utils import calc_overlap, count_car, set_cover_radius
from utils.A2C import ActorCritic
from utils.Env import Env

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('config/hyperparameter.yaml') as f:
    Config = yaml.safe_load(f)

def test(model, env):
    score = 0.0
    done = 0
    num_test = 1
    s = env.reset()
    cover_lst = []
    sent_lst = []
    ovl_lst = []

    for _ in range(num_test):
        while done < 150:
            prob = model.pi(torch.unsqueeze(torch.from_numpy(np.array(s)), 0).float().to(device)).detach()   #(1, h, w)
            a = env.env.map_to_action(torch.squeeze(prob)) # ( h, w)
            send_car = len(count_car(a))
            total_car = len(count_car(env.env.map))
            
            car_list = count_car(a)
            new_cover_map = set_cover_radius(a, car_list)
            new_cover_map = torch.where(new_cover_map > env.env.cover_map, new_cover_map, env.env.cover_map)
            overlap = send_car * ((Config.get('cover_radius') * 2 + 1) ** 2) - torch.count_nonzero(torch.where(new_cover_map == 1, 1, 0)).item()
            overlap /= Config.get('road_length') * Config.get('road_width')
            
            s_prime, r = env.step(a)
            
            cover_score = torch.count_nonzero(env.env.cover_map).item()
            total_score = Config.get('road_length') * Config.get('road_width')
            avg_a = (send_car / total_car) * 100
            
            cover_lst.append(cover_score / total_score * 100)
            sent_lst.append(avg_a)
            ovl_lst.append(overlap)
            
            s = s_prime
            score += r
            done += 1

    print(f"avg score : {score/num_test:.1f}, avg_cover_radius : {sum(cover_lst) / len(cover_lst) : .2f}, avg_sent_package: {sum(sent_lst) / len(sent_lst) : .2f}, avg_overlap: {sum(ovl_lst) / len(ovl_lst)}")
    
    
demo_map = Map(Config)
demo_map.seed(1)
demo_map.create_map()
cover_lst = []
sent_lst = []
ovl_lst = []
for i in range(1):
    prob = (i + 4.5) * 0.1
    for _ in range(150):
        car = count_car(demo_map.map)
        action = sample(car, int(len(car) * prob))
        action_map = torch.zeros(Config.get('road_length'), Config.get('road_width'))

        for x, y in action:
            action_map[x, y] = 1
            
        demo_map.step(action_map)
        new_cover_map = set_cover_radius(action_map, action)
        overlap = len(count_car(action_map)) * ((Config.get('cover_radius') * 2 + 1) ** 2) - torch.count_nonzero(new_cover_map).item()
        demo_map.run_per_second()
        cover = torch.count_nonzero(demo_map.cover_map).item() / (Config.get('road_length') * Config.get('road_width'))
        overlap /= (Config.get('road_length') * Config.get('road_width'))
        sent = len(count_car(action_map))
        total = len(car)
        cover_lst.append(cover)
        sent_lst.append(sent / total)
        ovl_lst.append(overlap)
        print(f'cover: {cover} - overlap: {overlap} - sent_car: {len(action)} - total_car: {len(car)}')

    print(f'Prob: 0.45 : Avg cover: {sum(cover_lst) / len(cover_lst)},  Avg sent: {sum(sent_lst) / len(sent_lst)}, Avg Overlap: {sum(ovl_lst) / len(ovl_lst)}')
    print('--------------------------')

# model
print('--------------------------')
print('model results')
env = Env(Config)
model = ActorCritic().to(device)
model.load_state_dict(torch.load('trained_model_rw6_alpha_07_lr_104.pth'))
model.eval()
test(model, env)
