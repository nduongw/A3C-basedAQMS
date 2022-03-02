import torch

def generate_map(Config):
    road = []

    for i in range(Config.get('road_number')):
        roadi = torch.rand(Config.get('road_length'), Config.get('road_width_list')[i])
        for u in range (roadi.size()[0]):
            for v in range(roadi.size()[1]):
                if roadi[u, v] > Config.get('generate_prob'):
                    roadi[u, v] = 1
                else : 
                    roadi[u, v] = 0
        road.append(roadi)

        if i < Config.get('road_number') - 1:
            roadi = torch.zeros(Config.get('road_length'), Config.get('road_dis_list')[i])
            road.append(roadi)

    new_road = torch.hstack([*road])
    return new_road

def count_car(road):
    count = 0
    car_list = []
    for i in range(road.shape[0]):
        for j in range(road.shape[1]):
            if road[i, j] == 1:
                count += 1
                car_list.append([i, j])
    return car_list

def set_cover_radius(road, cover_radius, car_list):
    cover_map = torch.zeros(road.shape[0], road.shape[1])

    # print(car_coord[0], car_coord[1])
    for car in car_list:
        # print(car[0], car[1])
        for i in range (max(car[0] - cover_radius, 0), 1 + min(car[0] + cover_radius, road.shape[0] - 1)):
            for j in range(max(car[1] - cover_radius, 0),1 + min(car[1] + cover_radius, road.shape[1] - 1)):
                cover_map[i, j] = 1
    return cover_map

def generate_air_quality_map(road, Config):
    car_list = count_car(road)
    cover_map = set_cover_radius(road, Config.get('cover_radius'), car_list)
    return cover_map

def get_coordinate_dis(Config):
    list_coord = []
    a = 0
    for i in range(Config.get('road_number') - 1):
        if i == 0:
            # list_coord.append(Config.roadWidthList[i])
            a += Config.get('road_width_list')[i]
            for j in range(a, a + Config.get('road_dis_list')[i]):
                list_coord.append(j)
            
        else:
            a += Config.get('road_width_list')[i] + Config.get('road_dis_list')[i - 1]
            for j in range(a, a + Config.get('road_dis_list')[i]):    
                list_coord.append(j)
    return list_coord
    