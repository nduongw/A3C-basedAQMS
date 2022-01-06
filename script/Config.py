class Config:
    # chu ki do chat luong khong khi
    cycleTime = 2
    # khoang thoi gian chat luong khong khi thay doi
    changingTime = 3600
    # do bao phu cua xe
    cover_radius = 3

    action_prob = 0.5
    action_range = 2
    #do dai quang duong
    roadLength = 5
    # roadLength = 100
    # roadWidth = 70
    roadWidth = 5
    #so con duong
    roadNumber = 3
    generate_prob = 0.5

    #list chieu rong cua tung con duong
    roadWidthList = [1, 1, 1]
    #list chieu trong cua le duong
    roadDisList = [1, 1]

    frame_num = 4
    #do giam su chac chan cua khong khi trong vung duoc bao phu
    air_discount = 0.2

    simulation_time = 1000