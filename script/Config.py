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
    # roadLength = 5
    roadLength = 100
    roadWidth = 70
    # roadWidth = 5
    #so con duong
    roadNumber = 5
    generate_prob = 0.85

    #list chieu rong cua tung con duong
    roadWidthList = [10, 10, 10, 10, 10]
    #list khoang cach giua cac duong
    roadDisList = [5, 5, 5, 5]

    frame_num = 4
    #do giam su chac chan cua khong khi trong vung duoc bao phu
    air_discount = 0.2

    simulation_time = 1000