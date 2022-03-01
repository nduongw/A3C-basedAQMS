from Map import Map


class Env:
    def __init__(self, config):
        self.map = Map().create_map()
        self.cover_map = Map().set_cover_radius()

    def reset(self):
        self.state_map_1 = Map().create_map()
        self.cover_map_1 = Map().set_cover_radius()

    def step(action):
        pass

    def run(self):
        for _ in range(4):
            self.map.
    