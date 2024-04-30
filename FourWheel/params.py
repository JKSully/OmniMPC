class Params():
    def __init__(self):
        self.m = 1.0  # Mass
        self.d = 0.5  # Distance between Wheels?
        self.r = 0.0413  # wheel radius
        self.wheel_speed_limit = 2.0  # rad /s
        self.time_horizon = 1.0
        self.sample_time = 0.1
        self.N = 21
        self.num_states = 3
        self.num_inputs = 4
        self.Tf = 5.0
