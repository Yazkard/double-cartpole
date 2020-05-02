import random2 as random
import math     


class Cart():
    def __init__(self, mass):
        self.x = 0
        self.mass = mass
        self.velocity = 0
        self.reset()
    
    def make_step(self, acceleration, time_step):
        self.x = self.x + (self.velocity*time_step)
        self.velocity = self.velocity + (acceleration*time_step)

    def reset(self):
        self.x = random.uniform(-20.0, 20.0)
        self.velocity = random.uniform(-0.2, 0.2)

    def print_info(self):
        print("x:", end =" ") 
        print(self.x, end =" ")
        print("vel:", end =" ") 
        print(self.velocity) 

class Pole():
    def __init__(self, mass, length, mass_center):
        self.theta = 0
        self.mass = mass
        self.theta_dot = 0
        self.length = length
        self.mass_center = mass_center
        self.inertia = mass * mass_center * mass_center
    
    def make_step(self, angular_acceleration, time_step):
        self.theta = self.theta + (self.theta_dot*time_step)
        self.theta_dot = self.theta_dot + (angular_acceleration*time_step)
        self.make_in_bounds()

    def make_in_bounds(self):
        if self.theta>math.pi:
            self.theta = self.theta - (2 * math.pi)
        if self.theta<-math.pi:
            self.theta = self.theta + (2 * math.pi)

    def reset(self):
        self.theta = math.pi + random.uniform(-0.10, 0.10)      #   math.pi   #  random.uniform(-math.pi, math.pi)
        self.make_in_bounds()
        self.theta_dot = 0       #random.uniform(-0.1, 0.1)

    def print_info(self):
        print("theta:", end =" ") 
        print(self.theta, end =" ")
        print("vel:", end =" ") 
        print(self.theta_dot) 

    def give_reward(self):
        x = math.fabs(self.theta)
        return -x + (math.pi/2)
