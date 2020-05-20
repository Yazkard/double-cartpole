import random2 as random
import numpy as np
import math     

def direction(x):
    if x >= 0:
        return 1
    else:
        return -1

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
        self.x = random.uniform(-5.0, 5.0)
        self.velocity = random.uniform(-0.0, 0.0)

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
        ok = False
        while not ok:
            if self.theta<=math.pi and self.theta>-math.pi:
                ok = True
            else:
                if self.theta>math.pi:
                    self.theta = self.theta - (2 * math.pi)
                elif self.theta<=-math.pi:
                    self.theta = self.theta + (2 * math.pi)

    def reset(self):
        self.theta = np.random.choice((0,1))* math.pi + random.uniform(-0.5, 0.5) #np.random.choice((0,1))* math.pi + random.uniform(-0.5, 0.5) #random.uniform(-math.pi, math.pi)#0  #random.uniform(-0.10, 0.10) #+ math.pi      #   math.pi   #  random.uniform(-math.pi, math.pi)
        self.make_in_bounds()
        self.theta_dot = 0       #random.uniform(-0.1, 0.1)

    def print_info(self):
        print("theta:", end =" ") 
        print(self.theta, end =" ")
        print("vel:", end =" ") 
        print(self.theta_dot) 

    def give_reward(self):
        th = self.theta
        th_dot = self.theta_dot
        c = math.cos(self.theta/2)
        pos_reward = (1/((th*th)+0.3))
        if math.fabs(th_dot)>20:
            vel_reward = 1000   #minus points - we dont want it to spin super fast
        else:
            vel_reward = ((2*(c*c))-1) * math.fabs(th_dot)
        return pos_reward, vel_reward
