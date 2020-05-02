import gym
import math
from  gym import spaces, logger
from gym.utils import seeding
import numpy as np 
from custom_gym.models import Pole, Cart
from gym.envs.classic_control import rendering

class DoubleCartPoleEnv(gym.Env):
    """
    Description:
        A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The pendulum
        starts upright, and the goal is to prevent it from falling over by increasing and reducing the cart's velocity.
    Source:
        This environment corresponds to the version of the cart-pole problem described by Barto, Sutton, and Anderson
    Observation: 
        Type: Box(4)
        Num	Observation               Min             Max
        0	Cart Position             -10.0           10.0
        1	Cart Velocity             -Inf            Inf
        2	Pole 1 Angle              -180 deg        180 deg
        3	Pole 1 Velocity At Tip    -Inf            Inf
        4   Pole 2 Angle              -180 deg        180 deg
        5   Pole 2 Velocity At Tip    -Inf            Inf
        
    Actions:
        Type: Discrete(2)
        Num	Action
        0	Push cart to the left
        1	Push cart to the right
        
        Note: The amount the velocity that is reduced or increased is not fixed; it depends on the angle the pole is
        pointing. This is because the center of gravity of the pole increases the amount of energy needed to move the
        cart underneath it
    Reward:
        Reward is 1 for every step when poles are no more than 20 degree from straight up
    Starting State:
        All observations are assigned a uniform random value
    Episode Termination: #to change
        Cart Position is more than 100 (center of the cart reaches the edge of the display)
        Episode length is greater than 2000
    """
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.cart = Cart(10)
        self.pole_1 = Pole(3, 10, 5)
        self.pole_2 = Pole(0.5, 10, 7)
        self.gravity = 9.8
        self.time_step = 0.02
        self.force_mag = 200.0


        self.x_threshold = 100
        self.theta_threshold_radians = math.pi
        
        high = np.array([self.x_threshold * 2,
                         np.finfo(np.float32).max,
                         self.theta_threshold_radians * 2,
                         np.finfo(np.float32).max,
                         self.theta_threshold_radians * 2,
                         np.finfo(np.float32).max],
                        dtype=np.float32)

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        
        self.seed()
        self.viewer = None
        self.state = None

        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action): # Execute one time step within the environment
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        force = self.force_mag if action==1 else -self.force_mag
        self.steps += 1


        m2 = self.pole_1.mass
        theta2 = self.pole_1.theta
        l2 = self.pole_1.mass_center
        L2 = self.pole_1.length

        m3 = self.pole_2.mass
        theta3 = self.pole_2.theta
        l3 = self.pole_2.mass_center
        
        G = np.zeros(3)
        G[1] = m3 * l3 * L2 * self.pole_2.theta_dot * self.pole_2.theta_dot * math.sin(theta2 - theta3)
        G[2] = -1 * m3 * l3 * L2 * self.pole_1.theta_dot * self.pole_1.theta_dot * math.sin(theta2 - theta3)

        V = np.zeros(3)
        V[1] = -1 * ((m2 * self.gravity * l2)+(m3 * self.gravity * L2)) * math.sin(theta2)
        V[2] = -1 * m3 * self.gravity * l3 * math.sin(theta3)

        b = np.zeros(3)
        b[0] = 1.0
        b[1] = -1 * ((m2 * l2)+(m3 * L2)) * math.cos(theta2)
        b[2] = -1 * m3 * l3 * math.cos(theta3)

        F = np.zeros((3,3))
        F[0,0] = 1.0
        F[1,1] = m3 * L2 * L2 + self.pole_1.inertia
        F[1,2] = m3 * l3 * L2 * math.cos(theta2 - theta3)
        F[2,1] = F[1,2]
        F[2,2] = self.pole_2.inertia

        Finv = np.linalg.inv(F)
        
        T = Finv.dot((force /self.cart.mass * b) - V - G)

        self.cart.make_step(  -T[0], self.time_step)
        self.pole_1.make_step(T[1], self.time_step)
        self.pole_2.make_step(T[2], self.time_step)
        
        #print(T)

        #self.cart.print_info()
        #self.pole_1.print_info()
        #self.pole_2.print_info()
        
        self.state = (self.cart.x, self.cart.velocity,self.pole_1.theta,self.pole_1.theta_dot, self.pole_2.theta,self.pole_2.theta_dot)
        out_of_bounds =  self.cart.x < -self.x_threshold or self.cart.x > self.x_threshold
        out_of_bounds = bool(out_of_bounds)

        out_of_steps = bool(self.steps > 400)

        done = bool(out_of_bounds or out_of_steps)

        if not done:
            xx = math.fabs(self.cart.x)
            r1 = 1 if xx < 50 else 0
            r2 = self.pole_1.give_reward()
            r3 = self.pole_2.give_reward()

            if r1>0 and r2>1.25 and r3>1.25:
                reward = 1.0
            else:
                reward = 0
            
        else:
            if out_of_bounds:
                reward = -1.0
            else :
                reward = 0.0

        return np.array(self.state), reward, done, {}



    def reset(self):  # Reset the state of the environment to an initial state
        self.cart.reset()
        self.pole_1.reset()
        self.pole_2.reset()
        self.steps = 0
        self.reward = 0
        self.state = (self.cart.x, self.cart.velocity,self.pole_1.theta,self.pole_1.theta_dot, self.pole_2.theta,self.pole_2.theta_dot)
        return np.array(self.state)

    
  
    def render(self, mode='human', close=False): # Render the environment to the screen
        screen_width = 1600
        screen_height = 1000

        world_width = self.x_threshold*2
        scale = screen_width/world_width
        carty = 180 # TOP OF CART
        polewidth = 6.0
        polelen1 = scale * self.pole_1.length
        cartwidth = 40.0
        cartheight = 15.0

        if self.viewer is None:
            
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l,r,t,b = -cartwidth/2, cartwidth/2, cartheight/2, -cartheight/2
            axleoffset =cartheight/4.0
            cart = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)

            l,r,t,b = -polewidth/2,polewidth/2,polelen1-polewidth/2,-polewidth/2
            pole1 = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            pole1.set_color(.8,.6,.4)
            self.poletrans1 = rendering.Transform(translation=(0, axleoffset))
            pole1.add_attr(self.poletrans1)
            pole1.add_attr(self.carttrans)
            self.viewer.add_geom(pole1)

            pole2 = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            pole2.set_color(.5,.6,.4)
            self.poletrans2 = rendering.Transform(translation=(0, axleoffset))
            pole2.add_attr(self.poletrans2)
            pole2.add_attr(self.carttrans)
            self.viewer.add_geom(pole2)

            self.axle = rendering.make_circle(polewidth/2)
            self.axle.add_attr(self.poletrans1)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5,.5,.8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0,carty), (screen_width,carty))
            self.track.set_color(0,0,0)
            self.viewer.add_geom(self.track)

            self._pole_geom1 = pole1
            self._pole_geom2 = pole2

        # Edit the pole polygon vertex
        pole1 = self._pole_geom1
        l,r,t,b = -polewidth/2,polewidth/2,polelen1-polewidth/2,-polewidth/2
        pole1.v = [(l,b), (l,t), (r,t), (r,b)]

        pole2 = self._pole_geom2
        pole2.v = [(l,b), (l,t), (r,t), (r,b)]

        cartx = self.cart.x * scale+screen_width/2.0 # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans2.set_translation(-math.sin(self.pole_1.theta)*polelen1, math.cos(self.pole_1.theta)*polelen1)
        self.poletrans1.set_rotation(self.pole_1.theta)
        self.poletrans2.set_rotation(self.pole_2.theta)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')
    