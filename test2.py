from custom_gym.doublecartpole import DoubleCartPoleEnv
import time

Env = DoubleCartPoleEnv()
while 1:
    Env.step(1)
    Env.render()
    time.sleep(Env.time_step)
