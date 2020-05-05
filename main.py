import sys
import math
import matplotlib.pyplot as plt

import tensorflow as tf
import gym
import cv2
import numpy as np
from gym import wrappers
import threading
import multiprocessing
from queue import Queue
import argparse

from custom_gym.doublecartpole import DoubleCartPoleEnv

import random
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


ENV_NAME = "CartPole-v0"

GAMMA = 0.95
LEARNING_RATE = 0.001

MEMORY_SIZE = 1000000
BATCH_SIZE = 200

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.998


class DQNSolver:

    def __init__(self, observation_space, action_space):
        self.exploration_rate = EXPLORATION_MAX

        self.action_space = action_space
        self.memory = deque(maxlen=MEMORY_SIZE)

        self.model = Sequential()
        self.model.add(Dense(80, input_shape=(observation_space,), activation="relu"))
        self.model.add(Dense(80, activation="relu"))
        self.model.add(Dense(40, activation="relu"))
        self.model.add(Dense(self.action_space, activation="linear"))
        self.model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE))

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def experience_replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, state_next, terminal in batch:
            q_update = reward
            if not terminal:
                q_update = (reward + GAMMA * np.amax(self.model.predict(state_next)[0]))
            q_values = self.model.predict(state)
            q_values[0][action] = q_update
            self.model.fit(state, q_values, verbose=0)
        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)
    
    def save_model(self):
        self.model.save_weights('./model/weights')

    def load_model(self):
        try:
            self.model.load_weights('./model/weights')
        except:
            pass

def cartpole():
    env = DoubleCartPoleEnv()
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    dqn_solver = DQNSolver(observation_space, action_space)
    dqn_solver.load_model()
    run = 0
    while True:
        run += 1
        state = env.reset()
        state = np.reshape(state, [1, observation_space])
        step = 0
        keep=1
        average_reward = 0
        reward_sum = 0
        while keep:
            step += 1
            env.render()
            action = dqn_solver.act(state)
            state_next, reward, terminal, info = env.step(action)
            
            state_next = np.reshape(state_next, [1, observation_space])
            dqn_solver.remember(state, action, reward, state_next, terminal)
            state = state_next
            average_reward = (average_reward * (step - 1) + reward)/step


            #print("Run: " + str(run))
            #print("step: " + str(step))
            #print("reward: " +str(reward))
            #action_str = "Left" if action == 0 else "Right"
            #print("action: " + action_str)
            #print("average_reward: " + str(average_reward))
            reward_sum += reward


            if terminal:
                print("Run: " + str(run) + ", exploration: " + str(dqn_solver.exploration_rate) + ", score: " + str(reward_sum))
                dqn_solver.experience_replay()
                keep=0
                break
        dqn_solver.save_model()


if __name__ == "__main__":
    cartpole()
