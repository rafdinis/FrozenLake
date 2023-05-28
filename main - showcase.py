import numpy as np
import gym
import random
from Agent import *
from SelectionAlgorithms import *
from CrossoverAlgorithms import *
from MutationAlgorithms import *
import matplotlib.pyplot as plt
import itertools


class Environment:
    def __init__(self, env_name, is_slippery=False, render_mode="human"):
        self.env = gym.make(env_name, is_slippery=is_slippery, render_mode="human")
        self.env_test = gym.make(env_name, is_slippery=is_slippery, render_mode="human")

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def close(self):
        self.env.close()


def Base(selection=0, crossover=0, mutation=0, metric="manhattan"):
    env = Environment("FrozenLake-v1", is_slippery=False, render_mode="human")

    #flattened best policy matrix for showcasing a solved case
    best_agent = [2,2,1,2,2,2,1,1,2,1,1,1,1,2,2,2]

    state = env.reset()
    steps = 0
    done = False
    while not done and steps < 100:
        action = best_agent[state[0]]
        new_state = env.step(action)
        done = new_state[2]
        state = new_state
        steps += 1

    env.close()
    return best_agent

if __name__ == "__main__":
    Base(metric="euclidean")