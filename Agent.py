import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, mean_squared_error
import numpy as np
import random


def manhattan_distance(state):
    """
    Calculate the Manhattan distance
    """
    # Calculate the row and column indices of the current state
    row, col = state[0] // 4, state[0] % 4

    # Compute the Manhattan distance
    distance = abs(row - 3) + abs(col - 3)

    return distance


class Agent:
    def __init__(self, policy_matrix=None, initRandom=False):
        if policy_matrix is None:
            if initRandom:
                self.policy_matrix = np.random.randint(4, size=(4, 4))
            else:
                self.policy_matrix = np.zeros((4, 4), dtype=int)
        else:
            self.policy_matrix = policy_matrix
        self.fitness = None

    def choose_action(self, state):
        row, col = state[0] // 4, state[0] % 4
        return self.policy_matrix[row, col]

    def set_fitness(self, fitness_value):
        self.fitness = fitness_value

    def get_fitness(self, state):
        distance = manhattan_distance(state)
        fitness_value = 1 / (1 + distance)
        return fitness_value
