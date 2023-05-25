import pandas as pd
import numpy as np
import math


def manhattan_distance(state):
    """
    Calculate the Manhattan distance
    """
    # Calculate the row and column indices of the current state
    row, col = state[0] // 4, state[0] % 4

    # Compute the Manhattan distance
    distance = abs(row - 3) + abs(col - 3)

    return distance

def radial_distance(state):
    """
    Calculate the radial distance
    """
    # Extract the coordinates of the state
    x, y = state[0] // 4, state[0] % 4

    if x == 0 or y == 0:
        distance = 3
    elif x == 1 or y == 1:
        distance = 2
    elif x == 2 or y == 2:
        distance = 1
    else:
        distance = 0

    return distance

def euclidean_distance(state):
    """
    Calculate the Euclidean distance
    """
    # Extract the coordinates of the state
    x, y = state[0] // 4, state[0] % 4

    # Calculate the Euclidean distance from the target position
    target_x, target_y = 3, 3
    distance = math.sqrt((x - target_x) ** 2 + (y - target_y) ** 2)

    return distance


class Agent:
    def __init__(self, policy_matrix=None, initRandom=False):
        if policy_matrix is None:
            self.policy_matrix = np.random.randint(4, size=(4, 4))
        else:
            self.policy_matrix = policy_matrix
        self.fitness = None

    def choose_action(self, state):
        row, col = state[0] // 4, state[0] % 4
        return self.policy_matrix[row, col]

    def set_fitness(self, fitness_value):
        self.fitness = fitness_value

    def get_fitness(self, state, metric="manhattan"):
        if metric == 'manhattan':
            distance = manhattan_distance(state)
        elif metric == 'radial':
            distance = radial_distance(state)
        elif metric == 'euclidean':
            distance = euclidean_distance(state)
        else:
            raise ValueError("Invalid metric. Please choose 'manhattan', 'radial', or 'euclidean'.")

        fitness_value = 1 / (1 + distance)
        return fitness_value