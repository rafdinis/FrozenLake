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

    return manhattan_distance

def radial_distance(state):
    """
    Calculate the radial distance
    """
    # Extract the coordinates of the state
    x, y = state[0] // 4, state[0] % 4

    # Calculate the radial distance from the center 
    center_x, center_y = 3, 3
    distance = math.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

    return radial_distance

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

    def set_fitness(self, fitness_value_manhattan, fitness_value_radial, fitness_value_euclidean):
        self.fitness_manhattan = fitness_value_manhattan
        self.fitness_radial = fitness_value_radial
        self.fitness_euclidean = fitness_value_euclidean

    def get_fitness_manhattan(self, state):
        distance = manhattan_distance(state)
        fitness_value_manhattan = 1 / (1 + distance)
        return fitness_value_manhattan
    
    def get_fitness_radial(self, state):
        distance = radial_distance(state)
        fitness_value_radial = 1 / (1 + distance)
        return fitness_value_radial
    
    def get_fitness_euclidean(self, state):
        distance = euclidean_distance(state)
        fitness_value = 1 / (1 + distance)
        return fitness_value_euclidean
