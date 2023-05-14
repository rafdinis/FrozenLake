import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, mean_squared_error
import numpy as np
import random
from Agent import *



def Single_point(population, crossover_rate=0.5):
    crossover_population = []

    parent_indexadd = []
    parent_index = list(range(len(population)))
    parent2_index = parent_index.copy()

    for parent1 in parent_index:
        if parent1 in parent2_index:
            parent2_index.remove(parent1)
        if parent1 in parent_indexadd:
            continue
        elif np.random.random() < crossover_rate or len(parent2_index) <= 0:
            # add directly to the population
            crossover_population.append(population[parent1])
            # remove from the indexes this parent
        else:
            parent2 = random.choice(parent2_index)

            while parent2 == parent1:
                parent2 = random.choice(parent2_index)
            # remove the parents from the indexes
            parent_indexadd.append(parent2)
            parent2_index.remove(parent2)

            # get the matrix policy of the first parent
            parent1_policy = population[parent1].policy_matrix.copy()
            # get the matrix policy of the seconds parent
            parent2_policy = population[parent2].policy_matrix.copy()

            # select one random point between 0 and 15 (the indexes of the policy matrix)
            crossover_point = random.randrange(0, 15)

            # flatten the parents
            flat_p1p = parent1_policy.flatten()
            flat_p2p = parent2_policy.flatten()

            flat_offspring1 = np.concatenate((flat_p1p[:crossover_point], flat_p2p[crossover_point:]))
            flat_offspring2 = np.concatenate((flat_p2p[:crossover_point], flat_p1p[crossover_point:]))

            offspring1_policy = flat_offspring1.reshape(parent1_policy.shape)
            offspring2_policy = flat_offspring2.reshape(parent1_policy.shape)

            population[parent1].policy_matrix = offspring1_policy
            crossover_population.append(population[parent1])
            population[parent2].policy_matrix = offspring2_policy
            crossover_population.append(population[parent2])
    return crossover_population


def Double_point(n):

    #Get info about parent 1

    #Get info about parent 2

    #Select two random different numbers

    #v_low = is the lower value
    #v_high = is the higher value

    #Shuffle the parents to choose

    #Create a

    #return new policy matrix
    return print(n)

def Alternated_mosaic(n):

    #Get info about parent 1

    #Get info about parent 2

    #Select a random parent to be the even values and another for odd numbers


    #return new policy matrix
    return print(n)





def manhattan_distance(state):
    """
    Calculate the Manhattan distance
    """
    # Calculate the row and column indices of the current state
    row, col = state[0] // 4, state[0] % 4

    # Compute the Manhattan distance
    distance = abs(row - 3) + abs(col - 3)

    return distance



def mutate_population(population):
    crossed_population = []

    for agent in population:
        # Copy the policy matrix
        mutated_policy = agent.policy_matrix.copy()

        # Random row and column
        row_idx = randint(0, 3)
        col_idx = randint(0, 3)

        # Replace with random int between 0 and 3
        mutated_policy[row_idx, col_idx] = randint(0, 3)

        # Create a new agent with the mutated policy
        mutated_agent = Agent(policy_matrix=mutated_policy)

        # Add mutated agent to the mutated population
        mutated_population.append(mutated_agent)

    return mutated_population

