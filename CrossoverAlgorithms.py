import numpy as np
import random
from Agent import *
from enum import IntEnum

class CrossoverEnum(IntEnum):
    SINGLE_POINT = 0
    ALTERNATED_MOSAIC = 1
    DOUBLE_POINT = 2

def apply_crossover(population, crossover_algorithm, crossover_rate=0.9):
    crossover_population = []
    
    if crossover_algorithm == CrossoverEnum.SINGLE_POINT:
        crossover_population = Single_point(population, crossover_rate)
    if crossover_algorithm == CrossoverEnum.ALTERNATED_MOSAIC:
        crossover_population = Alternated_mosaic(population, crossover_rate)
    if crossover_algorithm == CrossoverEnum.DOUBLE_POINT:
        crossover_population = Double_point(population, crossover_rate)

    return crossover_population

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

def Alternated_mosaic(population, crossover_rate=0.5):
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

            # Initialize offspring
            flat_offspring1 = np.empty_like(flat_p1p)
            flat_offspring2 = np.empty_like(flat_p2p)

            # Iterate over each element of the policy
            for i in range(len(flat_p1p)):
                if random.random() < crossover_rate:
                    flat_offspring1[i] = flat_p1p[i]
                    flat_offspring2[i] = flat_p2p[i]
                else:
                    flat_offspring1[i] = flat_p2p[i]
                    flat_offspring2[i] = flat_p1p[i]

            offspring1_policy = flat_offspring1.reshape(parent1_policy.shape)
            offspring2_policy = flat_offspring2.reshape(parent1_policy.shape)

            population[parent1].policy_matrix = offspring1_policy
            crossover_population.append(population[parent1])
            population[parent2].policy_matrix = offspring2_policy
            crossover_population.append(population[parent2])
    return crossover_population

def Double_point(population, crossover_rate=0.5):
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

            # select two random points between 0 and 15
            crossover_points = sorted(random.sample(range(0, 16), 2))

            # flatten the parents
            flat_p1p = parent1_policy.flatten()
            flat_p2p = parent2_policy.flatten()

            # create offsprings with two crossover points
            flat_offspring1 = np.concatenate((flat_p1p[:crossover_points[0]],
                                              flat_p2p[crossover_points[0]:crossover_points[1]],
                                              flat_p1p[crossover_points[1]:]))

            flat_offspring2 = np.concatenate((flat_p2p[:crossover_points[0]],
                                              flat_p1p[crossover_points[0]:crossover_points[1]],
                                              flat_p2p[crossover_points[1]:]))

            offspring1_policy = flat_offspring1.reshape(parent1_policy.shape)
            offspring2_policy = flat_offspring2.reshape(parent1_policy.shape)

            population[parent1].policy_matrix = offspring1_policy
            crossover_population.append(population[parent1])
            population[parent2].policy_matrix = offspring2_policy
            crossover_population.append(population[parent2])
    return crossover_population