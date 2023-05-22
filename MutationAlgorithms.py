from random import *
from Agent import *
from enum import IntEnum

class MutationEnum(IntEnum):
    RANDOM_RESETTING = 0
    SWAP_MUTATION = 1

def mutate_population(population, mutation_type):
    
    if mutation_type == MutationEnum.RANDOM_RESETTING:
        return random_resetting(population)
    if mutation_type == MutationEnum.SWAP_MUTATION:
        return swap_mutation(population)


def random_resetting(population): 
    mutated_population = []

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

def swap_mutation(population):
    mutated_population = []

    for agent in population:
        # Copy the policy matrix
        mutated_policy = agent.policy_matrix.copy()
        print("before:", mutated_policy)
        # Random row and column
        cell1 = (randint(0, 3), randint(0, 3))
        cell2 = (randint(0, 3), randint(0, 3))

        # Replace with random int between 0 and 3
        temp = mutated_policy[cell1]
        mutated_policy[cell1] = mutated_policy[cell2]
        mutated_policy[cell2] = temp
        print("after:", mutated_policy)

        # Create a new agent with the mutated policy
        mutated_agent = Agent(policy_matrix=mutated_policy)

        # Add mutated agent to the mutated population
        mutated_population.append(mutated_agent)

    return mutated_population

def standard_mutation(individual, mutation_rate):
    mutated_individual = individual.copy()  # Create a copy of the individual

    for i in range(len(mutated_individual)):
        if random.random() < mutation_rate:  # Check if mutation should occur
            mutated_individual[i] = random.randint(0, 1)  # Mutate the gene with a random value (0 or 1)

    return mutated_individual
