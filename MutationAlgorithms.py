from random import *
from Agent import *
from enum import IntEnum

class MutationEnum(IntEnum):
    RANDOM_RESETTING = 0
    SWAP = 1
    SCRAMBLE = 2

def mutate_population(population, mutation_type):
    mutated_population = []
    
    for agent in population:
        # Copy the policy matrix
        policy = agent.policy_matrix.copy()

        if mutation_type == MutationEnum.RANDOM_RESETTING:
            mutated_population.append(random_resetting(policy))
        if mutation_type == MutationEnum.SWAP:
            mutated_population.append(swap_mutation(policy))
        if mutation_type == MutationEnum.SCRAMBLE:
            mutated_population.append(swap_mutation(policy))

    return mutated_population


def random_resetting(policy): 
    # Random row and column
    row_idx = randint(0, 3)
    col_idx = randint(0, 3)

    # Replace with random int between 0 and 3
    policy[row_idx, col_idx] = randint(0, 3)

    # Create a new agent with the mutated policy
    mutated_agent = Agent(policy_matrix=policy)

    return mutated_agent

def swap_mutation(policy):
    # Random cells
    cell1 = (randint(0, 3), randint(0, 3))
    cell2 = (randint(0, 3), randint(0, 3))

    # Replace with random int between 0 and 3
    temp = policy[cell1]
    policy[cell1] = policy[cell2]
    policy[cell2] = temp

    # Create a new agent with the mutated policy
    mutated_agent = Agent(policy_matrix=policy)

    return mutated_agent

def scramble_mutation(policy):

    # two random rows
    rowIndex1 = randint(0, 3)
    rowIndex2 = randint(0, 3)
    row1 = policy[rowIndex1]
    row2 = policy[rowIndex2]

    policy[rowIndex1] = np.random.shuffle(row1)
    policy[rowIndex2] = np.random.shuffle(row2)

    # Create a new agent with the mutated policy
    mutated_agent = Agent(policy_matrix=policy)

    return mutated_agent

def standard_mutation(individual, mutation_rate):
    mutated_individual = individual.copy()  # Create a copy of the individual

    for i in range(len(mutated_individual)):
        if random.random() < mutation_rate:  # Check if mutation should occur
            mutated_individual[i] = random.randint(0, 1)  # Mutate the gene with a random value (0 or 1)

    return mutated_individual
