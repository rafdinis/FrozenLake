from random import *
from Agent import *


def standard_mutation(population):
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