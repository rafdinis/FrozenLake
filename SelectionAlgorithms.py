import numpy as np
import random


def elitist_selection(population, top_percent=0.3):
    # Sort the population by the fitness value of each agent
    sorted_population = sorted(population, key=lambda agent: agent.fitness, reverse=True)

    # Calculate the number of top agents to select
    num_best = int(top_percent * len(sorted_population))

    # Select the top agents
    selected_population = sorted_population[:num_best]

    # Fill with the rest of the population by randomly selecting from the top agents
    while len(selected_population) < len(population):
        chosen_agent = random.choice(sorted_population[:num_best])
        selected_population.append(chosen_agent)

    return selected_population


