import numpy as np
import random


# def elitist_selection(population, top_percent=0.3):
# Sort the population by the fitness value of each agent
#    sorted_population = sorted(population, key=lambda agent: agent.fitness, reverse=True)
# Calculate the number of top agents to select
#    num_best = int(top_percent * len(sorted_population))
# Select the top agents
#    selected_population = sorted_population[:num_best]
# Fill with the rest of the population by randomly selecting from the top agents
#    while len(selected_population) < len(population):
#        chosen_agent = random.choice(sorted_population[:num_best])
#        selected_population.append(chosen_agent)
#    return selected_population

# Tournment Selection Algorithm
def tournament_algorithm(population, elite_percent=0.1):
    # Calculate total fitness and create a list of fitness values
    fitness_sum = 0
    fitness_list = []
    for i in population:
        fitness_sum += i.fitness
        fitness_list.append(i.fitness)

    # Set the tournament size
    tournament_size = 4

    # Calculate the number of elite individuals to select
    elite_count = int(len(population) * elite_percent)

    # Initialize the selected individuals list with elite individuals
    selected = population[:elite_count]

    # Perform tournament selection for the remaining individuals
    for _ in range(elite_count, len(population)):
        # Select individuals for the tournament
        tournament = random.sample(population, tournament_size)

        # Find the fittest individual in the tournament
        # Using max() with a lambda function to compare fitness values
        winner = max(tournament, key=lambda i: i.fitness)

        # Add the winner to the selected individuals list
        selected.append(winner)

    return selected


# Ranking Selection Algorithm
def ranking_selection(population, elite_percent=0.1):
    fitness_sum = 0
    fitness_list = []
    for i in population:
        fitness_sum += i.fitness
        fitness_list.append(i.fitness)

    total_fitness = fitness_sum
    probabilities = [score / total_fitness for score in fitness_list]

    # Calculate the number of elite individuals to select
    elite_count = int(len(population) * elite_percent)

    # Create a cumulative probability distribution
    cumulative_probabilities = [sum(probabilities[:i + 1]) for i in range(len(probabilities))]

    # Initialize the selected individuals list with elite individuals
    selected_individuals = population[:elite_count]

    # Select the remaining individuals
    for _ in range(elite_count, len(population)):
        # Generate a random number between 0 and 1
        r = random.random()

        # Find the smallest rank whose cumulative probability is greater than or equal to r
        selected_rank = next(rank for rank, cum_prob in enumerate(cumulative_probabilities) if cum_prob >= r)

        # Select the individual with the selected rank
        selected_individuals.append(population[selected_rank])

    return selected_individuals


def roulette_wheel_selection(population):
    fitness_sum = 0
    fitness_list = []

    for i in population:
        fitness_sum += i.fitness
        fitness_list.append(i.fitness)

    total_fitness = fitness_sum
    probabilities = [score / total_fitness for score in fitness_list]

    # Create a cumulative probability distribution
    cumulative_probabilities = [sum(probabilities[:i + 1]) for i in range(len(probabilities))]

    selected = []
    for _ in range(len(population)):
        # Spin the roulette wheel
        spin = random.uniform(0, 1)

        # Find the selected individual based on the spin
        for i, individual in enumerate(population):
            if spin <= cumulative_probabilities[i]:
                selected.append(individual)
                break

    return selected



