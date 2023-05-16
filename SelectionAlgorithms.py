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

def tournament_algorithm(population, fitness_scores):
    tournament_results = []
    tournament_size = 2 

    while len(population) >= tournament_size:
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_individuals = [population[i] for i in tournament_indices]
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]

        winners = []
        max_fitness = max(tournament_fitness)
        for i, fitness in enumerate(tournament_fitness):
            if fitness == max_fitness:
                winners.append(tournament_individuals[i])

        tournament_results.append((winners, max_fitness))

        population = [individual for i, individual in enumerate(population) if i not in tournament_indices]
        fitness_scores = [score for i, score in enumerate(fitness_scores) if i not in tournament_indices]

    return tournament_results


def ranking_selection(population, fitness_scores ):

    # Sort the individuals based on their fitness scores
    sorted_population = [ind for _, ind in sorted(zip(fitness_scores, population), reverse=True)]

    # Assign ranks to individuals
    ranks = list(range(1, len(population) + 1))

    # Calculate selection probabilities based on ranks
    selection_probs = [rank / sum(ranks) for rank in ranks]

    # Generate cumulative probability distribution
    cumulative_probs = [sum(selection_probs[:i+1]) for i in range(len(population))]

    # Select individuals for reproduction
    selected_individuals = []
    for _ in range(len(population)):
        # Generate a random number between 0 and 1
        r = random.random()

        # Find the smallest rank whose cumulative probability is greater than or equal to r
        selected_rank = next(rank for rank, cum_prob in enumerate(cumulative_probs) if cum_prob >= r)

        # Select the individual with the selected rank
        selected_individuals.append(sorted_population[selected_rank])

    return selected_individuals

def roulette_wheel_selection(population, fitness_scores):
    total_fitness = sum(fitness_scores)
    probabilities = [score / total_fitness for score in fitness_scores]

    # Create a cumulative probability distribution
    cumulative_probabilities = [sum(probabilities[:i+1]) for i in range(len(probabilities))]

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
