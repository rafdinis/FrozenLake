import numpy as np
import gym
import random
from Agent import *
from SelectionAlgorithms import *
from CrossoverAlgorithms import *
from MutationAlgorithms import *
import matplotlib.pyplot as plt
import itertools


class Environment:
    def __init__(self, env_name, is_slippery=False, render_mode="human"):
        self.env = gym.make(env_name, is_slippery=is_slippery)
        self.env_test = gym.make(env_name, is_slippery=is_slippery, render_mode=render_mode)

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def close(self):
        self.env.close()


class GeneticAlgorithm:
    def __init__(self, env, population_size, generations):
        self.env = env
        self.population_size = population_size
        self.generations = generations
        self.population = []
        self.max_fitness = []

    def initialize_population(self, policy_matrix=None):
        agent_population = [Agent(policy_matrix=policy_matrix) for _ in
                            range(self.population_size)]
        self.population = agent_population

    def evaluate_fitness(self, agent, metric="manhattan"):
        state = self.env.reset()
        steps = 0
        done = False
        while not done and steps < 100:
            action = agent.choose_action(state)
            new_state = self.env.step(action)
            done = new_state[2]
            state = new_state
            steps += 1

        fitness_value = agent.get_fitness(state, metric)
        agent.set_fitness(fitness_value)

    def run(self, selection=0, crossover=0, mutation=0, metric="manhattan"):
        self.initialize_population(policy_matrix=None)

        for gen in range(self.generations):
            for agent in self.population:
                self.evaluate_fitness(agent, metric)

            self.max_fitness.append(max(agent.fitness for agent in self.population))

            if selection == 0:
                # selection, crossover, and mutation operations
                self.population = tournament_algorithm(self.population)
            elif selection == 1:
                self.population = ranking_selection(self.population)
            elif selection == 2:
                self.population = roulette_wheel_selection(self.population)

            if crossover == 0:
                self.population = Single_point(self.population)
            if crossover == 1:
                self.population = Alternated_mosaic(self.population)
            if crossover == 2:
                self.population = Double_point(self.population)

            self.population = mutate_population(self.population, mutation)

        for agent in self.population:
            self.evaluate_fitness(agent, metric)

        self.max_fitness.append(max(agent.fitness for agent in self.population))

        # Return the best agent after running the GA
        return max(self.population, key=lambda agent: agent.fitness), self.max_fitness

    def plot_fitness_curve(self):
        plt.figure()
        plt.plot(range(self.generations + 1), self.max_fitness)
        plt.title("Max Fitness (1 is the goal)")
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.show()


def Base(selection=0, crossover=0, mutation=0, metric="manhattan"):
    env = Environment("FrozenLake-v1", is_slippery=False)
    ga = GeneticAlgorithm(env, population_size=10, generations=100)
    best_agent = ga.run(selection, crossover, mutation, metric)

    state = env.reset()
    steps = 0
    done = False
    while not done and steps < 100:
        action = best_agent[0].choose_action(state)
        new_state = env.step(action)
        done = new_state[2]
        state = new_state
        steps += 1

    env.close()
    return best_agent


def plot_fitness_curve(abf_list, combinations_abs, amf_list, combinations_amf, asr_list, combinations_asr, metric):
    plt.figure()
    labels_abf = []
    for idx, a in enumerate(abf_list):
        plt.plot(range(len(a)), a)
        labels_abf.append("".join(map(str, combinations_abs[idx])))

    plt.legend(labels_abf)
    plt.title("Max Fitness for "+metric)
    plt.xlabel("Generation")
    plt.ylabel("Fitness")

    # Plot median fitness
    plt.figure()

    labels_amf = []
    for idx, a in enumerate(amf_list):
        plt.plot(range(len(a)), a)
        labels_amf.append("".join(map(str, combinations_amf[idx])))

    plt.legend(labels_amf)
    plt.title("Median Fitness for "+metric)
    plt.xlabel("Generation")
    plt.ylabel("Fitness")

    # Plot success rates
    plt.figure()

    labels_asr = []
    for idx, a in enumerate(asr_list):
        plt.plot(range(len(a)), a)
        labels_asr.append("".join(map(str, combinations_asr[idx])))

    plt.legend(labels_asr)
    plt.title("Success Rates for "+metric)
    plt.xlabel("Generation")
    plt.ylabel("Success Rate")

    plt.show()


def get_top_5(algorithm_ABF, combinations):
    averages = [sum(sublist) / len(sublist) for sublist in algorithm_ABF]

    # Sort the sublists based on their average
    sorted_sublists = sorted(enumerate(algorithm_ABF), key=lambda x: averages[x[0]], reverse=True)

    # Get the top 5 sublists with their averages and indexes
    indexes_top_5 = [index for index, _ in sorted_sublists[:5]]
    abfs_top_5 = [algorithm_ABF[index] for index in indexes_top_5]
    combinations_top_5 = [combinations[index] for index in indexes_top_5]
    return abfs_top_5, combinations_top_5


def statistical_mode(runs=30, search=0, top_n=5,  metric="radial"):
    # Perform grid search if search is zero
    if search == 0:
        values = [0, 1, 2]
        combination_length = 3
        combinations = list(itertools.product(values, repeat=combination_length))

    algorithm_ABF = []
    algorithm_median_fitness = []
    algorithm_success_rates = []

    c = 0
    while c < len(combinations):
        best_fitness_of_run = []

        for _ in range(runs):
            best_agent = Base(combinations[c][0], combinations[c][1], combinations[c][2], metric)
            best_fitness_of_run.append((best_agent[1]))

        ABF = []
        median_fitness = []
        success_rates = []

        num_generations = len(best_fitness_of_run[0])

        for gen in range(num_generations):
            ABF.append(sum(run[gen] for run in best_fitness_of_run) / runs)
            fitness_values = [run[gen] for run in best_fitness_of_run]
            median_fitness.append(np.median(fitness_values))
            success_rate = sum(1 for fitness in fitness_values if fitness == 1) / runs
            success_rates.append(success_rate)

        algorithm_ABF.append(ABF)
        algorithm_median_fitness.append(median_fitness)
        algorithm_success_rates.append(success_rates)
        print(c)
        c += 1


    abs_top_5, combinations_abs_top_5 = get_top_5(algorithm_ABF, combinations)
    amf_top_5, combinations_amf_top_5 = get_top_5(algorithm_median_fitness, combinations)
    asr_top_5, combinations_asr_top_5 = get_top_5(algorithm_success_rates, combinations)

    plot_fitness_curve(abs_top_5, combinations_abs_top_5, amf_top_5, combinations_amf_top_5, asr_top_5,
                       combinations_asr_top_5, metric)


if __name__ == "__main__":
    statistical_mode(metric="manhattan")