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

    def initialize_population(self, policy_matrix=None, initRandom=False):
        agent_population = [Agent(policy_matrix=policy_matrix, initRandom=initRandom) for _ in range(self.population_size)]
        self.population = agent_population

    def evaluate_fitness(self, agent):
        state = self.env.reset()
        steps = 0
        done = False
        while not done and steps < 100:
            action = agent.choose_action(state)
            new_state = self.env.step(action)
            done = new_state[2]
            state = new_state
            steps += 1

        fitness_value = agent.get_fitness(state)
        agent.set_fitness(fitness_value)

    def run(self, selection = 0, crossover = 0, mutation = 0):
        if mutation == MutationEnum.SWAP or mutation == MutationEnum.SCRAMBLE:
            self.initialize_population(policy_matrix=None, initRandom=True)
        else:
            self.initialize_population(policy_matrix=None, initRandom=False)

        for gen in range(self.generations):
            for agent in self.population:
                self.evaluate_fitness(agent)

            self.max_fitness.append(max(agent.fitness for agent in self.population))

            if selection == 0:
            # selection, crossover, and mutation operations
                self.population = elitist_selection(self.population)
            elif selection == 1:
                self.population = elitist_selection(self.population)
            elif selection == 2:
                self.population = elitist_selection(self.population)

            if crossover == 0:
                self.population = Single_point(self.population)
            if crossover == 1:
                self.population = Alternated_mosaic(self.population)
            if crossover == 2:
                self.population = Double_point(self.population)

            self.population = mutate_population(self.population, mutation)

        for agent in self.population:
            self.evaluate_fitness(agent)

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


def Base(selection=0, crossover=0, mutation=1):
    env = Environment("FrozenLake-v1", is_slippery=False)
    ga = GeneticAlgorithm(env, population_size=10, generations=100)
    best_agent = ga.run(selection, crossover, mutation)

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


def plot_fitness_curve(ABFlist):
    plt.figure()
    for a in ABFlist:
        plt.plot(range(len(a)), a)
        plt.legend(a)
    plt.title("Max Fitness (1 is the goal)")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.show()


def statistical_mode(runs=30, search=0):

    #Perform grid search if search is zero
    if search == 0:
        values = [0, 1, 2]
        combination_length = 3
        combinations = list(itertools.product(values, repeat=combination_length))

    algorithm_ABF = []

    c = 0
    while c < len(combinations):
        best_fitness_of_run = []

        for _ in range(runs):
            best_agent = Base(combinations[c][0], combinations[c][1], combinations[c][2])
            best_fitness_of_run.append((best_agent[1]))

        ABF = []
        num_generations = len(best_fitness_of_run[0])

        for gen in range(num_generations):
            ABF.append(sum(run[gen] for run in best_fitness_of_run) / runs)
        algorithm_ABF.append(ABF)
        c += 1

    plot_fitness_curve(algorithm_ABF)


if __name__ == "__main__":
    statistical_mode()