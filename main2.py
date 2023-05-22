import numpy as np
import gym
import random
from Agent import *
from SelectionAlgorithms import *
from CrossoverAlgorithms import *
from MutationAlgorithms import *
import matplotlib.pyplot as plt

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
        self.initialize_population()
        self.max_fitness = []

    def initialize_population(self, initRandom=False):
        agent_population = [Agent(initRandom=initRandom) for _ in range(self.population_size)]
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
            self.initialize_population(initRandom=True)

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
        return max(self.population, key=lambda agent: agent.fitness)

    def plot_fitness_curve(self):
        plt.figure()
        plt.plot(range(self.generations + 1), self.max_fitness)
        plt.title("Max Fitness (1 is the goal)")
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.show()


def test():
    env = Environment("FrozenLake-v1", is_slippery=False)
    ga = GeneticAlgorithm(env, population_size=10, generations=200)
    best_agent = ga.run(selection=0, crossover=0, mutation=2)

    state = env.reset()
    steps = 0
    done = False
    while not done and steps < 100:
        action = best_agent.choose_action(state)
        new_state = env.step(action)
        done = new_state[2]
        state = new_state
        steps += 1

    print(best_agent.policy_matrix)
    print(best_agent.fitness)

    ga.plot_fitness_curve()

    env.close()


if __name__ == "__main__":
    test()