import numpy as np
import gym
import random
from Agent import *
from SelectionAlgorithms import *
from CrossoverAlgorithms import *
from MutationAlgorithms import *


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
        self.population = self.initialize_population()

    def initialize_population(self):
        agent_population = [Agent() for _ in range(self.population_size)]
        return agent_population

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

    def run(self):
        for gen in range(self.generations):
            for agent in self.population:
                self.evaluate_fitness(agent)

            # selection, crossover, and mutation operations
            self.population = elitist_selection(self.population)
            self.population = Alternated_mosaic(self.population)
            self.population = mutate_population(self.population, MutationEnum.SWAP_MUTATION)

        for agent in self.population:
            self.evaluate_fitness(agent)

        # Return the best agent after running the GA
        return max(self.population, key=lambda agent: agent.fitness)


def test():
    env = Environment("FrozenLake-v1", is_slippery=False)
    ga = GeneticAlgorithm(env, population_size=10, generations=100)
    best_agent = ga.run()

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


    env.close()


if __name__ == "__main__":
    test()

