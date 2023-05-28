# FrozenLake

Genetic Algorithm implementation of solutions for the FrozenLake problem, provided by the OpemAI Gym library.

## Running modes

- Main1.py: In this mode, the best combination was selected and set with a population of 100.

To change the fitness:

    statistical_mode(metric="euclidean") #change to "radial" or "manhattan" to use another distance - line 224

To change the slippery mode:

    env = Environment("FrozenLake-v1", is_slippery=False) #change to True to apply the slippery mode - line 98

- Main2.py: In this mode, the code performs the grid_search for all the combinations of algorithms present in this project.

To change the fitness:

    statistical_mode(metric="euclidean") #change to "radial" or "manhattan" to use another distance - line 227

To change the slippery mode:

    env = Environment("FrozenLake-v1", is_slippery=False) #change to True to apply the slippery mode - line 98

- main - showcase.py: In this mode, the goal is to see an example of a solution being shown in a rendered way


## Overview

This problem consists of an agent in a grid maze and trying to reach the goal without falling into a hole. 
The agent moves are:
- Left: 0
- Down: 1
- Right: 2
- Up: 3

We define a population of these different agents that have an internal representation of the map (a policy_matrix)
We then run combinations of selection, crossover, and mutation algorithms (3 of each) over different generations to observe which are the highest performers.

### Selection Algorithms

We used Tournament, Ranking, and Roulette Wheel selection algorithms, with Elitism being an option that is incorporated into the functions.

### Crossover

We used Single point, Double Point, and Uniform, with the crossover rate of 0.8

### Mutation

When it comes to mutations, we've used three mutation types: Random Resetting, Swap, and Scramble. We applied also a 0.2 mutation rate, which allows the algorithm to converge in less generations.
