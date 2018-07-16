from game.GOLAI.arena import Arena
import numpy as np
import random
from numba import jit
# from numba import jitclass
import numba
import argparse

BOARD_DIMS = (512, 512)
GRID_DIMS = (64,64)

# types = [
#     ('population_size', numba.int32),
#     ('mutation_size', numba.int32),
#     ('mutation_rate', float32),
# ]

# @jitclass(types)
class EvolutionaryBenchmark:
    def __init__(self, population_size, mutation_rate):
        if population_size % 2 is not 0:
            raise ValueError("Population size must be even")
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.population = None
        self.fitness_history = []

    def random_initialization(self):
        self.population = []
        for _ in range(self.population_size):
            self.population.append(np.random.randint(2, size=GRID_DIMS))

    def do_n_cycles(self, n, num_steps=1000):
        ''' Runs n evolution cycles, made of (population_size / 2) simultaneous
        rounds, each of them performing num_steps.
        If num_steps is set to 'gradual', the number of steps is increased at
        each cycle. '''
        # Initialize if it hasn't been done already
        if self.population is None:
            self.random_initialization()
        
        for i in range(n):
            if num_steps == 'gradual':
                num_steps = min(int(2 + 1.0001 ** i), 1000)
            # Perform one cycle of the Game of life and evolve
            self.do_one_cycle(num_steps)
            self.assess_fitness()
            selection = self.select()
            next_generation = [self.population[idx] for idx in selection]
            ## skipping crossover for now ##
            self.population = self.mutate(next_generation)
    
    @jit(cache=True)
    def do_one_cycle(self, num_steps=1000):
        ''' Creates a Game of Life and performs num_steps '''
        self.games = []

        for pair in self.make_pairs(self.population):
            game = Arena(BOARD_DIMS[0], BOARD_DIMS[1])
            game.add_players(pair[0], pair[1])
            game.setup()
            game.run_steps(num_steps)
            self.games.append(game)

    # @jit(cache=True)
    def assess_fitness(self):
        ''' Assess fitness of all individual according to their performance in the
        Game of Life, as calculated by the fitness function. '''
        # Fitness function = Total number of tiles - number of tiles of the opponent
        def fitness_function(final_state):
            player1 = (final_state == 1).sum()
            player2 = (final_state == 2).sum()
            # return (player1 - player2), (player2 - player1)
            return player1, player2

        self.fitness = np.zeros(self.population_size)
        final_states = [game.grid() for game in self.games]
        for i, final_state in enumerate(final_states):
            scores = fitness_function(final_state)
            self.fitness[i] = scores[0]
            self.fitness[i + self.population_size // 2] = scores[1]
        self.fitness_history.append(self.fitness.max())

    @jit(cache=True)
    def select(self):
        ''' Returns a numpy array of indices of the players who are selected
        by evolution as being the fittest '''

        self.fitness = self.fitness - self.fitness.min() + 1
        selection_probs = self.fitness / self.fitness.sum()
        return np.random.choice(np.arange(self.population_size),
                                size=self.population_size,
                                p=selection_probs)
    @jit(cache=True)
    def mutate(self, population):
        ''' Performs random mutation. Each cell of each individual has a
        probability (mutation_rate) of being mutated '''
        for individual in population:
            it = np.nditer(individual, flags=['multi_index'])
            while not it.finished:
                if np.random.rand() < self.mutation_rate:
                    individual[it.multi_index] = 1 if it[0] == 0 else 0
                it.iternext()
        return population

    def make_pairs(self, individuals):
        cutoff = len(individuals) // 2
        first_half = individuals[:cutoff]
        second_half = individuals[cutoff:]
        return zip(first_half, second_half)

# def crossover(mating_pool):
#     ''' Implements two types of code recombination. Twice, parents are
#     selected from the mating pool and combine their genes '''
#     offspring = []
#
#     # Keep the top half from the mother and the bottom half from the father
#     for mother, father in make_pairs(mating_pool):
#         assert mother.shape[0] % 2 == 0
#         cutoff = mother.shape[0] // 2
#         child = np.vstack(mother[:cutoff,:], father[cutoff:,:])
#         offspring.append(child)
#
#     # Keep the left half from the mother and the right half from the father
#     for mother, father in make_pairs(mating_pool, shuffle=True):
#         assert mother.shape[1] % 2 == 0
#         cutoff = mother.shape[1] // 2
#         child = np.hstack(mother[:, :cutoff], father[:, cutoff:])
#         offspring.append(child)
#     return offspring

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--population", type=int, default=50, help="Population size")
parser.add_argumet("-m", "--mutation", type=float, default=0.3, help="Mutation rate")
parser.add_argument("-c", "--cycles", type=int, default=1000000, help="Number of evolution cycles")
arser.add_argument("-s", "--steps", type=int, default=100, help="Number of steps per round")

if __name__ == "__main__":
    args = parser.parse_args()
    evol = EvolutionaryBenchmark(args.population, args.mutation)
    evol.do_n_cycles(args.cycles, args.steps)