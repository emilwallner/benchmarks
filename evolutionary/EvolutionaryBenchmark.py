import numpy as np
import random

BOARD_DIMS = (512, 512)
GRID_DIMS = (64,64)

class EvolutionaryBenchmark:
    def __init__(self, population_size, mutation_rate):
        if population_size % 2 is not 0:
            raise ValueError("Population size must be even")
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.population = None

    def random_initialization(self):
        self.population = []
        for _ in range(self.population_size):
            self.population.append(np.random.randint(2, size=GRID_DIMS))

    def do_n_cycles(self, n):
        # Initialize if it hasn't been done already
        if self.population is None:
            self.random_initialization()
        # Perform one cycle of the Game of life and evolve
        for i in range(n):
            self.do_one_cycle()
            self.assess_fitness()
            next_generation = self.select()
            ## skipping crossover for now ##
            self.population = self.mutate(next_generation)

    def do_one_cycle(self):
        ''' Performs one cycle of the Game of Life'''
        games = []

        for pair in make_pairs(self.population):
            game = GameContainer(BOARD_DIMS[0], BOARD_DIMS[1])
            # TODO: load players to the game
            # TODO: play the game

    def assess_fitness(self):
        # TODO
        # Total number of tiles - number of tiles of the opponent
        # Assign in NUMPY array
        self.fitness = np.arange(self.population_size)
        self.fitness_history.append(self.fitness.max())

    def select(self):
        ''' Returns a numpy array of indices of the players who are selected
        by evolution as being the fittest '''

        selection_probs = self.fitness / self.fitness.sum()
        return np.random.choice(np.arange(self.population_size),
                                size=self.population_size,
                                p=selection_probs)

    def mutate(self, population):
        ''' Performs random mutation. Each cell of each individual has a
        probability (mutation_rate) of being mutated '''
        for individual in population:
            for cell in np.nditer(individual, op_flags=['readwrite']):
                if np.random.rand() < self.mutation_rate:
                    cell[...] = 1 if cell == 0 else 0
        return population

    def make_pairs(individuals, shuffle=False):
        if shuffle is True:
            random.shuffle(individuals)
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
