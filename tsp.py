"""
Traveller salesman problem http://www.math.uwaterloo.ca/tsp/vlsi/index.html
"""
import random
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool

from deap import base, creator, tools, algorithms

# Creates a new class from the base class:
# FitnessMin(base.Fitness) and Individual(list)
creator.create('FitnessMin', base.Fitness, weights=(-1.0,))
creator.create('Individual', list, fitness=creator.FitnessMin)


def data_reader(filename):
    coordinates = list()
    with open(filename, 'r') as file:
        line = file.readline()
        while line:
            _, x, y = line.strip('\n').split(' ')
            coordinates.append((int(x), int(y)))
            line = file.readline()

    return coordinates


def plot_route(coordinates, order=None, is_route=True):
    x = list()
    y = list()
    if order is not None:
        for point in order:
            x.append(coordinates[point][0])
            y.append(coordinates[point][1])
    else:
        for c in coordinates:
            x.append(c[0])
            y.append(c[1])

    linestyle = '--' if is_route else ''
    plt.figure()
    plt.plot(x, y, label='Points set', linestyle=linestyle, marker='o')
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.tight_layout()
    # plt.grid()
    plt.show()


class DistanceKeeper:
    def __init__(self, coordinates):
        self.distances = np.zeros((len(coordinates), len(coordinates)))
        self.calculate_all_distances(coordinates)

    def calculate_all_distances(self, coordinates):
        for i in range(len(coordinates)):
            for j in range(len(coordinates)):
                self.distances[i, j] = \
                    np.sqrt((abs(coordinates[i][0] - coordinates[j][0]) ** 2)
                            + (abs(coordinates[i][1] - coordinates[j][1]) ** 2))


def evaluation(dk, individual):
    distance = 0
    for i in range(len(individual) - 1):
        # prev = coordinates[individual[i]]
        # next = coordinates[individual[i + 1]]
        # distance += np.linalg.norm(np.asarray(prev) - np.asarray(next))
        # dist = np.sqrt((abs(prev[0] - next[0]) ** 2) + (abs(prev[1] - next[1]) ** 2))
        dist = dk.distances[individual[i]][individual[i + 1]]
        distance += dist

    return distance,


class TravellerSalesmanProblemGA:
    def __init__(self, coordinates, pop_size, generations, cross_prob, mut_prob, dk):
        self.pool = Pool(10)  # 10
        self.individual_size = len(coordinates)
        self.pop_size = pop_size
        self.generations = generations
        self.cross_prob = cross_prob
        self.mut_prob = mut_prob

        self.dk = dk

        self.current_iter = 0
        self.step = abs(cross_prob - mut_prob) / (generations - 1000)

        random.seed(39)
        toolbox = base.Toolbox()
        toolbox.register("map", self.pool.map)

        # Register all in a toolbox
        # We need 'indices' to create a random route through all cities
        toolbox.register('indices', random.sample, range(self.individual_size), self.individual_size)
        toolbox.register('individual', tools.initIterate, creator.Individual, toolbox.indices)
        toolbox.register('population', tools.initRepeat, list, toolbox.individual)
        toolbox.register('evaluate', evaluation, self.dk)
        toolbox.register('mate', tools.cxOrdered)
        toolbox.register('mutate', tools.mutShuffleIndexes, indpb=0.01)  # 0.05, 0.01
        toolbox.register('select', tools.selTournament, tournsize=30)   # 10, 30, 4

        self.toolbox = toolbox

    def __call__(self, *args, **kwargs):
        pop = self.toolbox.population(n=self.pop_size)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register('avg', np.mean)
        stats.register('std', np.std)
        stats.register('min', np.min)
        stats.register('max', np.max)

        # self.current_iter += 1
        # if self.current_iter > 1000 and self.cross_prob >= self.mut_prob:
        #     cross_prob = self.cross_prob - self.current_iter * self.step
        #     mut_prob = self.mut_prob + self.current_iter * self.step
        # elif self.current_iter > 1000 and self.cross_prob < self.mut_prob:
        #     cross_prob = self.cross_prob + self.current_iter * self.step
        #     mut_prob = self.mut_prob - self.current_iter * self.step
        # else:
        #     cross_prob = self.cross_prob
        #     mut_prob = self.mut_prob

        algorithms.eaMuPlusLambda(pop, self.toolbox, self.pop_size, self.pop_size,
                                  self.cross_prob, self.mut_prob,
                                  self.generations, stats=stats, halloffame=hof)

        # algorithms.eaSimple(pop, self.toolbox, cxpb=0.7, mutpb=0.05,
        #                     ngen=self.generations, stats=stats, halloffame=hof)

        return pop, stats, hof


def main():
    coordinates = data_reader('xqf131')  # xqf131 - 564, xqg237 - 1019, pma343 - 1368

    print(f'Individual size: {len(coordinates)}')

    pop_size = 200
    generations = 2000
    cross_prob = 0.5    # 0.6; 0.15; 0.7
    mut_prob = 0.5      # 0.05; 0.5; 0.2

    dk = DistanceKeeper(coordinates)

    # plot_route(coordinates, None, False)

    model = TravellerSalesmanProblemGA(coordinates, pop_size,
                                       generations, cross_prob, mut_prob, dk)

    pop, stats, hof = model()

    # best = tools.selBest(pop, 1)[0]
    # print('Best Path', best)
    print('Best Path', hof[0])

    plot_route(coordinates, hof[0])


if __name__ == '__main__':
    main()
