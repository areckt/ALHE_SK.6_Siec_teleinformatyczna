import copy
import math
import random
import xml.dom.minidom
from optparse import OptionParser
import numpy as np
from numpy.random import randint
from numpy.random import rand


class Link:
    def __init__(self, name, flow_load):
        self.name = name
        self.flow_load = flow_load


class EvolutionAlgorithm:
    def __init__(self, list_of_demands, population_size=150, crossover_prob=0.2, mutation_prob=0.2,
                 generations=100, modularity=10, aggregation=False, seed=17):
        self.demands = list_of_demands
        self.population_size = population_size
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.generations = generations
        self.modularity = modularity
        self.aggregation = aggregation
        self.seed = seed
        self.max_num_of_paths = 7
        self.min_value = 0
        self.max_value = max([x.demand_value for x in list_of_demands])
        self.num_of_demands = len(list_of_demands)

    def initialize(self):
        # returns population_size vectors, where each vector has num_of_demands (66) vectors,
        # where each of those vectors has max_num_of_paths (7) random integers
        # in the range [min_value, max_value)
        return np.random.randint(self.min_value,
                                 self.max_value,
                                 (self.population_size, self.num_of_demands, self.max_num_of_paths))

    def generate(self, population, scores, k=3):
        # first random selection
        selection_ix = randint(len(population))
        for ix in randint(0, len(population), k - 1):
            # check if better (e.g. perform a tournament)
            if scores[ix] < scores[selection_ix]:
                selection_ix = ix
        return population[selection_ix]

    def evaluation_method(self, specimen, show_links=False):
        num_of_systems = 0  # this is what we want to minimize

        list_of_links = []

        # add all loads on links to list
        for i, flows_for_demand in enumerate(specimen):
            for j, flow in enumerate(flows_for_demand):
                if flow <= 0:
                    continue

                corresponding_path = demands[i].admissible_paths[j]
                for link_name in corresponding_path:
                    does_link_exist = False
                    link_index = 0
                    for index, l in enumerate(list_of_links):
                        if l.name == link_name:
                            does_link_exist = True
                            link_index = index
                            break

                    if does_link_exist:
                        list_of_links[link_index].flow_load += flow
                    else:
                        list_of_links.append(Link(link_name, flow))

        # for every loaded link add system depending on modularity
        for link in list_of_links:
            if show_links:
                print("Link: " + str(link.name) + "\n\tLoad: " + str(link.flow_load))
            num_of_systems += math.ceil(link.flow_load / self.modularity)
        return num_of_systems

    def evaluate(self, trial_specimen, specimen, population, index):
        specimen_value = self.evaluation_method(specimen)
        trial_specimen_value = self.evaluation_method(trial_specimen)
        if trial_specimen_value <= specimen_value:
            population[index] = trial_specimen

    def crossover(self, first_specimen, second_specimen):
        # children are copies of parents by default
        first_child, second_child = first_specimen.copy(), second_specimen.copy()
        # check for recombination
        if rand() < self.crossover_prob:
            # select crossover point that is not on the end of the string
            pt = randint(1, len(first_specimen) - 2)
            # perform crossover
            first_child = first_specimen[:pt] + second_specimen[pt:]
            second_child = second_specimen[:pt] + first_specimen[pt:]
        return [first_child, second_child]

    def mutate_population(self, population):
        mutated_population = []
        for specimen in population:
            random.seed()
            random_num = random.uniform(0, 1)
            if random_num <= self.mutation_prob:
                mutated_specimen = self.mutate(specimen)
                mutated_repaired_specimen = self.repair(mutated_specimen)
                mutated_population.append(mutated_repaired_specimen)
            else:
                mutated_population.append(specimen)
        random.seed(self.seed)
        return mutated_population

    def mutate(self, specimen):
        mutated_vector = copy.deepcopy(specimen)

        i = random.randint(0, self.num_of_demands - 1)
        j = random.randint(0, self.max_num_of_paths - 1)

        index_of_flow = 0
        for index, flow in enumerate(specimen[i]):
            if flow > 0:
                index_of_flow = index

        mutated_vector[i][index_of_flow] = 0

        mutated_vector[i][j] = specimen[i][index_of_flow]

        return mutated_vector

    def repair(self, specimen):

        # at first we switch all negative flows to positive
        for demand in specimen:
            for i, flow in enumerate(demand):
                if demand[i] < 0:
                    demand[i] *= -1

        # if aggregation is on (i.e. only one flow, rest must be 0)
        # we find the max flow and zero the rest
        # then we give the found max full flow
        if self.aggregation:
            for i, demand in enumerate(specimen):
                max_index = 0
                max = -1
                for j, flow in enumerate(demand):
                    if flow > max:
                        max_index = j
                        max = flow
                    demand[j] = 0
                demand[max_index] = demands[i].demand_value

        # if aggregation is off we normalize all flows to prevent overflow
        else:
            for i, demand in enumerate(specimen):
                ratio = demands[i].demand_value / np.sum(demand)

                for j, flow in enumerate(demand):
                    demand[j] = round(demand[j] * ratio)

        return specimen

    # TODO main 'run' function, generally we want to:
    #  1) GENERATE init population
    #  in loop for num_of_generations times:
    #    2) EVALUATE population
    #    3) MUTATE some % of population
    #    4) CROSSOVER some % of population (optional)
    #    5) SELECT population_size specimens to create next generation (preferably use tournament selection)
    #    back to 2)
    def run(self):

        random.seed(self.seed)
        np.random.seed(self.seed)

        print(str(random.uniform(0, 1)))


        #  1) GENERATE init population
        population = self.initialize()
        print("num of specimens in population: " + str(len(population)))
        print("len of one specimen: " + str(len(population[0])))
        print("len of one element of specimen: " + str(len(population[0][0])))

        print(population[0][0])

        for i, specimen in enumerate(population.tolist()):
            population[i] = self.repair(specimen)

        print(population[0][0])

        for generation in range(self.generations):
            # 3) MUTATE some % of population
            population = self.mutate_population(population)
            print(population[0][0])

        return population


class Demand:
    def __init__(self, source, target, demand_value, admissible_paths):
        self.source = source
        self.target = target
        self.demand_value = demand_value
        self.admissible_paths = admissible_paths

    def __repr__(self):
        paths_str = []
        for i, path in enumerate(self.admissible_paths):
            links_str = []
            for link in path:
                links_str.append("\t\t" + link)
            paths_str.append("\n\tPath " + str(i) + ": " + "\n" + "\n".join(links_str) + "\n")

        return f'Source: {self.source}\n' \
               f'Target: {self.target}\n' \
               f'DemandValue: {self.demand_value}\n' \
               f'AdmissiblePaths: {"".join(paths_str)}'


def get_demands_from_network(network):
    list_of_demands = []
    for demand in network.getElementsByTagName('demand'):
        paths = []
        for path in demand.getElementsByTagName('admissiblePath'):
            links = []
            for link in path.getElementsByTagName('linkId'):
                links.append(link.firstChild.data)
            paths.append(links)

        list_of_demands.append(Demand(demand.getElementsByTagName('source')[0].firstChild.data,
                                      demand.getElementsByTagName('target')[0].firstChild.data,
                                      int(float(demand.getElementsByTagName('demandValue')[0].firstChild.data)),
                                      paths))

    return list_of_demands


if __name__ == '__main__':
    print("hello world")

    usage = "usage: %prog [options]\n" \
            "Params: -s, -i, -m, -a, -n, -c, -w, -g, -u\n"
    parser = OptionParser(usage=usage)

    parser.add_option("-d", "--demands", action="store_true", dest="demands", default=False,
                      help="Prints demands")

    parser.add_option("-i", "--iterations", type="int", dest="iterations", default=10,
                      help="Number of algorithms runs, incrementing seed by 1 (default 10)")

    parser.add_option("-s", "--seed", type="int", dest="seed", default=17,
                      help="Initial seed for numpy and random (default 17)")
    parser.add_option("-m", "--modularity", type="int", dest="modularity", default=10,
                      help="Modularity for systems counting (default 10)")
    parser.add_option("-a", "--aggregation", action="store_true", dest="aggregation", default=True,
                      help="Aggregate flows")
    parser.add_option("-g", "--generations", type="int", dest="generations", default=100,
                      help="Max generations (default 100)")

    parser.add_option("-n", "--population_size", type="int", dest="population_size", default=150,
                      help="Population size (default 150)")
    parser.add_option("-c", "--crossover", type="float", dest="crossover_probability", default=0.2,
                      help="Crossover probability (default 0.2)")
    parser.add_option("-u", "--mutation", type="float", dest="mutation_probability", default=0.2,
                      help="Mutation probability (default 0.2)")

    (options, args) = parser.parse_args()
    network_data = xml.dom.minidom.parse('network.xml')
    demands = get_demands_from_network(network_data)

    # for d in demands:
    #     print(d)

    algorithm = EvolutionAlgorithm(demands, population_size=options.population_size,
                                   crossover_prob=options.crossover_probability,
                                   mutation_prob=options.mutation_probability,
                                   generations=options.generations,
                                   modularity=options.modularity,
                                   aggregation=options.aggregation,
                                   seed=options.seed)

    iterations = options.iterations

    for i in range(iterations):
        population = algorithm.run()

        best = population[0]
        best_value = EvolutionAlgorithm.evaluation_method(algorithm, best)

        for specimen in population:
            current_value = EvolutionAlgorithm.evaluation_method(algorithm, specimen)
            if current_value < best_value:
                best = specimen
                best_value = EvolutionAlgorithm.evaluation_method(algorithm, best)

        print("\n### " + str(i) + " ###")
        print("\nBest value: " + str(best_value))
