import copy
import math
import random
import xml.dom.minidom
from optparse import OptionParser
import numpy as np
from numpy.random import randint
from util_classes import *


class EvolutionAlgorithm:
    def __init__(self, list_of_demands, population_size=100, crossover_prob=0.2, mutation_prob=0.2,
                 generations=10, modularity=10, aggregation=False, seed=17):
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

    def __initialize(self):
        # returns population_size vectors, where each vector has num_of_demands (66) vectors,
        # where each of those vectors has max_num_of_paths (7) random integers
        # in the range [min_value, max_value)
        return np.random.randint(self.min_value,
                                 self.max_value,
                                 (self.population_size, self.num_of_demands, self.max_num_of_paths))

    def __tournament_selection(self, old_population):
        random.seed()
        new_population = []
        while len(new_population) != len(old_population):
            first_specimen = old_population[random.randrange(0, len(old_population), 1)]
            second_specimen = old_population[random.randrange(0, len(old_population), 1)]
            if self.evaluation_method(first_specimen) < self.evaluation_method(second_specimen):
                new_population.append(first_specimen)
            else:
                new_population.append(second_specimen)

        return new_population

    def evaluation_method(self, specimen_to_evaluate, show_links=False):
        num_of_systems = 0  # this is what we want to minimize

        list_of_links = []

        # add all loads on links to list
        for flows_list_index, flows_for_demand in enumerate(specimen_to_evaluate):
            for flow_index, flow in enumerate(flows_for_demand):
                if flow <= 0:
                    continue

                corresponding_path = demands[flows_list_index].admissible_paths[flow_index]
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

    def __evaluate(self, trial_specimen, specimen_to_evaluate, cur_population, index):
        specimen_value = self.evaluation_method(specimen_to_evaluate)
        trial_specimen_value = self.evaluation_method(trial_specimen)
        if trial_specimen_value <= specimen_value:
            cur_population[index] = trial_specimen

    def __crossover_population(self, population_to_crossover):
        crossovered_population = []
        first, second = 0, 1
        while second < len(population_to_crossover):
            random.seed()
            random_num = random.uniform(0, 1)
            first_specimen = population_to_crossover[first]
            second_specimen = population_to_crossover[second]
            if random_num <= self.crossover_prob:
                first_specimen, second_specimen = self.__crossover(first_specimen, second_specimen)
            crossovered_population.append(first_specimen)
            crossovered_population.append(second_specimen)
            first += 2
            second += 2

        if len(population_to_crossover) % 2 != 0:
            crossovered_population.append(population_to_crossover[-1])

        if len(population_to_crossover) != len(crossovered_population):
            raise Exception

        random.seed(self.seed)
        return crossovered_population

    @staticmethod
    def __crossover(first_specimen, second_specimen):
        # children are copies of parents by default
        first_child, second_child = first_specimen.copy(), second_specimen.copy()

        # select crossover point that is not on the end of the vector
        pt = random.randrange(0, len(first_child) - 2, 1)

        # perform crossover
        first_child = np.concatenate((first_specimen[:pt], second_specimen[pt:]))
        second_child = np.concatenate((second_specimen[:pt], first_specimen[pt:]))

        return first_child, second_child

    def __mutate_population(self, population_to_mutate):
        mutated_population = []
        for cur_specimen in population_to_mutate:
            random.seed()
            random_num = random.uniform(0, 1)
            if random_num <= self.mutation_prob:
                mutated_specimen = self.__mutate(cur_specimen)
                mutated_repaired_specimen = self.__repair(mutated_specimen)
                mutated_population.append(mutated_repaired_specimen)
            else:
                mutated_population.append(cur_specimen)
        random.seed(self.seed)
        return mutated_population

    def __mutate(self, specimen_to_mutate):
        mutated_vector = copy.deepcopy(specimen_to_mutate)

        i = random.randint(0, self.num_of_demands - 1)
        j = random.randint(0, self.max_num_of_paths - 1)

        index_of_flow = 0
        for index, flow in enumerate(specimen_to_mutate[i]):
            if flow > 0:
                index_of_flow = index

        mutated_vector[i][index_of_flow] = 0

        mutated_vector[i][j] = specimen_to_mutate[i][index_of_flow]

        return mutated_vector

    def __repair(self, specimen_to_repair):

        # at first we switch all negative flows to positive
        for demand in specimen_to_repair:
            for demand_index, flow in enumerate(demand):
                if demand[demand_index] < 0:
                    demand[demand_index] *= -1

        # if aggregation is on (i.e. only one flow, rest must be 0)
        # we find the max flow and zero the rest
        # then we give the found max full flow
        if self.aggregation:
            for demand_index, demand in enumerate(specimen_to_repair):
                max_index = 0
                max_value = -1
                for flow_index, flow in enumerate(demand):
                    if flow > max_value:
                        max_index = flow_index
                        max_value = flow
                    demand[flow_index] = 0
                demand[max_index] = demands[demand_index].demand_value

        # if aggregation is off we normalize all flows to prevent overflow
        else:
            for demand_index, demand in enumerate(specimen_to_repair):
                ratio = demands[demand_index].demand_value / np.sum(demand)

                for flow_index, flow in enumerate(demand):
                    demand[flow_index] = round(demand[flow_index] * ratio)

        return specimen_to_repair

    def run(self):
        random.seed(self.seed)
        np.random.seed(self.seed)

        # GENERATE init cur_population
        cur_population = self.__initialize()

        for index, cur_specimen in enumerate(cur_population.tolist()):
            cur_population[index] = self.__repair(cur_specimen)

        for generation in range(self.generations):
            # MUTATION of some % of cur_population
            cur_population = self.__mutate_population(cur_population)

            # CROSSOVER of some % of cur_population
            cur_population = self.__crossover_population(cur_population)

            # TOURNAMENT SELECTION
            cur_population = self.__tournament_selection(cur_population)

        return cur_population


if __name__ == '__main__':
    print("hello world")

    usage = "usage: %prog [options]\n" \
            "Debug: -q, -d\n" \
            "Params: -s, -i, -m, -a, -n, -c, -g, -u\n"
    parser = OptionParser(usage=usage)

    parser.add_option("-q", "--debug", action="store_true", dest="debug", default=False,
                      help="Prints debug info")
    parser.add_option("-d", "--demands", action="store_true", dest="demands", default=False,
                      help="Prints demands")

    parser.add_option("-i", "--iterations", type="int", dest="iterations", default=10,
                      help="Number of algorithms runs, incrementing seed by 1 (default 10)")
    parser.add_option("-s", "--seed", type="int", dest="seed", default=17,
                      help="Initial seed for numpy and random (default 17)")
    parser.add_option("-m", "--modularity", type="int", dest="modularity", default=10,
                      help="Modularity for systems counting (default 10)")
    parser.add_option("-a", "--aggregation", action="store_true", dest="aggregation", default=False,
                      help="Aggregate flows")
    parser.add_option("-g", "--generations", type="int", dest="generations", default=10,
                      help="Max generations (default 10)")
    parser.add_option("-n", "--population_size", type="int", dest="population_size", default=100,
                      help="Population size (default 100)")
    parser.add_option("-c", "--crossover", type="float", dest="crossover_probability", default=0.2,
                      help="Crossover probability (default 0.2)")
    parser.add_option("-u", "--mutation", type="float", dest="mutation_probability", default=0.2,
                      help="Mutation probability (default 0.2)")

    (options, args) = parser.parse_args()
    network_data = xml.dom.minidom.parse('network.xml')
    demands = get_demands_from_network(network_data)

    # global debug
    debug = options.debug

    if options.demands:
        for demand in demands:
            print(demand)

    algorithm = EvolutionAlgorithm(demands, population_size=options.population_size,
                                   crossover_prob=options.crossover_probability,
                                   mutation_prob=options.mutation_probability,
                                   generations=options.generations,
                                   modularity=options.modularity,
                                   aggregation=options.aggregation,
                                   seed=options.seed)

    iterations = options.iterations

    all_values = []
    for i in range(iterations):
        population = algorithm.run()

        best = population[0]
        best_value = EvolutionAlgorithm.evaluation_method(algorithm, best)

        for specimen in population:
            current_value = EvolutionAlgorithm.evaluation_method(algorithm, specimen)
            if current_value < best_value:
                best = specimen
                best_value = EvolutionAlgorithm.evaluation_method(algorithm, best)

        all_values.append(best_value)

        if debug:
            print("BEST SPECIMEN: ")
            print(str(best)+'\n')
        else:
            print("\n### " + str(i) + " ###")
        print("\nBest value: " + str(best_value))
    info = f'\nAverage of {str(len(all_values))} runs: {str(sum(all_values) / len(all_values))}\n' \
           f'Best: {str(min(all_values))}\tWorst: {str(max(all_values))} '
    print(info)
