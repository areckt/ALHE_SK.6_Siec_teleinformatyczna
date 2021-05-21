import xml.dom.minidom
import numpy as np


class Link:
    def __init__(self, name, flow_load):
        self.name = name
        self.flow_load = flow_load


class EvolutionAlgorithm:
    def __init__(self, list_of_demands, population_size=150, crossover_prob=0.2, generations=100,
                 modularity=10, aggregation=False, seed=17):
        self.demands = list_of_demands
        self.population_size = population_size
        self.crossover_prob = crossover_prob
        self.generations = generations
        self.modularity = modularity
        self.aggregation = aggregation
        self.seed = seed
        self.max_num_of_paths = 7
        self.min_value = 0
        self.max_value = max([x.demand_value for x in list_of_demands])
        self.num_of_demands = len(list_of_demands)

    def initialize(self):
        # returns (population_size * num_of_demands * max_num_of_paths) random integers
        # in the range [min_value, max_value)
        return np.random.randint(self.min_value,
                                 self.max_value,
                                 (self.population_size, self.num_of_demands, self.max_num_of_paths))

    # TODO method to generate population (maybe 'select' would be a better name than 'generate'?)
    def generate(self, population, index):
        pass

    # TODO evaluation method - count used systems (the lesser the better)
    def evaluation_method(self, specimen, show_links=False):
        pass

    def evaluate(self, trial_specimen, specimen, population, index):
        specimen_value = self.evaluation_method(specimen)
        trial_specimen_value = self.evaluation_method(trial_specimen)
        if trial_specimen_value <= specimen_value:
            population[index] = trial_specimen

    # TODO crossover method (it's possible to skip it if it doesn't make sense to use it)
    def crossover(self, first_specimen, second_specimen):
        pass

    # TODO mutation - careful not to end up with some silly values of flow
    #  maybe some sort of repair method will be necessary
    def mutate(self, specimen):
        pass

    # TODO main 'run' function, generally we want to:
    #  1) GENERATE init population
    #  in loop for num_of_generations times:
    #    2) EVALUATE population
    #    3) MUTATE some % of population
    #    4) CROSSOVER some % of population (optional)
    #    5) SELECT population_size specimens to create next generation (preferably use tournament selection)
    #    back to 2)
    def run(self):
        pass


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

    network_data = xml.dom.minidom.parse('network.xml')
    demands = get_demands_from_network(network_data)

    for d in demands:
        print(d)

    # TODO parse options given by the user

    # TODO fill in missing variables
    algorithm = EvolutionAlgorithm(demands, ...)

    # TODO number of iterations should come from options given by the user
    iterations = ...

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
