import xml.dom.minidom


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
                                      paths
                                      ))
    return list_of_demands


if __name__ == '__main__':
    print("hello world")

    network_data = xml.dom.minidom.parse('network.xml')
    demands = get_demands_from_network(network_data)

    for d in demands:
        print(d)
