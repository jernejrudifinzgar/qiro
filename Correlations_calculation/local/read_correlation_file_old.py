import pickle
import networkx as nx


"""Structure of correlation file:
{regularity: {number_of_nodes: {p:[general information about graphs, weighted graphs with weights corresponding to correlations]}}}"""

#Function to create unweighted graph/problem:
def create_graph(graph):
    G = nx.from_edgelist(graph.edges())
    return G

#Function to calculate overall MAXCut energy:
def calc_energy(graph):
    E = 0
    for edge in graph.edges():
        E += graph.get_edge_data(edge[0], edge[1])['weight']
    Ed = graph.number_of_edges()
    return (Ed - E)/2


if __name__ == '__main__':

    imported_graphs_dict = pickle.load(open('correlations_MAXCut.pkl', 'rb'))


    #Show all available graphs:
    print('All available graphs and QAOAs:')
    for regularity in range(3,6):
        for nodes in [100, 150, 200]:
            if regularity==3:
                for p in range(1,5):
                    print(imported_graphs_dict[regularity][nodes][p][0])
            if regularity==4:
                for p in range(1,4):
                    print(imported_graphs_dict[regularity][nodes][p][0])
            if regularity==5:
                for p in range(1,3):
                    print(imported_graphs_dict[regularity][nodes][p][0])
                    #The individual graphs with the corresponding features are saved as imported_graphs_dict[regularity][nodes][p][1-10]. E.g., see next few lines.



    #Get correlations of an example graph:
    example_graph = imported_graphs_dict[3][100][4][5]
    for edge in example_graph.edges():
        print("correlation of edge {}: {}".format(edge, example_graph.get_edge_data(edge[0], edge[1])['weight']))

    #Calculate MAXCut energy: 
    E = calc_energy(example_graph)

    #Create unweighted graph from data:
    G = create_graph(example_graph)
 
 

