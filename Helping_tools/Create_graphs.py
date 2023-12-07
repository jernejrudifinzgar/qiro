import networkx as nx
import pickle

reg=3
ns=[30]
num_graphs=100

for n in ns:
    list_graphs=[]
    for i in range(num_graphs):
        G=nx.random_regular_graph(reg, n)
        list_graphs.append(G)
    pickle.dump(list_graphs, open(f"{num_graphs}_regular_graphs_nodes_{n}_reg_{reg}.pkl", 'wb'))


