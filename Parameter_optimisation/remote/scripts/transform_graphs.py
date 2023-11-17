import json
import pickle
import networkx as nx


with open('regular_graphs_maxcuts.json', 'r') as file:
    data = json.load(file)

for n in data['3'].keys():
    list_graphs = []
    for graph in data['3'][n].keys():
        edgelist = data["3"][n][graph]["edgelist"]
        G = nx.from_edgelist(edgelist)
        list_graphs.append(G)

    pickle.dump(list_graphs, open(f"rudis_100_regular_graphs_nodes_{n}_reg_{3}.pkl", 'wb'))






