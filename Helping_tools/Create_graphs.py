import networkx as nx
import pickle
import json

reg=3
ns=[50]
num_graphs=100

# for n in ns:
#     list_graphs=[]
#     for i in range(num_graphs):
#         G=nx.random_regular_graph(reg, n)
#         list_graphs.append(G)
#     pickle.dump(list_graphs, open(f"{num_graphs}_regular_graphs_nodes_{n}_reg_{reg}.pkl", 'wb'))








graphs = {}
with open(f'./graphs/100_regular_graphs_nodes_{50}_reg_3.pkl', 'rb') as f:
    data = pickle.load(f)
with open(f'./graphs/100_regular_graphs_nodes_{50}_reg_3_solutions.pkl', 'rb') as f:
    solutions = pickle.load(f)

for i in range(4):
    graphs[f'{i}'] = {'maxcut': solutions[i], 'graph': list(data[i].edges())}

for i in range(5, 31):
    graphs[f'{i-1}'] = {'maxcut': solutions[i], 'graph': list(data[i].edges())}

print(graphs)
with open('30_regular_graphs_nodes_50_reg_3.json', 'w') as f:
    json.dump(graphs, f)