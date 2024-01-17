import networkx as nx
import pickle
import json
import os
import matplotlib.pyplot as plt
import matplotlib
from IPython.display import set_matplotlib_formats
__plot_height = 9.119
matplotlib.rcParams['figure.figsize'] = (1.718*__plot_height, __plot_height)
set_matplotlib_formats('svg')
import numpy as np


my_path = os.path.dirname(__file__)
my_path = os.path.dirname(my_path)

ns = [60]
ps = [1, 2, 3]
runs = list(range(10, 20))
regularity = 3
version = 1


ns_graphs_rudi = list(range(60, 220, 20))
ns_graphs_maxi = [30, 50]

cuts_qtensor = [
    [78, 80, 80, 79, 79, 81, 80, 81, 79, 79],
    [80, 82, 81, 80, 80, 81, 82, 81, 81, 81],
    [80, 82, 81, 80, 80, 82, 82, 81, 81, 81],
]

        
for n in ns:
    # if n in ns_graphs_rudi:
    #     list_exact=[]
    #     with open(my_path + f"/data/regular_graphs_maxcuts.json", 'r') as f:
    #         graphs = json.load(f)
    #     for run in runs: 
    #         optimal_cuts = graphs[str(regularity)][str(n)][str(run)]['optimal_cut']
    #         list_exact.append(optimal_cuts)

    # elif n in ns_graphs_maxi:
    #     with open(f'100_regular_graphs_nodes_{n}_reg_3_solutions.pkl', 'rb') as file:
    #         data = pickle.load(file)
    #     list_exact = data.copy()

    # print(list_exact)


    for p in ps:
        for run in runs:
            if p==1:
                with open (my_path + f"/data/results_run_{run}_n_{n}_p_{p}_wo_recalc_version_{version}.pkl", 'rb') as f:
                    data = pickle.load(f)
                data['cuts_qtensor'] = cuts_qtensor[p-1][run-10]
            else:
                data = {}
                data['cuts_qtensor'] = cuts_qtensor[p-1][run-10]


            pickle.dump(data, open(my_path + f"/data/results_run_{run}_n_{n}_p_{p}_wo_recalc_version_{version}.pkl", 'wb'))

# for n in ns:
#     for p in ps:
#         for run in runs:
#             try:
#                 with open (my_path + f"/data/results_run_{run}_n_{n}_p_{p}_wo_recalc_version_{version}.pkl", 'rb') as f:
#                     data_0 = pickle.load(f)

#                 with open (my_path + f"/data/results_run_{run+10}_n_{n}_p_{p}_wo_recalc_version_{version}.pkl", 'rb') as f:
#                     data_1 = pickle.load(f)
#                 data_1['solution_qtensor'] = data_0['solution_qtensor']
#                 data_1['energies_qtensor'] = data_0['energies_qtensor']

#                 pickle.dump(data_1, open(my_path + f"/data/results_run_{run+10}_n_{n}_p_{p}_wo_recalc_version_{version}.pkl", 'wb'))
#             except:
#                 print(run)


               
