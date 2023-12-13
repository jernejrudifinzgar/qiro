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


def plot_cuts_recalculation_per_graph(ns, ps, runs, regularity, version):
    for n in ns:
        list_exact = []
        list_x = []
        counter = 0
        fig = plt.figure()
        plt.title(f'Number of cuts for MAXCUT graphs with {n} nodes for different types of calculations')

        for run in runs: 
            #with open(my_path + f"/data/regular_graphs_maxcuts.json", 'r') as f:
            #    graphs = json.load(f)
            #optimal_cuts = graphs[str(regularity)][str(n)][str(run)]['optimal_cut']
            with open('100_regular_graphs_nodes_30_reg_3_solutions.pkl', 'rb') as file:
                data = pickle.load(file)
            list_exact = data.copy()
            list_x.append(f'Graph {run}')  
        
        list_single_cuts_norm = []

        for p in ps:

            list_qtensor_cuts = []
            list_qtensor_cuts_recalc_3 = []
            list_qtensor_cuts_recalc_6 = []
            list_single_cuts = []
            list_qtensor_cuts_norm_recalc_3 = []
            list_qtensor_cuts_norm_recalc_6 = []
            list_qtensor_cuts_norm = []
            list_graphs_recalc = []

            list_graphs = []

            for run in runs:
                #try: 
                #     with open (my_path + f"/data/results_test_run_{run}_n_{n}_p_{p}_version_{version}.txt") as f:
                #         lines = f.readlines()
                #     # if p==1:
                #     #     for line in lines:
                #     #         if line.find("Calculated number of cuts with analytic method:") != -1:
                #     #             line_analytic = line
                #     #             split_word = 'method:: '
                #     #             res = line_analytic.split(split_word, 1)
                #     #             cuts_analytic = res[1]
                    
                #     cuts_qtensor = lines[4]
                #     split_word = "networks: "
                #     res = cuts_qtensor.split(split_word, 1)
                #     cuts_qtensor = int(res[1])
                #     list_qtensor_cuts.append(cuts_qtensor)
                #     list_qtensor_cuts_norm.append(cuts_qtensor/list_exact[run])
                #     list_graphs.append(run)
                
                # except:
                #     print('file not available')

                try:
                    #with open (my_path + f"/data/results_run_{run}_n_{n}_p_{p}.pkl", 'rb') as f:
                    with open (my_path + f"/data/results_run_{run}_n_{n}_p_{p}_wo_recalc_version_{version}.pkl", 'rb') as f:
                        data = pickle.load(f)
                    cuts_qtensor = data['cuts_qtensor']
                    list_qtensor_cuts.append(cuts_qtensor)
                    list_qtensor_cuts_norm.append(cuts_qtensor/list_exact[run])
                    list_graphs.append(run)
                    if p == 1:
                        cuts_single = data['cuts_single']
                        list_single_cuts.append(cuts_single)
                        list_single_cuts_norm.append(cuts_single/list_exact[run])

                except:
                    print(f'file for {n} nodes, p={p} and run {run} is not available')

                try:
                    #with open (my_path + f"/data/results_run_{run}_n_{n}_p_{p}.pkl", 'rb') as f:
                    with open (my_path + f"/data/results_run_{run}_n_{n}_p_{p}_recalc_3_version_{version}.pkl", 'rb') as f:
                        data = pickle.load(f)
                    cuts_qtensor = data['cuts_qtensor']
                    list_qtensor_cuts_recalc_3.append(cuts_qtensor)
                    list_qtensor_cuts_norm_recalc_3.append(cuts_qtensor/list_exact[run])

                except:
                    print(f'file for {n} nodes, p={p} and run {run} is not available')

                try:
                    #with open (my_path + f"/data/results_run_{run}_n_{n}_p_{p}.pkl", 'rb') as f:
                    with open (my_path + f"/data/results_run_{run}_n_{n}_p_{p}_recalc_6_version_{version}.pkl", 'rb') as f:
                        data = pickle.load(f)
                    cuts_qtensor = data['cuts_qtensor']
                    list_qtensor_cuts_recalc_6.append(cuts_qtensor)
                    list_qtensor_cuts_norm_recalc_6.append(cuts_qtensor/list_exact[run])

                except:
                    print(f'file results_run_{run}_n_{n}_p_{p}_recalc_6_version_{version}.pkl is not available')

            

            #counter += 1
            #plt.scatter(list_graphs_recalc, list_qtensor_cuts_recalc, c=colors[counter], label = f'tensor network cuts for p={p} with 6-recalc')
            plt.scatter(list_graphs, list_qtensor_cuts_norm_recalc_3, c=colors[counter], label = f'tensor network cuts for p={p} with 3-recalc')
            counter += 1
            plt.scatter(list_graphs, list_qtensor_cuts_norm_recalc_6, c=colors[counter], label = f'tensor network cuts for p={p} with 6-recalc')
            counter += 1
            plt.scatter(list_graphs, list_qtensor_cuts_norm, c=colors[counter], label = f'tensor network cuts for p={p} with 1-recalc')            
            counter += 1
            #if p==1: 
                #plt.scatter(list_graphs_recalc, list_single_cuts, c=colors[counter], label = 'cuts analytic for p=1')
                #plt.scatter(list_graphs, list_single_cuts_norm, c=colors[counter], label = 'cuts analytic for p=1')
                #counter += 1
        plt.scatter(list_graphs, list_single_cuts_norm, c=colors[counter], label = 'cuts analytic for p=1')
        counter += 1
        counter += 1
        #plt.scatter(runs, list_exact, c=colors[counter], label = 'exact num of cuts')
        plt.xticks(runs, list_x)
        plt.legend()
        #fig.savefig(my_path + f'/results/Cuts_different_calculations_n_{n}.png')
        plt.show()
        #plt.close()

def plot_cuts_recalculation_per_p(ns, ps, runs, regularity, version):
    num_runs = len(runs)
    for n in ns: 
        counter = 0
        fig = plt.figure()
        plt.title(f'Number of cuts for MAXCUT graphs with {n} nodes for different types of calculations')

        cuts_list = []
        cuts_recalc_3_list = []
        cuts_recalc_6_list = []
        cuts_single_list = []
        list_exact = []

        for run in runs: 
            #with open(my_path + f"/data/regular_graphs_maxcuts.json", 'r') as f:
            #    graphs = json.load(f)
            #optimal_cuts = graphs[str(regularity)][str(n)][str(run)]['optimal_cut']
            with open('100_regular_graphs_nodes_30_reg_3_solutions.pkl', 'rb') as file:
                data = pickle.load(file)
            list_exact = data.copy()

        for p in ps:
            cuts_qtensor = 0
            cuts_qtensor_recalc_3 = 0
            cuts_qtensor_recalc_6 = 0
            cuts_single = 0
            for run in runs:
             
                with open (my_path + f"/data/results_run_{run}_n_{n}_p_{p}_recalc_6_version_{version}.pkl", 'rb') as f:
                    data = pickle.load(f)
                cuts = data['cuts_qtensor']
                cuts_norm = cuts/list_exact[run]
                cuts_qtensor_recalc_6 += cuts_norm
                if p == 1:
                        cuts = data['cuts_single']
                        cuts_norm = cuts/list_exact[run]
                        cuts_single += cuts_norm

                with open (my_path + f"/data/results_run_{run}_n_{n}_p_{p}_recalc_3_version_{version}.pkl", 'rb') as f:
                    data = pickle.load(f)
                cuts = data['cuts_qtensor']
                cuts_norm = cuts/list_exact[run]
                cuts_qtensor_recalc_3 += cuts_norm

                with open (my_path + f"/data/results_run_{run}_n_{n}_p_{p}_wo_recalc_version_{version}.pkl", 'rb') as f:
                    data = pickle.load(f)
                cuts = data['cuts_qtensor']
                cuts_norm = cuts/list_exact[run]
                cuts_qtensor += cuts_norm
            
            cuts_list.append(cuts_qtensor/num_runs)
            cuts_recalc_3_list.append(cuts_qtensor_recalc_3/num_runs)
            cuts_recalc_6_list.append(cuts_qtensor_recalc_6/num_runs)
            if p==1:
                cuts_single_list.append(cuts_single/num_runs)

        plt.scatter([1], cuts_single_list, c=colors[counter], label = 'cuts analytic for p=1')
        counter += 1
        plt.plot(ps, cuts_list, c=colors[counter], label = f'tensor network cuts for p={p} with 1-recalc')
        counter += 1
        plt.plot(ps, cuts_recalc_3_list, c=colors[counter], label = f'tensor network cuts for p={p} with 3-recalc')
        counter += 1
        plt.plot(ps, cuts_recalc_6_list, c=colors[counter], label = f'tensor network cuts for p={p} with 6-recalc')

        plt.legend()
        #fig.savefig(my_path + f'/results/Cuts_different_calculations_n_{n}.png')
        plt.show()


            




if __name__ == '__main__':
    ns = [30] #[60, 80, 100, 120, 140, 160, 180, 200]
    ps= [1, 2, 3]
    recalculations = [3, 6]
    regularity = 3
    runs = list(range(20))
    version = 1

    colors=['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:pink', 'tab:brown', 'tab:grey', 'tab:olive', 'tab:cyan']

    my_path = os.path.dirname(__file__)
    my_path = os.path.dirname(my_path)

    #plot_cuts_recalculation_per_graph(ns, ps, runs, regularity, version)
    plot_cuts_recalculation_per_p(ns, ps, runs, regularity, version)

    