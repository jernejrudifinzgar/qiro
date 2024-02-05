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


        with open(f'100_regular_graphs_nodes_{n}_reg_3_solutions.pkl', 'rb') as file:
            data = pickle.load(file)
        list_exact = data.copy()        
        list_single_cuts_norm = []
        list_graphs_single = []

        for p in ps:

            list_qtensor_cuts = []
            list_qtensor_cuts_recalc_3 = []
            list_qtensor_cuts_recalc_6 = []
            list_single_cuts = []
            list_qtensor_cuts_norm_recalc_3 = []
            list_qtensor_cuts_norm_recalc_6 = []
            list_qtensor_cuts_norm = []

            list_graphs_qtensor = []
            list_graphs_qtensor_recalc_3 = []
            list_graphs_qtensor_recalc_6 = []

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
                    #with open (my_path + f"/data/results_run_{run}_iteration_{0}_n_{n}_p_{p}_recalc_5_initialization_fixed_angles_optimization_version_{version}.pkl", 'rb') as f:
                        data = pickle.load(f)
                    cuts_qtensor = data['cuts_qtensor']
                    list_qtensor_cuts.append(cuts_qtensor)
                    list_qtensor_cuts_norm.append(cuts_qtensor/list_exact[run])
                    list_graphs_qtensor.append(run)
                    if p == 1:
                        cuts_single = data['cuts_single']
                        list_single_cuts.append(cuts_single)
                        list_single_cuts_norm.append(cuts_single/list_exact[run])
                        list_graphs_single.append(run)
                    
                    list_x.append(f'{run}')  

                except:
                    print(f'file for {n} nodes, p={p} and run {run} is not available')

                #3recalc
                try:
                    #with open (my_path + f"/data/results_run_{run}_n_{n}_p_{p}.pkl", 'rb') as f:
                    with open (my_path + f"/data/results_run_{run}_n_{n}_p_{p}_recalc_3_version_{version}.pkl", 'rb') as f:
                        data = pickle.load(f)
                    cuts_qtensor = data['cuts_qtensor']
                    list_qtensor_cuts_recalc_3.append(cuts_qtensor)
                    list_qtensor_cuts_norm_recalc_3.append(cuts_qtensor/list_exact[run])
                    list_graphs_qtensor_recalc_3.append(run)

                except:
                    print(f'file results_run_{run}_n_{n}_p_{p}_recalc_3_version_{version}.pkl is not available')

                #6recalc
                try:
                    #with open (my_path + f"/data/results_run_{run}_n_{n}_p_{p}.pkl", 'rb') as f:
                    with open (my_path + f"/data/results_run_{run}_n_{n}_p_{p}_recalc_6_version_{version}.pkl", 'rb') as f:
                        data = pickle.load(f)
                    cuts_qtensor = data['cuts_qtensor']
                    list_qtensor_cuts_recalc_6.append(cuts_qtensor)
                    list_qtensor_cuts_norm_recalc_6.append(cuts_qtensor/list_exact[run])
                    list_graphs_qtensor_recalc_6.append(run)

                except:
                    print(f'file results_run_{run}_n_{n}_p_{p}_recalc_6_version_{version}.pkl is not available')

            
            counter_cuts = 0
            average_list = []
            for cut in list_qtensor_cuts_norm:
                counter_cuts += cut
            average = counter_cuts/len(list_graphs_qtensor)
            for i in list_graphs_qtensor:
                average_list.append(average)

            #counter += 1
            #plt.scatter(list_graphs_recalc, list_qtensor_cuts_recalc, c=colors[counter], label = f'tensor network cuts for p={p} with 6-recalc')
            plt.scatter(list_graphs_qtensor_recalc_3, list_qtensor_cuts_norm_recalc_3, c=colors[counter], label = f'tensor network cuts for p={p} with 3-recalc')
            counter += 1
            plt.scatter(list_graphs_qtensor_recalc_6, list_qtensor_cuts_norm_recalc_6, c=colors[counter], label = f'tensor network cuts for p={p} with 6-recalc')
            counter += 1
            plt.scatter(list_graphs_qtensor, list_qtensor_cuts_norm, c=colors[counter], label = f'tensor network cuts for p={p} with 1-recalc')   
            plt.plot(list_graphs_qtensor, average_list, color=colors[counter], linestyle='dashed', label = f'average cut ratio with tensor network p={p}')
         
            counter += 1









            #if p==1: 
                #plt.scatter(list_graphs_recalc, list_single_cuts, c=colors[counter], label = 'cuts analytic for p=1')
                #plt.scatter(list_graphs, list_single_cuts_norm, c=colors[counter], label = 'cuts analytic for p=1')
                #counter += 1
        counter_cuts = 0
        average_list = []
        for cut in list_single_cuts_norm:
            counter_cuts += cut
        average = counter_cuts/len(list_graphs_single)
        for i in list_graphs_single:
            average_list.append(average)
        plt.plot(list_graphs_single, average_list, color=colors[counter], linestyle='dashed', label = 'average cut ratio with analytic p=1')
        #plt.scatter(list_graphs_single, list_single_cuts_norm, c=colors[counter], label = 'cuts analytic for p=1')
        counter += 1
        counter += 1
        #plt.scatter(runs, list_exact, c=colors[counter], label = 'exact num of cuts')
        plt.xticks(runs, list_x)
        plt.xlabel('Graphs')
        plt.ylabel('Optimal cuts ratio')
        plt.legend()
        #fig.savefig(my_path + f'/results/Cuts_different_calculations_per_graph_n_{n}_version_{version}.png')
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
            with open(f'100_regular_graphs_nodes_{n}_reg_3_solutions.pkl', 'rb') as file:
                data = pickle.load(file)
            list_exact = data.copy()

        for p in ps:
            cuts_qtensor = 0
            cuts_qtensor_recalc_3 = 0
            cuts_qtensor_recalc_6 = 0
            cuts_single = 0
            counter_single = 0
            counter_qtensor = 0
            counter_qtensor_recalc_3 = 0
            counter_qtensor_recalc_6 = 0
            for run in runs:
             
                try:
                    with open (my_path + f"/data/results_run_{run}_n_{n}_p_{p}_recalc_6_version_{version}.pkl", 'rb') as f:
                        data = pickle.load(f)
                    cuts = data['cuts_qtensor']
                    cuts_norm = cuts/list_exact[run]
                    cuts_qtensor_recalc_6 += cuts_norm
                    counter_qtensor_recalc_6 += 1
                except:
                    print(f'file results_run_{run}_n_{n}_p_{p}_recalc_6_version_{version}.pkl is not available')

                try: 
                    with open (my_path + f"/data/results_run_{run}_n_{n}_p_{p}_recalc_3_version_{version}.pkl", 'rb') as f:
                        data = pickle.load(f)
                    cuts = data['cuts_qtensor']
                    cuts_norm = cuts/list_exact[run]
                    cuts_qtensor_recalc_3 += cuts_norm
                    counter_qtensor_recalc_3 += 1
                except:
                    print(f'file results_run_{run}_n_{n}_p_{p}_recalc_3_version_{version}.pkl is not available')
                    
                try:
                    with open (my_path + f"/data/results_run_{run}_n_{n}_p_{p}_wo_recalc_version_{version}.pkl", 'rb') as f:
                        data = pickle.load(f)
                    cuts = data['cuts_qtensor']
                    cuts_norm = cuts/list_exact[run]
                    cuts_qtensor += cuts_norm
                    counter_qtensor += 1
                    if p == 1:
                            cuts = data['cuts_single']
                            cuts_norm = cuts/list_exact[run]
                            cuts_single += cuts_norm
                            counter_single += 1
                except:
                    print(f'file results_run_{run}_n_{n}_p_{p}_wo_recalc_version_{version}.pkl is not available')
            
            cuts_list.append(cuts_qtensor/counter_qtensor)
            print('qtensor:', counter_qtensor)
            
            cuts_recalc_3_list.append(cuts_qtensor_recalc_3/(counter_qtensor_recalc_3))
            cuts_recalc_6_list.append(cuts_qtensor_recalc_6/(counter_qtensor_recalc_6))
            if p==1:
                cuts_single_list.append(cuts_single/counter_single)

        plt.scatter([1], cuts_single_list, c=colors[counter], label = 'cuts analytic for p=1')
        counter += 1
        plt.plot(ps, cuts_list, c=colors[counter], label = f'tensor network cuts for p={p} with 1-recalc')
        counter += 1
        plt.plot(ps, cuts_recalc_3_list, c=colors[counter], label = f'tensor network cuts for p={p} with 3-recalc')
        counter += 1
        plt.plot(ps, cuts_recalc_6_list, c=colors[counter], label = f'tensor network cuts for p={p} with 6-recalc')

        for i in range(3):
            plt.text(ps[i], cuts_list[i], cuts_list[i])
            plt.text(ps[i], cuts_recalc_3_list[i], cuts_recalc_3_list[i])
            plt.text(ps[i], cuts_recalc_6_list[i], cuts_recalc_6_list[i])

        plt.text(1, cuts_single_list[0]*(1-0.002), cuts_single_list[0])

        plt.xticks([1, 2, 3], [1, 2, 3])
        plt.ylabel('Optimal cuts ratio')
        plt.xlabel('p')

        plt.legend()
        #fig.savefig(my_path + f'/results/Cuts_different_calculations_per_p_n_{n}_version_{version}.png')
        plt.show()

def plot_cuts_per_graph(ns, ps, runs, regularity, version):
    colors=['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:pink', 'tab:brown', 'tab:grey', 'tab:olive', 'tab:cyan']
    markers = ['s', 'D', 'v', '.']
    linestyles = ['solid', 'dashed', 'dashdot', 'dotted']
    for n in ns:
        list_exact = []
        list_x = []
        counter = 0
        ns_graphs_rudi = list(range(60, 220, 20))
        ns_graphs_maxi = [30, 50]

        fig = plt.figure()
        plt.title(f'Optimal cuts ratio for MAXCUT graphs with {n} nodes for different types of calculations')


        if n in ns_graphs_rudi:
            with open(my_path + f"/data/regular_graphs_maxcuts.json", 'r') as f:
                graphs = json.load(f)
            for run in range(100): 
                optimal_cuts = graphs[str(regularity)][str(n)][str(run)]['optimal_cut']
                list_exact.append(optimal_cuts)

        elif n in ns_graphs_maxi:
            with open(f'100_regular_graphs_nodes_{n}_reg_3_solutions.pkl', 'rb') as file:
                data = pickle.load(file)
            list_exact = data.copy()
        
        for p in ps:

            list_qtensor_cuts = []
            list_qtensor_cuts_norm = []
            list_graphs_qtensor = []

            if p == 1:
                list_graphs_single = []
                list_single_cuts_norm = []
                list_single_cuts = []

            for run in runs:
                
                try:
                    with open (my_path + f"/data/results_run_{run}_n_{n}_p_{p}_wo_recalc_version_{version}.pkl", 'rb') as f:
                        data = pickle.load(f)
                    cuts_qtensor = data['cuts_qtensor']
                    list_qtensor_cuts.append(cuts_qtensor)
                    list_qtensor_cuts_norm.append(cuts_qtensor/list_exact[run])
                    list_graphs_qtensor.append(run)
                    if p == 1:
                        cuts_single = data['cuts_single']
                        list_single_cuts.append(cuts_single)
                        list_single_cuts_norm.append(cuts_single/list_exact[run])
                        list_graphs_single.append(run)

                except:
                    print(f'file for {n} nodes, p={p} and run {run} is not available')

            if p==1: 
                counter_cuts = 0
                average_list = []
                for cut in list_single_cuts_norm:
                    counter_cuts += cut
                average = counter_cuts/len(list_graphs_single)
                for i in list_graphs_single:
                    average_list.append(average)
                plt.scatter(list_graphs_single, list_single_cuts_norm, c=colors[counter], s = 100, marker = markers[counter], label = 'cuts analytic for p=1')
                plt.plot(list_graphs_single, average_list, color=colors[counter], linestyle=linestyles[counter], label = 'average cut ratio with analytic p=1')
                counter += 1    

            counter_cuts = 0
            average_list = []
            for cut in list_qtensor_cuts_norm:
                counter_cuts += cut
            average = counter_cuts/len(list_graphs_qtensor)
            for i in list_graphs_qtensor:
                average_list.append(average)

            plt.scatter(list_graphs_qtensor, list_qtensor_cuts_norm, c=colors[counter], marker = markers[counter], label = f'tensor network cuts for p={p} with 1-recalc')   
            plt.plot(list_graphs_qtensor, average_list, color=colors[counter], linestyle=linestyles[counter], label = f'average cut ratio with tensor network p={p}')
            counter += 1



        plt.xticks(runs, runs)
        plt.xlabel('Graphs')
        plt.ylabel('Optimal cuts ratio')
        plt.ylim((0.952, 1.002))
        plt.legend(loc=4)
        #fig.savefig(my_path + f'/results/Cuts_different_calculations_per_graph_n_{n}_version_{version}.png')
        plt.show()
        #plt.close()

def plot_cuts_per_p(ns, ps, runs, regularity, version):
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
            with open(f'100_regular_graphs_nodes_{n}_reg_3_solutions.pkl', 'rb') as file:
                data = pickle.load(file)
            list_exact = data.copy()

        for p in ps:
            cuts_qtensor = 0
            cuts_qtensor_recalc_3 = 0
            cuts_qtensor_recalc_6 = 0
            cuts_single = 0
            counter_single = 0
            counter_qtensor = 0
            counter_qtensor_recalc_3 = 0
            counter_qtensor_recalc_6 = 0
            for run in runs:
             
                try:
                    with open (my_path + f"/data/results_run_{run}_n_{n}_p_{p}_recalc_6_version_{version}.pkl", 'rb') as f:
                        data = pickle.load(f)
                    cuts = data['cuts_qtensor']
                    cuts_norm = cuts/list_exact[run]
                    cuts_qtensor_recalc_6 += cuts_norm
                    counter_qtensor_recalc_6 += 1
                except:
                    print(f'file results_run_{run}_n_{n}_p_{p}_recalc_6_version_{version}.pkl is not available')

                try: 
                    with open (my_path + f"/data/results_run_{run}_n_{n}_p_{p}_recalc_3_version_{version}.pkl", 'rb') as f:
                        data = pickle.load(f)
                    cuts = data['cuts_qtensor']
                    cuts_norm = cuts/list_exact[run]
                    cuts_qtensor_recalc_3 += cuts_norm
                    counter_qtensor_recalc_3 += 1
                except:
                    print(f'file results_run_{run}_n_{n}_p_{p}_recalc_3_version_{version}.pkl is not available')
                    
                try:
                    with open (my_path + f"/data/results_run_{run}_n_{n}_p_{p}_wo_recalc_version_{version}.pkl", 'rb') as f:
                        data = pickle.load(f)
                    cuts = data['cuts_qtensor']
                    cuts_norm = cuts/list_exact[run]
                    cuts_qtensor += cuts_norm
                    counter_qtensor += 1
                    if p == 1:
                            cuts = data['cuts_single']
                            cuts_norm = cuts/list_exact[run]
                            cuts_single += cuts_norm
                            counter_single += 1
                except:
                    print(f'file results_run_{run}_n_{n}_p_{p}_wo_recalc_version_{version}.pkl is not available')
            
            cuts_list.append(cuts_qtensor/counter_qtensor)
            print('qtensor:', counter_qtensor)
            
            cuts_recalc_3_list.append(cuts_qtensor_recalc_3/(counter_qtensor_recalc_3))
            cuts_recalc_6_list.append(cuts_qtensor_recalc_6/(counter_qtensor_recalc_6))
            if p==1:
                cuts_single_list.append(cuts_single/counter_single)

        plt.scatter([1], cuts_single_list, c=colors[counter], label = 'cuts analytic for p=1')
        counter += 1
        plt.plot(ps, cuts_list, c=colors[counter], label = f'tensor network cuts for p={p} with 1-recalc')
        counter += 1
        plt.plot(ps, cuts_recalc_3_list, c=colors[counter], label = f'tensor network cuts for p={p} with 3-recalc')
        counter += 1
        plt.plot(ps, cuts_recalc_6_list, c=colors[counter], label = f'tensor network cuts for p={p} with 6-recalc')

        for i in range(3):
            plt.text(ps[i], cuts_list[i], cuts_list[i])
            plt.text(ps[i], cuts_recalc_3_list[i], cuts_recalc_3_list[i])
            plt.text(ps[i], cuts_recalc_6_list[i], cuts_recalc_6_list[i])

        plt.text(1, cuts_single_list[0]*(1-0.002), cuts_single_list[0])

        plt.xticks([1, 2, 3], [1, 2, 3])
        plt.ylabel('Optimal cuts ratio')
        plt.xlabel('p')

        plt.legend()
        #fig.savefig(my_path + f'/results/Cuts_different_calculations_per_p_n_{n}_version_{version}.png')
        plt.show()



def plot_cuts_per_graph_recalculation(ns, ps, runs, regularity, recalculation, iteration, version):
    colors=['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:pink', 'tab:brown', 'tab:grey', 'tab:olive', 'tab:cyan']
    markers = ['s', 'D', 'v', '.']
    linestyles = ['solid', 'dashed', 'dashdot', 'dotted']
    for n in ns:
        list_exact = []
        list_x = []
        counter = 0
        ns_graphs_rudi = list(range(60, 220, 20))
        ns_graphs_maxi = [30, 50]

        fig = plt.figure()
        plt.title(f'Optimal cuts ratio for MAXCUT graphs with {n} nodes for different types of calculations')


        if n in ns_graphs_rudi:
            with open(my_path + f"/data/regular_graphs_maxcuts.json", 'r') as f:
                graphs = json.load(f)
            for run in range(100): 
                optimal_cuts = graphs[str(regularity)][str(n)][str(run)]['optimal_cut']
                list_exact.append(optimal_cuts)

        elif n in ns_graphs_maxi:
            with open(f'100_regular_graphs_nodes_{n}_reg_3_solutions.pkl', 'rb') as file:
                data = pickle.load(file)
            list_exact = data.copy()
        
        for p in ps:

            list_qtensor_cuts = []
            list_qtensor_cuts_norm = []
            list_graphs_qtensor = []

            if p == 1:
                list_graphs_single = []
                list_single_cuts_norm = []
                list_single_cuts = []

            for run in runs:
                
                try:
                    with open (my_path + f"/data/results_run_{run}_iteration_{iteration}_n_{n}_p_{p}_recalc_{recalculation}_initialization_fixed_angles_optimization_version_{version}.pkl", 'rb') as f:
                        data = pickle.load(f)
                    cuts_qtensor = data['cuts_qtensor']
                    list_qtensor_cuts.append(cuts_qtensor)
                    list_qtensor_cuts_norm.append(cuts_qtensor/list_exact[run])
                    list_graphs_qtensor.append(run)
                    print(data['connectivity'])
                    if p == 1:
                        cuts_single = data['cuts_single']
                        list_single_cuts.append(cuts_single)
                        list_single_cuts_norm.append(cuts_single/list_exact[run])
                        list_graphs_single.append(run)

                except:
                    print(f'file for {n} nodes, p={p} and run {run} is not available')

            if p==1: 
                counter_cuts = 0
                average_list = []
                for cut in list_single_cuts_norm:
                    counter_cuts += cut
                average = counter_cuts/len(list_graphs_single)
                for i in list_graphs_single:
                    average_list.append(average)
                plt.scatter(list_graphs_single, list_single_cuts_norm, c=colors[counter], s = 100, marker = markers[counter], label = 'cuts analytic for p=1')
                plt.plot(list_graphs_single, average_list, color=colors[counter], linestyle=linestyles[counter], label = 'average cut ratio with analytic p=1')
                counter += 1    

            counter_cuts = 0
            average_list = []
            for cut in list_qtensor_cuts_norm:
                counter_cuts += cut
            average = counter_cuts/len(list_graphs_qtensor)
            for i in list_graphs_qtensor:
                average_list.append(average)

            plt.scatter(list_graphs_qtensor, list_qtensor_cuts_norm, c=colors[counter], marker = markers[counter], label = f'tensor network cuts for p={p} with 1-recalc')   
            plt.plot(list_graphs_qtensor, average_list, color=colors[counter], linestyle=linestyles[counter], label = f'average cut ratio with tensor network p={p}')
            counter += 1



        plt.xticks(runs, runs)
        plt.xlabel('Graphs')
        plt.ylabel('Optimal cuts ratio')
        plt.ylim((0.952, 1.002))
        plt.legend(loc=4)
        #fig.savefig(my_path + f'/results/Cuts_different_calculations_per_graph_n_{n}_version_{version}.png')
        plt.show()
        #plt.close()



if __name__ == '__main__':
    ns = [50] #[60, 80, 100, 120, 140, 160, 180, 200]
    ps= [1, 2, 3]
    recalculations = [5]
    regularity = 3
    runs = list(range(5, 10))
    recalculation = 5
    version = 1
    iterations = list(range(5))

    colors=['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:pink', 'tab:brown', 'tab:grey', 'tab:olive', 'tab:cyan']

    my_path = os.path.dirname(__file__)
    my_path = os.path.dirname(my_path)

    #plot_cuts_per_graph(ns, ps, runs, regularity, version)
    #plot_cuts_recalculation_per_p(ns, ps, runs, regularity, version)

    for iteration in iterations:
        plot_cuts_per_graph_recalculation(ns, ps, runs, regularity, recalculation, iteration, version)

    