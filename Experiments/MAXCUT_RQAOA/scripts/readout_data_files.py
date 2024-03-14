import networkx as nx
import pickle
import json
import sys
sys.path.append("./../../../../miniconda3/envs/qiro/lib/python3.9/site-packages/")
import os
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patheffects as pe
from IPython.display import set_matplotlib_formats
#__plot_height = 8.7
#matplotlib.rcParams['figure.figsize'] = (1.718*__plot_height, __plot_height)
#set_matplotlib_formats('svg')
import numpy as np
import matplotlib
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
 
os.environ['PATH'] = os.environ['PATH'] + ':/Library/TeX/texbin/' # for latex, you might need to change this
#os.environ['PATH'] = os.environ['PATH'] + ':/usr/bin/java/' # for latex, you might need to change this


#print(os.environ['PATH'])# = os.environ['PATH'] + 'c:/Library/TeX/texbin/' # for latex, you might need to change this
label_size = 8
# plt.style.use('fivethirtyeight')
plt.rc("text", usetex=True)
plt.rc("text.latex", preamble=r'\usepackage{dsfont}\usepackage{amsmath}\usepackage{physics}')
#matplotlib.rcParams['text.latex.preamble']=[r'/usepackage{dsfont}/usepackage{amsmath}/usepackage{physics}']
 
pr ={"axes.labelsize": 8,               # LaTeX default is 10pt font.
    "font.size": 8,
    "legend.fontsize": 8,               # Make the legend/label fonts
    "xtick.labelsize": 8,               # a little smaller
    "ytick.labelsize": 8,
    'figure.figsize': (2.0476 * 3.35, 1.* 9 * 3.35 / 16),
    "errorbar.capsize": 1.5,
    "font.family": "serif",
    "font.serif": [],                   # blank entries should cause plots
    "font.sans-serif": [],    
    }
for k, v in pr.items():
    matplotlib.rcParams[k] = v
   
matplotlib.rcParams["font.family"] = "serif"
# mpl.rcParams["font.serif"] = ["STIX"]
matplotlib.rcParams["mathtext.fontset"] = "stix"


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
                    #print(data['connectivity'])
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

def grouped_bar_chart(ns, ps, runs, regularity, recalculation, iterations, version):
    colors=['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:pink', 'tab:brown', 'tab:grey', 'tab:olive', 'tab:cyan']
    linestyles = ['solid', 'dashed', 'dashdot', 'dotted']

    data_dic = {f'p={p}': [] for p in ps}
    error_dic = {f'p={p}': [[], []] for p in ps}
    #error_dic = {f'p={p}': [] for p in ps}

    list_exact = []
    counter = 0

    ns_graphs_rudi = list(range(60, 220, 20))
    ns_graphs_maxi = [30, 50]

    for n in ns: 
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

            with open(f'100_regular_graphs_nodes_{n}_reg_3.pkl', 'rb') as f:
                graphs = pickle.load(f)

        for p in ps:
            list_energies = []
            list_cuts_overall = []
            for run in runs:
                list_cuts = []
                
                for iteration in iterations:
                    graph = graphs[iteration]
                    num_edges = graph.number_of_edges()
                    try:
                        with open (my_path + f"/data/results_run_{run}_iteration_{iteration}_n_{n}_p_{p}_recalc_{recalculation}_initialization_fixed_angles_optimization_version_{version}.pkl", 'rb') as f:
                            data = pickle.load(f)
                        cuts_qtensor = data['cuts_qtensor']
                        #print(cuts_qtensor)
                        if p==1:
                            energy= (num_edges - float(data['energies_single'][0]))/2
                        else:
                            energy= (num_edges - float(data['energies_qtensor'][0]))/2
                        #print(energy)
                        list_energies.append(energy/list_exact[run])
                        list_cuts.append(cuts_qtensor/list_exact[run])
                        list_cuts_overall.append(cuts_qtensor/list_exact[run])
                    except Exception as error:
                        print(error)

                average = sum(list_cuts)/len(list_cuts)
                data_dic[f'p={p}'].append(average)
                error_dic[f'p={p}'][0].append(round(average-min(list_cuts), 5))
                error_dic[f'p={p}'][1].append(round(max(list_cuts)-average, 5))
                #error_dic[f'p={p}'].append(np.std(list_cuts))
            
            average_bare = np.mean(list_energies)
            std_bare = np.std(list_energies)
            print('bare', p, average_bare, std_bare)

            average_overall = np.mean(list_cuts_overall)
            std_overall = np.std(list_cuts_overall)
            print('overall', p, average_overall, std_overall)

    # try:
    #     data_dic['p=3'][3]=1
    #     print('done')
    # except Exception as error:
    #     print(error)

    x = np.arange(len(runs))
    x2 = np.arange(len(runs)+1)
    x2 = np.insert(x2, 0, -1) # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')

    
    for attribute, measurement in data_dic.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, edgecolor = 'black', linewidth=0.5, yerr = error_dic[attribute], error_kw=dict(lw=1, ecolor='gray', capthick=1), color=colors[6+multiplier], alpha=0.6, label=f'${attribute}$')
        #ax.bar_label(rects, padding=3)
        multiplier += 1

    multiplier = 0
    for attribute, measurement in data_dic.items():
        average = np.mean(measurement)
        print(average)
        plt.plot(x2, [average for i in x2], color=colors[6+multiplier], linestyle='dashed', linewidth=1.3, path_effects=[pe.Stroke(linewidth=1.8, foreground='black'), pe.Normal()], label = f'${attribute}$ mean')    
        #plt.plot(x2, [average for i in x2], color=colors[6+multiplier], linestyle=(5, (10, 3)), linewidth=2, label = f'Average optimal cuts ratio of QAOA with {attribute}')    
        multiplier += 1

    
    handles, labels = ax.get_legend_handles_labels()
    order = [3, 4, 5, 0, 1, 2]
    order = [0, 1, 2, 3, 4, 5]
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Approximation ratio')
    ax.set_xlabel('Problem instance')
    #ax.set_title(f'MAXCUT RQAOA approximation ratio with recalculation every {recalculation}''$^\\text{th}$ shrinking step')
    x_labels = [f'{i+1}' for i in runs]
    ax.set_xticks(x + width, x_labels)
    ax.set_yticks([0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.0], [0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.0])
    #ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc='upper right', ncols = 1, bbox_to_anchor=(1.26, 1.047)) #1.028
    #ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc='upper left', ncols = 6, columnspacing=1.394, bbox_to_anchor=(-0.01, 1.28))
    ax.legend(loc='upper left', ncols = 2)
    ax.set_ylim(0.92, 1.002)
    ax.set_xlim(-0.5, len(runs))
    ax.tick_params(bottom=False)
    box = ax.get_position()
    #ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.85])
    
    #ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc='upper center', handletextpad=0.5, ncols = 6, columnspacing=1, bbox_to_anchor=(0.5, 1.24))


    #fig.savefig(my_path + f'/results/Cuts_ratio_per_graph_per_p_iterations_{len(iterations)}_graphs_{runs[0]}_{runs[-1]}_n_{n}_version_{version}.pdf', format="pdf", dpi=2000)
    plt.show()

def plot_time(ns, ps, runs, regularity, recalculation, iterations, version):
    colors=['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:pink', 'tab:brown', 'tab:grey', 'tab:olive', 'tab:cyan']
    linestyles = ['solid', 'dashed', 'dashdot', 'dotted']

    for n in ns: 
        for p in ps: 
            for run in runs: 
                for iteration in iterations:
                    fig, ax1 = plt.subplots()
                    plt.title(f'Time analysis of RQAOA steps of iteration {iteration} of graph {run} ')

                    ax2 = ax1.twinx()
                    ax3 = ax1.twinx()
                    ax4 = ax1.twinx()
                    counter = 0
                    
                    steps=[]
                    times=[]
                    max_connectivity=[]
                    average_connectivity=[]
                    num_nodes=[]
                    try:
                        with open (my_path + f"/data/time_results_run_{run}_iteration_{iteration}_n_{n}_p_{p}_recalc_{recalculation}_initialization_fixed_angles_optimization_version_{version}.pkl", 'rb') as f:
                            data = pickle.load(f)

                        steps = list(range(int(len(data.keys())/3)))
                        for step in steps:
                            times.append(data[f'{step}_time'])
                            max_connectivity.append(max(data[f'{step}_connectivity']))
                            average_connectivity.append(np.mean(data[f'{step}_connectivity']))
                            num_nodes.append(data[f'{step}_nodes'])
                    except Exception as error:
                        print(error)

                    a1 = ax1.plot(steps, times, color=colors[counter], linestyle = linestyles[0], label=f'Required time per RQAOA step')
                    ax1.set_xlabel('Steps')
                    ax1.set_ylabel('Time (s)')  
                    ax1.yaxis.label.set_color(colors[counter])    
                    counter += 1

                    a2 = ax2.plot(steps, max_connectivity, color=colors[counter], linestyle = linestyles[0], label=f'Maximum connectivity of graph')
                    ax2.set_ylabel('Maximum connectivity') 
                    ax2.yaxis.label.set_color(colors[counter])    
                    counter += 1

                    a3 = ax3.plot(steps, average_connectivity, color=colors[counter], linestyle = linestyles[0], label=f'Average connectivity of graph')
                    ax3.set_ylabel('Average connectivity') 
                    ax3.spines['right'].set_position(('outward', 60))
                    ax3.yaxis.label.set_color(colors[counter])    
                    counter += 1
                    
                    a4 = ax4.plot(steps, num_nodes, color=colors[counter], linestyle = linestyles[0], label=f'Number of nodes in graph')
                    ax4.set_ylabel('Number of nodes')    
                    ax4.spines['right'].set_position(('outward', 120))
                    ax4.yaxis.label.set_color(colors[counter])  
                    counter += 1

                    ax1.legend(handles=a1+a2+a3+a4, loc = 'lower center')

                    fig.tight_layout()
                    fig.savefig(my_path + f'/results/Time_per_step_graph_{run}_p_{p}_iterations_{iteration}_n_{n}_regularity_{regularity}_version_{version}.png')

                    plt.show()

                    

                    











if __name__ == '__main__':
    ns = [50] #[60, 80, 100, 120, 140, 160, 180, 200]
    ps= [1, 2, 3]
    recalculations = [1]
    regularity = 3
    #runs = [0, 1, 2, 3, 5, 6, 7, 9]
    runs = list(range(0, 3)) + list(range(5, 15)) + list(range(16, 18)) + list(range(19, 28)) + list(range(29, 30))
    #runs = list(range(0, 30))
    recalculation = 1
    version = 1
    iterations = list(range(5))

    colors=['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:pink', 'tab:brown', 'tab:grey', 'tab:olive', 'tab:cyan']

    my_path = os.path.dirname(__file__)
    my_path = os.path.dirname(my_path)

    #plot_cuts_per_graph(ns, ps, runs, regularity, version)
    #plot_cuts_recalculation_per_p(ns, ps, runs, regularity, version)

    #for iteration in iterations:
    #    plot_cuts_per_graph_recalculation(ns, ps, runs, regularity, recalculation, iteration, version)

    grouped_bar_chart(ns, ps, runs, regularity, recalculation, iterations, version)

    #plot_time(ns, ps, runs, regularity, recalculation, iterations, version)

    