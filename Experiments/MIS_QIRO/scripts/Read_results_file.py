import networkx as nx
import pickle 
import matplotlib.pyplot as plt
import matplotlib
from IPython.display import set_matplotlib_formats
__plot_height = 9.119
matplotlib.rcParams['figure.figsize'] = (1.718*__plot_height, __plot_height)
set_matplotlib_formats('svg')
import os
import sys
sys.path.append("../../../Qtensor")
sys.path.append("../../../Qtensor/qtree_git")
sys.path.append("../../../classical_benchmarks")
from greedy_mis import greedy_mis

def plot_MIS_size_per_graph(ns, ps, runs, version, regularity):
    colors=['tab:blue', 'tab:orange', 'tab:pink', 'tab:red', 'tab:cyan', 'tab:green', 'tab:brown', 'tab:grey', 'tab:olive', 'tab:purple']
    markers = ['s', 'D', 'v', '*', '.']
    linestyles = ['solid', 'dashed', 'dashdot', 'dotted', (0, (3, 5, 1, 5, 1, 5))]
    my_path = os.path.dirname(__file__)
    my_path = os.path.dirname(my_path)
    counter = 0
    for n in ns:
        fig = plt.figure()
        plt.title(f"MIS size of graphs with {n} nodes for different types of calculation")

        for p in ps:
            MIS_size_qtensor_list = []
            list_graphs_qtensor = []
            
            if p==1:
                MIS_size_single_list = []
                list_graphs_single = []
                MIS_size_greedy_list = []


            for run in runs:
                try:
                    with open(my_path + f"/data/results_run_{run}_n_{n}_p_{p}_version_{version}.pkl", 'rb') as file:
                        data = pickle.load(file)
                    MIS_size_qtensor_list.append(data['size_solution_qtensor'])
                    list_graphs_qtensor.append(run)

                    if p==1:
                        MIS_size_single_list.append(data['size_solution_single'])
                        list_graphs_single.append(run)
                        MIS_size_greedy_list.append(data['size_solution_greedy'])
                except:
                    print(f'file results_run_{run}_n_{n}_p_{p}_version_{version}.pkl not available')
            
            if p==1:

                counter_size = 0
                average_list = []
                for size in MIS_size_greedy_list:
                    counter_size += size
                average = counter_size/len(list_graphs_single)
                for i in list_graphs_single:
                    average_list.append(average)
                plt.scatter(list_graphs_single, MIS_size_greedy_list, c=colors[counter], s=100, marker = markers[counter], label = f'greedy algorithm')
                plt.plot(list_graphs_single, average_list, color=colors[counter], linestyle=linestyles[counter], label = 'average MIS size with greedy algorithm')
                counter +=1

                counter_size = 0
                average_list = []
                for size in MIS_size_single_list:
                    counter_size += size
                average = counter_size/len(list_graphs_single)
                for i in list_graphs_single:
                    average_list.append(average)
                plt.scatter(list_graphs_single, MIS_size_single_list, c=colors[counter], marker = markers[counter], label = f'analytic simulation with p=1')
                plt.plot(list_graphs_single, average_list, color=colors[counter], linestyle=linestyles[counter], label = 'average MIS size with analytic p=1')
                counter +=1


            counter_size = 0
            average_list = []
            for size in MIS_size_qtensor_list:
                counter_size += size
            average = counter_size/len(list_graphs_qtensor)
            for i in runs:
                average_list.append(average)
            plt.scatter(list_graphs_qtensor, MIS_size_qtensor_list, c=colors[counter], marker = markers[counter], label = f'tensor network simulation for p={p}')
            plt.plot(runs, average_list, color=colors[counter], linestyle=linestyles[counter], label = f'average MIS size with tensor network p={p}')
            counter += 1
            
        plt.xticks(runs, list(range(len(MIS_size_single_list))))
        plt.xlabel('Graphs')
        plt.ylabel('MIS size')
        plt.legend()    
        plt.show()
        #fig.savefig(my_path + f'/results/MIS_size_reg_{regularity}_n_{n}_runs_{runs[0]}_{runs[-1]}_version_{version}.png')


def plot_energies(ns, ps, runs, version, regularity, per_node=False):
    colors=['tab:blue', 'tab:orange', 'tab:pink', 'tab:red', 'tab:cyan', 'tab:green', 'tab:brown', 'tab:grey', 'tab:olive', 'tab:purple']
    markers = ['s', 'D', 'v', '*', '.']
    linestyles = ['solid', 'dashed', 'dashdot', 'dotted', (0, (3, 5, 1, 5, 1, 5))]
    my_path = os.path.dirname(__file__)
    my_path = os.path.dirname(my_path)
    for n in ns:
        fig = plt.figure()
        fig.suptitle(f"QAOA energies of QIRO steps for graphs with {n} nodes for different types of QAOA calculation")
        for run, i in zip(runs, range(len(runs))):
            counter = 0
            plt.subplot(3, 4, i+1)
            for p in ps:
                try:
                    with open(my_path + f"/data/results_run_{run}_n_{n}_p_{p}_version_{version}.pkl", 'rb') as file:
                        data = pickle.load(file)
                    
                    if p==1:
                        num_nodes_single = data['num_nodes_single']
                        energies_single = [float(j) for j in data['energies_single']]
                        if per_node==True:
                            energies_single = [energies_single[j]/num_nodes_single[j] for j in range(len(num_nodes_single))]
                        plt.plot(list(range(len(energies_single))), energies_single, color=colors[counter], linestyle=linestyles[counter], label = f'Energies in analytic calculation')
                        counter += 1

                    num_nodes_qtensor = data['num_nodes_qtensor']
                    energies_qtensor = [float(j) for j in data['energies_qtensor']]
                    if per_node==True:
                        energies_qtensor = [energies_qtensor[j]/num_nodes_qtensor[j] for j in range(len(num_nodes_qtensor))]
                    plt.plot(list(range(len(energies_qtensor))), energies_qtensor, color=colors[counter], linestyle=linestyles[counter], label = f'Energies tensor network p={p} simulation')
                    counter += 1
                except:
                    print(f'file results_run_{run}_n_{n}_p_{p}_version_{version}.pkl not available')

            if i==0 or i==4 or i==8:
                plt.ylabel('Energy')
            if i==6 or i==7 or i==8 or i==9:
                plt.xlabel('Qiro steps')
            plt.title(f'Graph {run+1}')
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.25)
        plt.legend(loc='lower left', bbox_to_anchor=(1,0.4))
        plt.show()
        fig.savefig(my_path + f'/results/Energies_reg_{regularity}_n_{n}_runs_{runs[0]}_{runs[9]}_version_{version}.png')


def plot_losses(ns, ps, runs, version, regularity, per_node=False):
    colors=['tab:blue', 'tab:orange', 'tab:pink', 'tab:red', 'tab:cyan', 'tab:green', 'tab:brown', 'tab:grey', 'tab:olive', 'tab:purple']
    markers = ['s', 'D', 'v', '*', '.']
    linestyles = ['solid', 'dashed', 'dashdot', 'dotted', (0, (3, 5, 1, 5, 1, 5)), (0, (3, 1, 1, 1, 1, 1))]
    my_path = os.path.dirname(__file__)
    my_path = os.path.dirname(my_path)
    for n in ns:
        for run, i in zip(runs, range(len(runs))):
            fig = plt.figure()
            fig.suptitle(f"QAOA optimization losses of QIRO steps for graph number {run} with {n} nodes for different types of QAOA calculation")
            counter = 0
            for p in ps:
                try:
                    with open(my_path + f"/data/results_run_{run}_n_{n}_p_{p}_version_{version}.pkl", 'rb') as file:
                        data = pickle.load(file)
                    num_nodes_qtensor = data['num_nodes_qtensor']
                    losses = data['losses_qtensor']
                    if p ==1:
                        energies_single = data['energies_single']
                        num_nodes_single = data['num_nodes_single']

                    for j in range(14):
                        plt.subplot(3, 5, j+1)
                        if p==1:
                            energy_single = float(energies_single[j])
                            if per_node==True:
                                energy_single = energy_single/num_nodes_single[j]
                            plt.scatter([49], [energy_single], c=colors[-1], label='Energy of p=1 analytic solution')

                        try:
                            losses_per_step = losses[j]
                            if per_node==True:
                                losses_per_step = [losses_per_step[k]/num_nodes_qtensor[j] for k in range(len(losses_per_step))]
                        
                            plt.plot(list(range(len(losses_per_step))), losses_per_step, color=colors[counter], linestyle=linestyles[counter], label = f'Losses tensor network p={p} simulation')
                            plt.title(f'QIRO step {j}')
                            if j==0 or j==5 or j==10:
                                plt.ylabel('Loss')
                            if j==9 or j==10 or j==11 or j==12 or j==13:
                                plt.xlabel('Optimization steps')
                        except:
                            pass
                except:
                    print(f'file results_run_{run}_n_{n}_p_{p}_version_{version}.pkl not available')
                counter += 1

            lines = []
            labels = []
            
            line, label = fig.axes[0].get_legend_handles_labels()
            lines.extend(line)
            labels.extend(label)
            plt.subplots_adjust(left=0.1, bottom=None, right=None, top=None, wspace=0.4, hspace=0.35)
            plt.legend(lines, labels, loc='lower left', bbox_to_anchor=(1,0.2))
            plt.show()
            #fig.savefig(my_path + f'/results/Losses_reg_{regularity}_n_{n}_run_{run}_version_{version}.png')

def plot_losses_multiple_versions(ns, ps, runs, versions, regularity, per_node=False):
    colors=['tab:blue', 'tab:orange', 'tab:pink', 'tab:red', 'tab:cyan', 'tab:green', 'tab:brown', 'tab:grey', 'tab:olive', 'tab:purple']
    markers = ['s', 'D', 'v', '*', '.']
    linestyles = ['solid', 'dashed', 'dashdot', 'dotted', (0, (3, 5, 1, 5, 1, 5)), (0, (3, 1, 1, 1, 1, 1))]
    my_path = os.path.dirname(__file__)
    my_path = os.path.dirname(my_path)
    for n in ns:
        for run, i in zip(runs, range(len(runs))):
            fig = plt.figure()
            fig.suptitle(f"QAOA optimization losses of QIRO steps for graph number {run} with {n} nodes for different types of QAOA calculation")
            counter = 0
            for p in ps:

                for version in versions:
                    try:
                        with open(my_path + f"/data/results_run_{run}_n_{n}_p_{p}_version_{version}.pkl", 'rb') as file:
                            data = pickle.load(file)
                        num_nodes_qtensor = data['num_nodes_qtensor']
                        losses = data['losses_qtensor']

                        for j in range(14):
                            plt.subplot(3, 5, j+1)
                            losses_per_step = losses[j]
                            if per_node==True:
                                losses_per_step = [losses_per_step[k]/num_nodes_qtensor[j] for k in range(len(losses_per_step))]
                        
                            plt.plot(list(range(len(losses_per_step))), losses_per_step, color=colors[counter], linestyle=linestyles[counter], label = f'Losses tensor network p={p} simulation verion {version}')
                            plt.title(f'QIRO step {j}')
                            if j==0 or j==5 or j==10:
                                plt.ylabel('Loss')
                            if j==9 or j==10 or j==11 or j==12 or j==13:
                                plt.xlabel('Optimization steps')

                    except:
                        print(f'file results_run_{run}_n_{n}_p_{p}_version_{version}.pkl not available')
                    counter += 1

            lines = []
            labels = []
            
            line, label = fig.axes[0].get_legend_handles_labels()
            lines.extend(line)
            labels.extend(label)
            plt.subplots_adjust(left=0.1, bottom=None, right=None, top=None, wspace=0.4, hspace=0.35)
            plt.legend(lines, labels, loc='lower left', bbox_to_anchor=(1,0.2))
            plt.show()
            fig.savefig(my_path + f'/results/Losses_reg_{regularity}_n_{n}_run_{run}_versions_{versions[0]}_{versions[1]}.png')
                
def solve_greedy():
    my_path = os.path.dirname(__file__)
    my_path = os.path.dirname(my_path)

    with open(f'100_regular_graphs_nodes_{30}_reg_{3}.pkl', 'rb') as file:
        data = pickle.load(file)
        
    for run in range(20):
        G = data[run]
        size_greedy = greedy_mis(G)

        with open(my_path + f"/data/results_run_{run}_n_{30}_p_{1}_version_{1}.pkl", 'rb') as file:
            dictio = pickle.load(file)
        dictio['size_solution_greedy'] = size_greedy
        pickle.dump(dictio, open(my_path + f"/data/results_run_{run}_n_{30}_p_{1}_version_{1}.pkl", 'wb'))

if __name__ == '__main__':

    ns = [50]
    ps = [1, 2, 3]
    runs = list(range(0, 20))
    regularity = 3
    versions = [1, 2]
    version = 1
    

    plot_MIS_size_per_graph(ns, ps, runs, version, regularity)
    #plot_energies(ns, ps, runs, version, regularity, per_node=True)
    #plot_losses(ns, ps, runs, version, regularity, per_node=True)
    #plot_losses_multiple_versions(ns, ps, runs, versions, regularity, per_node=True)

        







"""     
    
    fig = plt.figure(figsize=(15,8))
    #ax=fig.add_subplot(111)

    numCases = 8
    p_single=[1]
    p_qtensor=[1,2]#,3]
    cmap = plt.cm.get_cmap('hsv', numCases)

    for i in range(numCases):
        ax=fig.add_subplot(2,4, i+1)
        size_indep_set_min_greedy=[imported_dict[f'reg_3_n_{n}_p_2_numCases_8'][i]['size_indep_set_min_greedy']]
        size_indep_set_single=[imported_dict[f'reg_3_n_{n}_p_2_numCases_8'][i]['output_single_p'][0]]
        size_indep_set_qtensor = []
        for p in p_qtensor:

            imported_dict_p = pickle.load(open(f'results_run_1_reg_3_n_{60}_p_{p}_numCases_8.pkl', 'rb'))
            size_indep_set_qtensor.append(imported_dict_p[f'reg_3_n_{60}_p_{p}_numCases_8'][i]['output_qtensor'][f'p={p}'][0])

        ax.plot(p_single, size_indep_set_min_greedy, 'o', c=cmap(i), label=f'Min Greedy')
        ax.plot(p_single, size_indep_set_single, 'x', c=cmap(i), label=f'Single', markersize= 15)
        ax.plot(p_qtensor, size_indep_set_qtensor, '-.', c=cmap(i), label='Qtensor', markersize=30)
        plt.legend()
    
    fig.suptitle(f'Independent set sizes for n = {n}')
    fig.savefig(f'results_run_1_reg_3_n_{n}_numCases_8.png')
    plt.show() """



