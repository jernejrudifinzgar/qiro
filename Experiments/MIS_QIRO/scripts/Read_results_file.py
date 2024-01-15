import networkx as nx
import pickle 
import matplotlib.pyplot as plt
import matplotlib
from IPython.display import set_matplotlib_formats
__plot_height = 9.119
matplotlib.rcParams['figure.figsize'] = (1.718*__plot_height, __plot_height)
set_matplotlib_formats('svg')
import os

def plot_MIS_size_per_graph(ns, ps, runs, version):
    colors=['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:pink', 'tab:brown', 'tab:grey', 'tab:olive', 'tab:cyan']
    my_path = os.path.dirname(__file__)
    my_path = os.path.dirname(my_path)
    counter = 0
    for n in ns:
        fig = plt.figure()
        plt.title(f"MIS size of graphs with {n} nodes for different types of calculation")

        for p in ps:
            MIS_size_qtensor_list = []
            if p==1:
                MIS_size_single_list = []


            for run in runs:
                try:
                    with open(my_path + f"/data/results_run_{run}_n_{n}_p_{p}_version_{version}.pkl", 'rb') as file:
                        data = pickle.load(file)
                    MIS_size_qtensor_list.append(data['size_solution_qtensor'])
                    
                    print(data['solution_qtensor'])
                            #print(data['solution_single'])
                    if p==1:
                        MIS_size_single_list.append(data['size_solution_single'])
                except:
                    print(f'file results_run_{run}_n_{n}_p_{p}_version_{version}.pkl not available')
            
            if p==1:
                plt.scatter(list(range(len(MIS_size_single_list))), MIS_size_single_list, c=colors[counter], label = f'analytic simulation with p=1')
                counter +=1
            plt.scatter(list(range(len(MIS_size_qtensor_list))), MIS_size_qtensor_list, c=colors[counter], label = f'tensor network simulation for p={p}')
            counter += 1
            
        plt.xticks(runs, list(range(len(MIS_size_qtensor_list))))
        plt.xlabel('Graphs')
        plt.ylabel('Optimal cuts ratio')
        plt.legend()    
        plt.show()

                












if __name__ == '__main__':

    ns = [30]
    ps = [1, 2, 3]
    runs = list(range(40))
    version = 1

    plot_MIS_size_per_graph(ns, ps, runs, version)
        







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



