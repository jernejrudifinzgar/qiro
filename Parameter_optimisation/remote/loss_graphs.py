import pickle
import networkx as nx

import matplotlib.pyplot as plt
import matplotlib
from IPython.display import set_matplotlib_formats
__plot_height = 8.919
matplotlib.rcParams['figure.figsize'] = (1.718*__plot_height, __plot_height)
set_matplotlib_formats('svg')

if __name__ == '__main__':

    version = 3
    reg = 3
    n = 50
    ps = [1, 2, 3]
    num_runs = 10
    instance = 0

    colors=['tab:red', 'tab:purple', 'tab:green', 'tab:blue', 'tab:orange', 'tab:brown', 'tab:pink', 'tab:grey', 'tab:olive']

    with open(f'energies_reg_{reg}_nodes_{n}_version_{version}.pkl', 'rb') as file:
        data_list = pickle.load(file)

    num_random = len(data_list[0]['random_init'][f'p={ps[1]}'])

    
    print(data_list[0]['random_init'][f'p={2}'])

    for p in ps:
         
        fig = plt.figure()
        fig.suptitle(f'Optimization losses for depth {p} with different initialization methods for graphs with n = {n} and reg = {reg}')
        for i in range(num_runs):
            plt.subplot(3,4,i+1)
            losses_fixed_angles_initialization = data_list[i]['fixed_angles_optimization'][f'p={p}']['losses'].copy()
            x = list(range(1, len(losses_fixed_angles_initialization)+1))
            plt.plot(x, losses_fixed_angles_initialization, colors[0], label=f'Optimization losses with fixed angles initialization')

            if p != 1:
                losses_transition_states_initialization = data_list[i]['transition_states'][f'p={p}']['losses'].copy()
                plt.plot(x, losses_transition_states_initialization, colors[1], label=f'Optimization losses with transition states initialization')
    
            
            for j in range(num_random):
                losses_random_initialization = data_list[i]['random_init'][f'p={p}'][j]['losses'].copy()
                #print(losses_random_initialization)
                plt.plot(x, losses_random_initialization, colors[4+j], label=f'Optimization losses with random initialization {j+1}')

            if i==0 or i==4 or i==8:
                plt.ylabel('Loss')
            if i==6 or i==7 or i==8 or i==9:
                plt.xlabel('Steps')
            #plt.xticks(ps, ps)
            plt.title(f'Run number {i+1}')

        plt.legend(loc='lower left', bbox_to_anchor=(1,0.4))
        fig.savefig(f'Losses_reg_{reg}_n_{n}_p_{p}_version_{version}.png')

    