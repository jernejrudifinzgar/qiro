import pickle
import networkx as nx

import matplotlib.pyplot as plt
import matplotlib
from IPython.display import set_matplotlib_formats
__plot_height = 8.919
matplotlib.rcParams['figure.figsize'] = (1.718*__plot_height, __plot_height)
set_matplotlib_formats('svg')

if __name__ == '__main__':

    version = 2
    reg = 3
    n = 150
    ps = [1, 2, 3]
    num_runs = 10
    instance = 0

    colors=['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']

    with open(f'energies_reg_{reg}_nodes_{n}_version_{version}.pkl', 'rb') as file:
        data_list = pickle.load(file)

    

    for p in ps:
         
        fig = plt.figure()
        fig.suptitle(f'Optimization losses for depth {p} with different initialization methods for graphs with n = {n} and reg = {reg}')
        for i in range(num_runs):

            plt.subplot(3,4,i+1)
            losses_random_initialization = data_list[i]['random_init'][f'p={p}']['losses'].copy()
            losses_fixed_angles_initialization = data_list[i]['fixed_angles_optimization'][f'p={p}']['losses'].copy()

            if p != 1:
                losses_transition_states_initialization = data_list[i]['transition_states'][f'p={p}']['losses'].copy()
                plt.plot(x, losses_transition_states_initialization, colors[4], label=f'Optimization losses with transition states initialization')

            x = list(range(1, len(losses_fixed_angles_initialization)+1))

            plt.plot(x, losses_random_initialization, colors[1], label=f'Optimization losses with random initialization')
            plt.plot(x, losses_fixed_angles_initialization, colors[3], label=f'Optimization losses with fixed angles initialization')
            if i==0 or i==4 or i==8:
                plt.ylabel('Loss')
            if i==6 or i==7 or i==8 or i==9:
                plt.xlabel('Steps')
            #plt.xticks(ps, ps)
            plt.title(f'Run number {i+1}')

        plt.legend(loc='lower left', bbox_to_anchor=(1,0.4))
        fig.savefig(f'Losses_reg_{reg}_n_{n}_p_{p}_version_{version}.png')

    