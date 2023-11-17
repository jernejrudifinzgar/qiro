import networkx as nx
import pickle
import os
import matplotlib.pyplot as plt
import matplotlib
from IPython.display import set_matplotlib_formats
__plot_height = 9.119
matplotlib.rcParams['figure.figsize'] = (1.718*__plot_height, __plot_height)
set_matplotlib_formats('svg')


def plot_energy_optimizer_different_lr(ns, regularity, graphs, optimizers, initializers):
    colors=['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:pink', 'tab:brown', 'tab:pink', 'tab:grey', 'tab:olive']
    my_path = os.path.dirname(__file__)
    my_path = os.path.dirname(my_path)

    for n in ns: 
        for optimizer in optimizers:
            if optimizer == 'SGD':
                learning_rates = [0.0001]#, 0.0005, 0.001]
            elif optimizer == 'RMSprop':
                learning_rates = [0.001]#, 0.005, 0.01, 0.05]
            elif optimizer == 'Adam':
                learning_rates = [0.001, 0.005, 0.01, 0.05]
            
            for initializer in initializers:
                fig = plt.figure()
                fig.suptitle(f'Energies for {initializer} initialization with {optimizer} optimizer with different learning rates for graphs with n = {n} and reg = {regularity}')
                for graph, i in zip(graphs, range(len(graphs))):
                    counter = 1
                    plt.subplot(3,4, i+1)
                    for learning_rate in learning_rates:
                        counter += 1
                        with open(my_path + f'/data/nodes_{n}_reg_{regularity}_graph_{graph}_opt_{optimizer}_lr_{learning_rate}.pkl', 'rb') as file:
                            data = pickle.load(file)

                        max_p = len(data[initializer])
                        ps=list(range(1, max_p+1))

                        if counter == 2:
                            energies_fixed_angles_list=[]
                            energy_single = data['analytic_single_p']['energy']
                            for p in ps:
                                energy_fixed_angles = data['fixed_angles'][f'p={p}']['energy']
                                energies_fixed_angles_list.append(energy_fixed_angles)
                            plt.scatter([1], energy_single, color=colors[0], label=f'Grid search with analytic expression')
                            plt.plot(ps, energies_fixed_angles_list, color=colors[1], label=f'Fixed angles without optimization')

                        energies_list=[]
                        for p in ps:
                            energy = data[initializer][f'p={p}']['energy']
                            energies_list.append(energy)    
                        plt.plot(ps, energies_list, color=colors[counter], label=f'Learning rate = {learning_rate}')

                    if i==0 or i==4 or i==8:
                        plt.ylabel('Energy')
                    if i==6 or i==7 or i==8 or i==9:
                        plt.xlabel('steps')
                    plt.xticks(ps, ps)
                    plt.title(f'Graph number {i+1}')
                plt.legend(loc='lower left', bbox_to_anchor=(1,0.4))
                fig.savefig(my_path + f'/graphs/Energies_reg_{regularity}_n_{n}_opt_{optimizer}_init_{initializer}.png')
                plt.close()
    return None

def plot_losses_optimizer_different_lr(ns, regularity, graphs, optimizers, initializers):
    colors=['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:pink', 'tab:brown', 'tab:pink', 'tab:grey', 'tab:olive']
    my_path = os.path.dirname(__file__)
    my_path = os.path.dirname(my_path)

    for n in ns: 
        for optimizer in optimizers:
            if optimizer == 'SGD':
                learning_rates = [0.0001]#, 0.0005, 0.001]
            elif optimizer == 'RMSprop':
                learning_rates = [0.001]#, 0.005, 0.01, 0.05]
            elif optimizer == 'Adam':
                learning_rates = [0.001, 0.005, 0.01, 0.05]
            
            for initializer in initializers:
                with open(my_path + f'/data/nodes_{n}_reg_{regularity}_graph_{0}_opt_{optimizer}_lr_{learning_rates[0]}.pkl', 'rb') as file:
                    data = pickle.load(file)
                max_p = len(data[initializer])
                ps=list(range(1, max_p+1))
                x = list(range(1, len(data[initializer][f'p={2}']['losses'])+1))
                
                for p in ps:

                    if p==1 and initializer=='transition_states':
                        pass
                    else:
                        fig = plt.figure()
                        fig.suptitle(f'Losses for p={p} with {initializer} initialization and {optimizer} optimizer with different learning rates for graphs with n = {n} and reg = {regularity}')
                    
                        for graph, i in zip(graphs, range(len(graphs))):

                            counter = 0
                            plt.subplot(3,4, i+1)

                            if p==1:
                                with open(my_path + f'/data/nodes_{n}_reg_{regularity}_graph_{graph}_opt_{optimizer}_lr_{learning_rates[0]}.pkl', 'rb') as file:
                                    data = pickle.load(file)
                                energy_single = data['analytic_single_p']['energy']
                                plt.scatter([50], energy_single, color=colors[0], label=f'Grid search with analytic expression')
                        
                            for learning_rate in learning_rates:
                                counter += 1

                                with open(my_path + f'/data/nodes_{n}_reg_{regularity}_graph_{graph}_opt_{optimizer}_lr_{learning_rate}.pkl', 'rb') as file:
                                    data = pickle.load(file)

                                losses = data[initializer][f'p={p}']['losses']    
                                plt.plot(x, losses, color=colors[counter], label=f'Learning rate = {learning_rate}')

                            if i==0 or i==4 or i==8:
                                plt.ylabel('Energy')
                            if i==6 or i==7 or i==8 or i==9:
                                plt.xlabel('Steps')
                            plt.title(f'Graph number {i+1}')
                        plt.legend(loc='lower left', bbox_to_anchor=(1,0.4))
                        fig.savefig(my_path + f'/graphs/Losses_reg_{regularity}_n_{n}_p_{p}_opt_{optimizer}_init_{initializer}.png')
                        plt.close()


graphs=[0, 1, 2, 3, 4]
ns=[50, 100, 150, 200]
regularity=3
optimizers_list = ['SGD', 'RMSprop', 'Adam']
learning_rates_SGD = [0.0001, 0.0005, 0.001]
learning_rates_RMSprop = [0.001, 0.005, 0.01, 0.05]
learning_rates_Adam = [0.001, 0.005, 0.01, 0.05]

graphs=[0, 1]
ns=[4]
regularity=3
optimizers_list = ['SGD', 'RMSprop']
initializers_list = ['random_init', 'transition_states', 'fixed_angles_optimization']

if __name__ == '__main__':
    plot_energy_optimizer_different_lr(ns, regularity, graphs, optimizers_list, initializers_list)
    plot_losses_optimizer_different_lr(ns, regularity, graphs, optimizers_list, initializers_list)
