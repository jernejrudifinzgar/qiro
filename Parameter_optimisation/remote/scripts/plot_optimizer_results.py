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
    colors=['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:pink', 'tab:brown', 'tab:grey', 'tab:olive']
    my_path = os.path.dirname(__file__)
    my_path = os.path.dirname(my_path)

    for n in ns: 
        for optimizer in optimizers:
            if optimizer == 'SGD':
                learning_rates = [0.0001, 0.0005, 0.001]
            elif optimizer == 'RMSprop':
                learning_rates = [0.001, 0.005, 0.01, 0.05]
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
                        plt.xlabel('p')
                    plt.xticks(ps, ps)
                    plt.title(f'Graph number {i+1}')
                plt.legend(loc='lower left', bbox_to_anchor=(1,0.2))
                plt.subplots_adjust(hspace=0.3)
                fig.savefig(my_path + f'/graphs/Energies_reg_{regularity}_n_{n}_opt_{optimizer}_init_{initializer}.png')
                plt.close()
   
def plot_losses_optimizer_different_lr(ns, ps, regularity, graphs, optimizers, initializers):
    colors=['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:pink', 'tab:brown', 'tab:grey', 'tab:olive']
    my_path = os.path.dirname(__file__)
    my_path = os.path.dirname(my_path)

    for n in ns: 
        for optimizer in optimizers:
            if optimizer == 'SGD':
                learning_rates = [0.0001, 0.0005, 0.001]
            elif optimizer == 'RMSprop':
                learning_rates = [0.001, 0.005, 0.01, 0.05]
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

                            counter = 1
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
                        plt.legend(loc='lower left', bbox_to_anchor=(1,0.2))
                        plt.subplots_adjust(hspace=0.3)
                        fig.savefig(my_path + f'/graphs/Losses_reg_{regularity}_n_{n}_p_{p}_opt_{optimizer}_init_{initializer}.png')
                        plt.close()

def energy_list_single_input(n, regularity, ps, graph, learning_rate, optimizer, initializer):
    my_path = os.path.dirname(__file__)
    my_path = os.path.dirname(my_path)
    energy_list=[]
    with open(my_path + f'/data/nodes_{n}_reg_{regularity}_graph_{graph}_opt_{optimizer}_lr_{learning_rate}.pkl', 'rb') as file:
        data = pickle.load(file)
    for p in ps:
        energy = data[initializer][f'p={p}']['energy']
        energy_list.append(energy)
    
    return energy_list

def losses_list_single_input(n, regularity, p, graph, learning_rate, optimizer, initializer):
    my_path = os.path.dirname(__file__)
    my_path = os.path.dirname(my_path)
    losses_list=[]
    with open(my_path + f'/data/nodes_{n}_reg_{regularity}_graph_{graph}_opt_{optimizer}_lr_{learning_rate}.pkl', 'rb') as file:
        data = pickle.load(file)
    losses_list = data[initializer][f'p={p}']['losses']
   
    return losses_list

def plot_energy__different_opt_init_n_50(ns, ps, regularity, graphs):
    colors=['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:pink', 'tab:brown', 'tab:grey', 'tab:olive', 'tab:cyan']
    my_path = os.path.dirname(__file__)
    my_path = os.path.dirname(my_path)

    arguments = [
                #[0.01, 'Adam', 'fixed_angles_optimization'],
                #[0.05, 'Adam', 'random_init'],
                [0.05, 'Adam', 'transition_states'], 
                #[0.001, 'RMSprop', 'fixed_angles_optimization'], 
                #[0.005, 'RMSprop', 'random_init'], 
                #[0.01, 'RMSprop', 'transition_states'],
                [0.0001, 'SGD', 'fixed_angles_optimization'], 
                #[0.0001, 'SGD', 'random_init'], 
                #[0.0001, 'SGD', 'transition_states']
                ]

    for n in ns: 
        fig = plt.figure()
        fig.suptitle(f'Energies for different initializations with optimized optimizers for for graphs with n = {n} and reg = {regularity}')

        for graph, i in zip(graphs, range(len(graphs))):
            fig = plt.figure()
            fig.suptitle(f'Energies for different initializations with optimized optimizers for for graphs with n = {n} and reg = {regularity}')

            if ps[0] == 1:
            
                with open(my_path + f'/data/nodes_{n}_reg_{regularity}_graph_{graph}_opt_SGD_lr_{0.0001}.pkl', 'rb') as file:
                    data = pickle.load(file)
                energy_single = data['analytic_single_p']['energy']
                plt.scatter([1], energy_single, color=colors[0], label=f'Grid search with analytic expression')

            counter = 1
            plt.subplot(1,1, 1)
            for argument in arguments:
                learning_rate = argument[0]
                optimizer = argument[1]
                initialization = argument[2]
                energies = energy_list_single_input(n, regularity, ps, i, learning_rate, optimizer, initialization)

                plt.plot(ps, energies, color=colors[counter], label=f'{initialization} initialization with {optimizer} and lr={learning_rate} optimization')
                counter += 1
                
            #if i==0 or i==4 or i==8:
            plt.ylabel('Energy')
            #if i==6 or i==7 or i==8 or i==9:
            plt.xlabel('p')
            plt.xticks(ps, ps)
            plt.title(f'Graph number {i+1}')
            plt.legend(loc='upper right', bbox_to_anchor=(1,1))
            plt.subplots_adjust(hspace=0.4)
            fig.savefig(my_path + f'/graphs/Energies_reg_{regularity}_n_{n}_graph_{graph}_different_optimized_initializations.png')
            plt.close()

def plot_losses__different_opt_init_n_50(ns, ps, regularity, graphs):
    colors=['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:pink', 'tab:brown', 'tab:grey', 'tab:olive', 'tab:cyan']
    my_path = os.path.dirname(__file__)
    my_path = os.path.dirname(my_path)

    arguments = [
                #[0.01, 'Adam', 'fixed_angles_optimization'],
                #[0.05, 'Adam', 'random_init'],
                #[0.05, 'Adam', 'transition_states'], 
                #[0.001, 'RMSprop', 'fixed_angles_optimization'], 
                #[0.005, 'RMSprop', 'random_init'], 
                #[0.01, 'RMSprop', 'transition_states'],
                [0.0001, 'SGD', 'fixed_angles_optimization'], 
                #[0.0001, 'SGD', 'random_initialiation'], 
                #[0.0001, 'SGD', 'transition_states']
                ]

    for n in ns: 
        with open(my_path + f'/data/nodes_{n}_reg_{regularity}_graph_{0}_opt_Adam_lr_{0.001}.pkl', 'rb') as file:
            data = pickle.load(file)
        x = list(range(1, len(data['transition_states'][f'p={2}']['losses'])+1))
        
        for p in ps: 
        
            for graph, i in zip(graphs, range(len(graphs))):
                fig = plt.figure()
                fig.suptitle(f'Losses for different initializations with optimized optimizers for p={p} for graphs with n = {n} and reg = {regularity}')

                with open(my_path + f'/data/nodes_{n}_reg_{regularity}_graph_{graph}_initialization_fixed_angles_optimization_opt_SGD_lr_{0.0001}.pkl', 'rb') as file:
                    data = pickle.load(file)
                #energy_single = data['analytic_single_p']['energy']
                #plt.scatter([1], energy_single, color=colors[0], label=f'Grid search with analytic expression')

                counter = 1
                plt.subplot(1,1, 1)
                for argument in arguments:
                    learning_rate = argument[0]
                    optimizer = argument[1]
                    initialization = argument[2]
                    counter_losses = 0
                    counter_losses_length = 0
                    losses = []
                    losses_short = []
                    losses = losses_list_single_input(n, regularity, p, i, learning_rate, optimizer, initialization)
                    for loss in losses:
                        
                        
                        
                        # if len(losses_short) == 0:
                        #     losses_min = loss
                        # losses_short.append(loss)
                        
                        # if abs(losses_min - loss) < 0.001:
                        #     counter_losses += 1
                        #     if counter_losses == 5:
                        #         break
                        # else:
                        #     counter_losses = 0

                        # if loss < losses_min:
                        #     losses_min = loss
                    


                        losses_short.append(loss)
                        if len(losses_short)>1:
                            if abs((losses_short[-1]-losses_short[-2])/losses_short[-1]) < 0.00025:
                                counter_losses += 1
                                if counter_losses == 5:
                                    break
                        else:
                            counter_losses = 0

                    x_short = list(range(1, len(losses_short)+1))
                    plt.plot(x, losses, color=colors[counter], label=f'{argument[2]} with {argument[1]} and lr={argument[0]} optimization')
                    counter += 1

                    if len(arguments)==1:
                        if initialization == 'transition_states':
                            plt.text(x[-1], losses[-1], losses[-1], ha='left', va='bottom')
                            plt.text(x_short[-1], losses_short[-1]*(1-0.02), losses_short[-1], ha='center', va='bottom')

                        else:
                            plt.text(x[-1], losses[-1], losses[-1], ha='left', va='top')
                            plt.text(x_short[-1], losses_short[-1]*(1+0.02), losses_short[-1], ha='center', va='top')

                    plt.plot(x_short, losses_short, color=colors[counter], label=f'{argument[2]} with {argument[1]} and lr={argument[0]} optimization short')
                    counter += 1

                #if i==0 or i==4 or i==8:
                plt.ylabel('Energy')
                #if i==6 or i==7 or i==8 or i==9:
                plt.xlabel('Steps')
                plt.title(f'Graph number {i+1}')
                plt.legend(loc='upper right', bbox_to_anchor=(1,1))
                plt.subplots_adjust(hspace=0.4)
                fig.savefig(my_path + f'/graphs/Losses_reg_{regularity}_n_{n}_graph_{graph}_p_{p}_different_optimized_initializations.png')
                plt.show()
                plt.close()

        
    
graphs=[i for i in range(10)]
ns=[50]#, 100, 150, 200]
regularity=3
optimizers_list = ['SGD', 'RMSprop', 'Adam']
ps = [3]

learning_rates_SGD = [0.0001, 0.0005, 0.001]
learning_rates_RMSprop = [0.001, 0.005, 0.01, 0.05]
learning_rates_Adam = [0.001, 0.005, 0.01, 0.05]
initializers_list = ['random_init', 'transition_states', 'fixed_angles_optimization']

"""
graphs=[0, 1]
ns=[4]
regularity=3
optimizers_list = ['SGD', 'RMSprop']
"""

graphs=[14, 19, 23, 35, 47, 54, 64]
optimizers_list = ['Adam']

if __name__ == '__main__':
    #plot_energy_optimizer_different_lr(ns, regularity, graphs, optimizers_list, initializers_list)
    #plot_losses_optimizer_different_lr(ns, regularity, graphs, optimizers_list, initializers_list)
    #plot_energy__different_opt_init_n_50([200], ps, regularity, graphs)
    plot_losses__different_opt_init_n_50([200], ps, regularity, graphs)
