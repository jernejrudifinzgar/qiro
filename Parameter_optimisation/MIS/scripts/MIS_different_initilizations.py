import sys 
sys.path.append("../../../Qtensor")
sys.path.append("../../../Qtensor/qtree_git")
sys.path.append("../../../")

from qtensor import ZZQtreeQAOAComposer, ZZQtreeQAOAComposer_MIS, ZZQtreeQAOAComposer_MAXCUT
from qtensor import QAOAQtreeSimulator, QAOAQtreeSimulator_MIS, QAOAQtreeSimulator_MAXCUT
from qtensor.contraction_backends import TorchBackend
import Generating_Problems as Generator
from Calculating_Expectation_Values import SingleLayerQAOAExpectationValues, QtensorQAOAExpectationValuesMIS,QtensorQAOAExpectationValuesMAXCUT
from QIRO import QIRO_MIS
import torch
import qtensor
import networkx as nx
import numpy as np
from scipy.optimize import minimize
import tqdm
import pickle
from scipy.optimize import Bounds
import pprint
from functools import partial
import random
import json
import matplotlib.pyplot as plt
import matplotlib
from IPython.display import set_matplotlib_formats
__plot_height = 9.119
matplotlib.rcParams['figure.figsize'] = (1.718*__plot_height, __plot_height)
set_matplotlib_formats('svg')


version=1
regs = [3]#, 4]
ns = [30]#[50, 100, 150, 200]
seed = 666
#G = nx.random_geometric_graph(30, 0.5)
#G = nx.gnp_random_graph(n, 0.5, seed = seed)
num_runs=10

colors=['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
ps=[1, 2, 3]

dictionary_reg = {}

for reg in regs:
    dictionary_n={}
    for n in ns:
        list_runs=[]
        fig = plt.figure()
        fig.suptitle(f'Energies for depth p with different evaluation methods for graphs with n = {n} and reg = {reg}')
        for i in range(num_runs):
            dictionary = {}
            G = nx.random_regular_graph(reg, n)
            dictionary['graph']=G
            problem = Generator.MIS(G)

            dictionary_single = {}
            expectation_values_single = SingleLayerQAOAExpectationValues(problem)
            expectation_values_single.optimize()
            energy_single = expectation_values_single.energy
            dictionary_single['energy']=energy_single
            dictionary_single['correlations'] = expectation_values_single.expect_val_dict.copy()
            
            x_single=[1]
            plt.subplot(3,4,i+1)
            plt.scatter(x_single, energy_single, color=colors[0], label=f'Grid search with analytic expression')
            #plt.set_title(f"Run number {i+1}")

            x = []
            energy_qtensor_transition_states = []
            energy_qtensor_random_init = []
            energy_qtensor_fixed_angle = []
            energy_qtensor_fixed_angle_optimization = []

            dictionary_p_transition_states={}
            dictionary_p_random_init={}
            dictionary_p_fixed_angles={}
            dictionary_p_fixed_angles_optimization={}
                        
            #transition states initialization
            for p in ps: 
                dictionary_p_transition_states_sub={}

                if p==1:
                    expectation_values_single = SingleLayerQAOAExpectationValues(problem)
                    expectation_values_single.optimize()
                    gamma = [expectation_values_single.gamma]
                    beta = [expectation_values_single.beta]
                    energy_single = expectation_values_single.energy
                    energy_qtensor_transition_states.append(energy_single)
                    dictionary_p_transition_states_sub['energy']=energy_single
                    dictionary_p_transition_states_sub['correlations']= expectation_values_single.expect_val_dict

                else:
                    for j in range(p):
                        gamma_ts=gamma.copy()
                        beta_ts=beta.copy()
                        gamma_ts.insert(j, 0)
                        beta_ts.insert(j, 0)
                        expectation_values_qtensor = QtensorQAOAExpectationValuesMIS(problem, p, gamma=gamma_ts, beta=beta_ts)
                        expectation_values_qtensor.optimize()
                        energy_qtensor = float(expectation_values_qtensor.energy)
                        if j==0:
                            energy_min=energy_qtensor
                            gamma_min=[float(i) for i in expectation_values_qtensor.gamma]
                            beta_min=[float(i) for i in expectation_values_qtensor.beta]
                            correlations_min = expectation_values_qtensor.expect_val_dict.copy()
                            losses_min = expectation_values_qtensor.losses.copy()

                        if energy_qtensor < energy_min:
                            energy_min=energy_qtensor
                            gamma_min=[float(i) for i in expectation_values_qtensor.gamma]
                            beta_min=[float(i) for i in expectation_values_qtensor.beta]
                            correlations_min = expectation_values_qtensor.expect_val_dict.copy()
                            losses_min = expectation_values_qtensor.losses.copy()

                    energy_qtensor_transition_states.append(energy_min)
                    dictionary_p_transition_states_sub['energy'] = energy_min
                    dictionary_p_transition_states_sub['correlations'] = correlations_min.copy()
                    dictionary_p_transition_states_sub['losses'] = losses_min.copy()
                    gamma=gamma_min.copy()
                    beta=beta_min.copy()

                dictionary_p_transition_states[f'p={p}']=dictionary_p_transition_states_sub

            for p in ps:

                dictionary_p_random_init_sub={}
                dictionary_p_fixed_angles_sub={}
                dictionary_p_fixed_angles_optimization_sub={}

                #random initialization
                expectation_values_qtensor = QtensorQAOAExpectationValuesMIS(problem, p)
                expectation_values_qtensor.optimize()
                energy_qtensor = float(expectation_values_qtensor.energy)
                energy_qtensor_random_init.append(energy_qtensor)
                dictionary_p_random_init_sub['energy'] = energy_qtensor
                dictionary_p_random_init_sub['correlations'] = expectation_values_qtensor.expect_val_dict.copy()
                dictionary_p_random_init_sub['losses'] = expectation_values_qtensor.losses.copy()
                x.append(p)

                dictionary_p_random_init[f'p={p}'] = dictionary_p_random_init_sub
                dictionary_p_fixed_angles[f'p={p}'] = dictionary_p_fixed_angles_sub


            dictionary['analytic_single_p']=dictionary_single
            dictionary['transition_states']=dictionary_p_transition_states
            dictionary['random_init']=dictionary_p_random_init
               
            list_runs.append(dictionary)
            
            plt.plot(x, energy_qtensor_random_init, colors[1], label=f'Qtensor optimization with random initialization')
            plt.plot(x, energy_qtensor_transition_states, colors[4], label=f'Qtensor optimization with transition states initialization')
            if i==0 or i==4 or i==8:
                plt.ylabel('Energy')
            if i==6 or i==7 or i==8 or i==9:
                plt.xlabel('p')
            plt.xticks(ps, x)
            plt.title(f'Run number {i+1}')
            #plt.set_title(f"Run number {i+1}")
       
        dictionary_n[n]=list_runs

        print(list_runs)
        pickle.dump(list_runs, open(f"energies_MIS_reg_{reg}_nodes_{n}_version_{version}.pkl", 'wb'))
        print('file saved successfully')


        plt.legend(loc='lower left', bbox_to_anchor=(1,0.4))
        fig.savefig(f'Energies_MIS_reg_{reg}_n_{n}_version_{version}.png')
    
    dictionary_reg[reg]=dictionary_n

#pickle.dump(dictionary_reg, open(f"energies_all.pkl", 'wb'))
print('file saved successfully')








#code for merging different pickle files:
"""    with open(f'correlations_MAXCUT_reg_{reg}_nodes_{100}_version_{1}.pkl', 'rb') as file:
        data_list_1 = pickle.load(file)

    with open(f'correlations_MAXCUT_reg_{reg}_nodes_{100}_version_{2}.pkl', 'rb') as file:
        data_list_2 = pickle.load(file)

    data_list_1.pop(4)
    data_list_1.pop(4)

    for i in data_list_2:
        data_list_1.append(i)

    pickle.dump(data_list_1, open(f"correlations_MAXCUT_reg_{reg}_nodes_{n}_version_{3}.pkl", 'wb'))


    colors=['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
    ps=[1, 2, 3]

    fig = plt.figure()
    fig.suptitle(f'Energies for depth p with different evaluation methods for graphs with n = {n} and reg = {reg}')

    for i in range(10):
        plt.subplot(3,4,i+1)
        plt.scatter([1], data_list_1[i]['analytic_single_p']['energy'], color=colors[0], label=f'Grid search with analytic expression')

        energy=[]
        for p in ps:
            energy.append(data_list_1[i]['random_init'][f'p={p}']['energy'])
            
        plt.plot(ps, energy, colors[1], label=f'Qtensor optimization with random initialization')
        
        if i==0 or i==4 or i==8:
            plt.ylabel('Energy')
        if i==6 or i==7 or i==8 or i==9:
            plt.xlabel('p')
        plt.xticks(ps, ps)
        plt.title(f'Run number {i+1}')
       
       

    plt.legend(loc='lower left', bbox_to_anchor=(1,0.5))
    fig.savefig(f'Energies_reg_{reg}_n_{n}_version_{3}.png')"""