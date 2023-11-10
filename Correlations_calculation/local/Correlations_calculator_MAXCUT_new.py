import sys 
sys.path.append("../../Qtensor")
sys.path.append("../../Qtensor/qtree_git")
sys.path.append("../../")

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
__plot_height = 8.919
matplotlib.rcParams['figure.figsize'] = (1.718*__plot_height, __plot_height)
set_matplotlib_formats('svg')


version=2
regs = [3]#, 4]
ns = [100]#[50, 100, 150, 200]
seed = 666
#G = nx.random_geometric_graph(30, 0.5)
#G = nx.gnp_random_graph(n, 0.5, seed = seed)
num_runs=2

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
            dictionary_single = {}
            G = nx.random_regular_graph(reg, n)
            dictionary['graph'] = G
            problem = Generator.MAXCUT(G)
            expectation_values_single = SingleLayerQAOAExpectationValues(problem)
            expectation_values_single.optimize()
            energy_single = expectation_values_single.energy
            dictionary_single['energy']=energy_single
            correlations=expectation_values_single.expect_val_dict.copy()
            for key in list(correlations.keys()):
                if len(key)==1:
                    correlations.pop(key)
            dictionary_single['correlations']=correlations
            expectation_values_single.expect_val_dict
            x_single=[1]
            plt.subplot(3,4,i+1)
            plt.scatter(x_single, energy_single, color=colors[0], label=f'Grid search with analytic expression')
            #plt.set_title(f"Run number {i+1}")

            x = []
            energy_qtensor_random_init = []
            dictionary_p_random_init={}

            for p in ps:

                #random initialization
                expectation_values_qtensor = QtensorQAOAExpectationValuesMAXCUT(problem, p)
                expectation_values_qtensor.optimize()
                energy_qtensor = float(expectation_values_qtensor.energy)
                energy_qtensor_random_init.append(energy_qtensor)
                dictionary_sub = {}
                dictionary_sub['energy'] = energy_qtensor
                dictionary_sub['correlations'] = expectation_values_qtensor.expect_val_dict.copy()
                dictionary_p_random_init[f"p={p}"] = dictionary_sub
                x.append(p)
            dictionary['analytic_single_p'] = dictionary_single
            dictionary['random_init'] = dictionary_p_random_init
 
            list_runs.append(dictionary)


            
            plt.plot(x, energy_qtensor_random_init, colors[1], label=f'Qtensor optimization with random initialization')
            if i==0 or i==4 or i==8:
                plt.ylabel('Energy')
            if i==6 or i==7 or i==8 or i==9:
                plt.xlabel('p')
            plt.xticks(ps, x)
            plt.title(f'Run number {i+1}')
            #plt.set_title(f"Run number {i+1}")
       
        dictionary_n[n]=list_runs
        print(list_runs)
        pickle.dump(list_runs, open(f"correlations_MAXCUT_reg_{reg}_nodes_{n}_version_{version}.pkl", 'wb'))
        print('file saved successfully')


        plt.legend(loc='lower left', bbox_to_anchor=(1,0.5))
        fig.savefig(f'Energies_reg_{reg}_n_{n}_version_{version}.png')
    
    dictionary_reg[reg]=dictionary_n

pickle.dump(dictionary_reg, open(f"energies_all.pkl", 'wb'))
print('file saved successfully')