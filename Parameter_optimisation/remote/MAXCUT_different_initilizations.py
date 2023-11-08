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
__plot_height = 8.519
matplotlib.rcParams['figure.figsize'] = (1.718*__plot_height, __plot_height)
set_matplotlib_formats('svg')


regs = [4]
ns = [50, 100, 150, 200]
seed = 666
#G = nx.random_geometric_graph(30, 0.5)
#G = nx.gnp_random_graph(n, 0.5, seed = seed)
num_runs=7

colors=['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
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
            problem = Generator.MAXCUT(G)
            expectation_values_single = SingleLayerQAOAExpectationValues(problem)
            expectation_values_single.optimize()
            energy_single = expectation_values_single.energy
            print(energy_single)
            dictionary['energy_single']=energy_single
            x_single=[1]
            plt.subplot(2,4,i+1)
            plt.scatter(x_single, energy_single, color=colors[0], label=f'Grid search with analytic expression')
            #plt.set_title(f"Run number {i+1}")

            x = []
            energy_qtensor_random_ini = []
            energy_qtensor_fixed_angle = []
            energy_qtensor_fixed_angle_optimization = []

            with open('angles_regular_graphs.json', 'r') as file:
                data = json.load(file)
            dictionary_p_random_init={}
            dictionary_p_fixed_angles={}
            dictionary_p_fixed_angles_optimization={}

            for p in ps:

                #random initialization
                expectation_values_qtensor = QtensorQAOAExpectationValuesMAXCUT(problem, p)
                expectation_values_qtensor.optimize()
                energy_qtensor = float(expectation_values_qtensor.energy)
                energy_qtensor_random_ini.append(energy_qtensor)
                dictionary_p_random_init[p]=energy_qtensor
                x.append(p)

                #fixed angles
                gamma, beta = data[f"{reg}"][f"{p}"]["gamma"], data[f"{reg}"][f"{p}"]["beta"] 
                gamma, beta = [value/(-2*np.pi) for value in gamma], [value/(2*np.pi) for value in beta]
                expectation_values_qtensor = QtensorQAOAExpectationValuesMAXCUT(problem, p, gamma=gamma, beta=beta)
                expectation_values_qtensor.calc_expect_val()
                energy_qtensor = float(expectation_values_qtensor.loss)
                energy_qtensor_fixed_angle.append(energy_qtensor)
                dictionary_p_fixed_angles[p]=energy_qtensor

                #fixed angles with optimization
                gamma, beta = data[f"{reg}"][f"{p}"]["gamma"], data[f"{reg}"][f"{p}"]["beta"] 
                gamma, beta = [value/(-2*np.pi) for value in gamma], [value/(2*np.pi) for value in beta]
                expectation_values_qtensor = QtensorQAOAExpectationValuesMAXCUT(problem, p, gamma=gamma, beta=beta)
                expectation_values_qtensor.optimize()
                energy_qtensor = float(expectation_values_qtensor.energy)
                energy_qtensor_fixed_angle_optimization.append(energy_qtensor)
                dictionary_p_fixed_angles_optimization[p]=energy_qtensor
            
            dictionary['energies_random_init']=dictionary_p_random_init
            dictionary['energies_fixed_angles']=dictionary_p_fixed_angles
            dictionary['energies_fixed_angles_optimization']=dictionary_p_fixed_angles_optimization
                 
            list_runs.append(dictionary)
            
            plt.plot(x, energy_qtensor_random_ini, colors[1], label=f'Qtensor optimization with random initialization')
            plt.plot(x, energy_qtensor_fixed_angle, colors[2], label=f'Qtensor fixed angles')
            plt.plot(x, energy_qtensor_fixed_angle_optimization, colors[3], label=f'Qtensor fixed angles with optimization')
            if i==0 or i==4:
                plt.ylabel('Energy')
            if i==3 or i==4 or i==5 or i==6 or i==7:
                plt.xlabel('p')
            plt.xticks(ps, x)
            plt.title(f'Run number {i+1}')
            #plt.set_title(f"Run number {i+1}")
       
        dictionary_n[n]=list_runs

        pickle.dump(list_runs, open(f"energies_reg_{reg}_nodes_{n}.pkl", 'wb'))
        print('file saved successfully')


        plt.legend(loc='lower left', bbox_to_anchor=(1,0.7))
        fig.savefig(f'Energies_reg_{reg}_n_{n}.png')
    
    dictionary_reg[reg]=dictionary_n

pickle.dump(dictionary_reg, open(f"energies_all.pkl", 'wb'))
print('file saved successfully')