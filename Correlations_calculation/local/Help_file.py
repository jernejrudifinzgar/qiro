import sys 
sys.path.append("../../Qtensor")
sys.path.append("../../Qtensor/qtree_git")
sys.path.append("../../")

from qtensor import ZZQtreeQAOAComposer, ZZQtreeQAOAComposer_MIS, ZZQtreeQAOAComposer_MAXCUT
from qtensor import QAOAQtreeSimulator, QAOAQtreeSimulator_MIS, QAOAQtreeSimulator_MAXCUT
from qtensor.contraction_backends import TorchBackend
import Generating_Problems as Generator
from Calculating_Expectation_Values import SingleLayerQAOAExpectationValues, QtensorQAOAExpectationValuesMIS,QtensorQAOAExpectationValuesMAXCUT, QtensorQAOAExpectationValuesQUBO
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
import torch.multiprocessing as mp
from functools import partial
import random
import json
import matplotlib.pyplot as plt
import matplotlib
from IPython.display import set_matplotlib_formats
__plot_height = 8.919
matplotlib.rcParams['figure.figsize'] = (1.718*__plot_height, __plot_height)
set_matplotlib_formats('svg')


def calculate_correlations_single_instance(reg, n, ps, run):
    dictionary = {}
    dictionary_single = {}
    G = nx.random_regular_graph(reg, n)
    problem = Generator.MAXCUT(G, weighted=True)
    dictionary['graph'] = problem.graph.copy()

    expectation_values_single = SingleLayerQAOAExpectationValues(problem)
    expectation_values_single.optimize()
    energy_single = expectation_values_single.energy

    dictionary_single['energy'] = energy_single
    correlations=expectation_values_single.expect_val_dict.copy()
    for key in list(correlations.keys()):
        if len(key)==1:
            correlations.pop(key)
    dictionary_single['correlations']=correlations.copy()
    expectation_values_single.expect_val_dict
    energy_qtensor_random_init = []

    dictionary_p_random_init={}
    for p in ps:
        #random initialization
        expectation_values_qtensor = QtensorQAOAExpectationValuesQUBO(problem, p)
        expectation_values_qtensor.optimize()
        energy_qtensor = float(expectation_values_qtensor.energy)
        energy_qtensor_random_init.append(energy_qtensor)

        dictionary_sub = {}
        dictionary_sub['energy'] = energy_qtensor
        dictionary_sub['correlations'] = expectation_values_qtensor.expect_val_dict.copy()
        dictionary_p_random_init[f"p={p}"] = dictionary_sub
    dictionary['analytic_single_p'] = dictionary_single.copy()
    dictionary['random_init'] = dictionary_p_random_init.copy()

    return [dictionary, energy_single, energy_qtensor_random_init]

def calculate_correlations_multiple_runs(regs, ns, ps, num_runs, version, parallel = False):
    colors=['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
    dictionary_reg = {}
    for reg in regs:
        dictionary_n={}
        for n in ns:
        
            if parallel:
                pool = mp.Pool(num_runs)
                results_list = pool.starmap(calculate_correlations_single_instance, [(reg, n, ps, i) for i in range(num_runs)])

            else:
                results_list = []
                for run in range(num_runs):
                    result = calculate_correlations_single_instance(reg, n, ps, run).copy()
                    results_list.append(result)
                
            fig = plt.figure()
            fig.suptitle(f'Energies for depth p with different evaluation methods for weighted MAXCUT graphs with n = {n} and reg = {reg}')

            for run in range(num_runs):
                x_single=[1]
                plt.subplot(3,4,run+1)
                plt.scatter(x_single, results_list[run][1], color=colors[0], label=f'Grid search with analytic expression')
                results_list[run].pop(1)

                plt.plot(ps, results_list[run][1], colors[1], label=f'Qtensor optimization with random initialization')
                results_list[run].pop(1)

                if run==0 or run==4 or run==8:
                    plt.ylabel('Energy')
                if run==6 or run==7 or run==8 or run==9:
                    plt.xlabel('p')
                plt.xticks(ps, ps)
                plt.title(f'Run number {run+1}')
            plt.legend(loc='lower left', bbox_to_anchor=(1,0.5))
            fig.savefig(f'Energies_weighted_MAXCUT_graphs_reg_{reg}_n_{n}_version_{version}.png')

            dictionary_n[n]=results_list
            print(results_list)
            pickle.dump(results_list, open(f"correlations_weighted_MAXCUT_graphs_reg_{reg}_nodes_{n}_version_{version}.pkl", 'wb'))
            print('file saved successfully')

        dictionary_reg[reg]=dictionary_n