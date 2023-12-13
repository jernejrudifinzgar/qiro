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
import itertools
import matplotlib.pyplot as plt
import matplotlib
import torch.multiprocessing as mp
from IPython.display import set_matplotlib_formats
__plot_height = 9.119
matplotlib.rcParams['figure.figsize'] = (1.718*__plot_height, __plot_height)
set_matplotlib_formats('svg')
from Calculation import *
    

def MIS_QAOA_optimization_individual_initializations(num_graphs, ns, regularity, max_p=2, parallel=True):
    if parallel==True:
        
        graphs_list = list(range(0,num_graphs))
        initializations_list = ['random', 'transition_states']#['random', 'fixed_angles_optimization']#['analytic', 'random', 'transition_states', 'fixed_angles', 'fixed_angles_optimization']
        optimizers_list = ['SGD', 'RMSprop', 'Adam']
        learning_rates_SGD = [0.0001, 0.0005, 0.001]
        learning_rates_RMSprop = [0.001, 0.005, 0.01, 0.05]
        learning_rates_Adam = [0.001, 0.005, 0.01, 0.05]
        arguments_list=[]
        for n in ns:
            for graph in graphs_list:
                for init in initializations_list:
                    if init=='analytic':
                        arguments_list.append((graph, n, regularity, max_p, init, '', ''))

                    elif init=='fixed_angles':
                        arguments_list.append((graph, n, regularity, max_p, init, '', ''))

                    else:
                        for optimizer in optimizers_list:
                            if optimizer=='SGD':
                                for lr in learning_rates_SGD:
                                    arguments_list.append((graph, n, regularity, max_p, init, optimizer, lr))
                            elif optimizer=='RMSprop':
                                for lr in learning_rates_RMSprop:
                                    arguments_list.append((graph, n, regularity, max_p, init, optimizer, lr))
                            elif optimizer=='Adam':
                                for lr in learning_rates_Adam:
                                    arguments_list.append((graph, n, regularity, max_p, init, optimizer, lr))
                
        num_processes=len(arguments_list)
        pool = mp.Pool(num_processes) 
        print(num_processes)
        #pool = mp.Pool(8)
        dictionary = {}
        dictionaries_list = pool.starmap(individual_MIS_QAOA_optimization_single_initialization, arguments_list)
        dictionary['results']=dictionaries_list
        dictionary['arguments']=arguments_list
        pickle.dump(dictionary, open(f"results_all.pkl", 'wb'))
        print('file saved successfully')


def MIS_QAOA_optimization_all_initializations(graphs, ns, regularity, max_p=3, parallel=True):
    optimizers_list = ['SGD', 'RMSprop', 'Adam']
    learning_rates_SGD = [0.0001, 0.0005, 0.001]
    learning_rates_RMSprop = [0.001, 0.005, 0.01, 0.05]
    learning_rates_Adam = [0.001, 0.005, 0.01, 0.05]
   
    if parallel==True:

        arguments_list=[]
        for n in ns:
            for graph in graphs:
                for optimizer in optimizers_list:
                    if optimizer=='SGD':
                        for lr in learning_rates_SGD:
                            arguments_list.append((graph, n, regularity, max_p, optimizer, lr))
                    elif optimizer=='RMSprop':
                        for lr in learning_rates_RMSprop:
                            arguments_list.append((graph, n, regularity, max_p, optimizer, lr))
                    elif optimizer=='Adam':
                        for lr in learning_rates_Adam:
                            arguments_list.append((graph, n, regularity, max_p, optimizer, lr))
                
        num_processes=len(arguments_list)
        pool = mp.Pool(num_processes) 
        print(num_processes)
        #pool = mp.Pool(8)
        dictionary = {}
        dictionaries_list = pool.starmap(individual_MIS_QAOA_optimization_all_initializations, arguments_list)
        dictionary['results']=dictionaries_list
        dictionary['arguments']=arguments_list
        #pickle.dump(dictionary, open(f"results_all.pkl", 'wb'))
        print('file saved successfully')

    if parallel==False: 

        for n in ns:
            for graph in graphs:
                for optimizer in optimizers_list:
                    if optimizer=='SGD':
                        for lr in learning_rates_SGD:  
                            individual_MIS_QAOA_optimization_all_initializations(graph, n, regularity, max_p, optimizer, lr)
                    elif optimizer=='RMSprop':
                        for lr in learning_rates_RMSprop:
                            individual_MIS_QAOA_optimization_all_initializations(graph, n, regularity, max_p, optimizer, lr)
                    elif optimizer=='Adam':
                        for lr in learning_rates_Adam:
                            individual_MIS_QAOA_optimization_all_initializations(graph, n, regularity, max_p, optimizer, lr)


if __name__ == '__main__':
    graphs=[0]#, 1, 2, 3, 4, 5]#, 6, 7, 8, 9]
    ns=[30]#, 100, 150, 200]
    regularity=3
    MIS_QAOA_optimization_all_initializations(graphs, ns, regularity, parallel=True)


