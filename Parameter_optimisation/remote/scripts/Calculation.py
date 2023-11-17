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
import os
import itertools
import matplotlib.pyplot as plt
import matplotlib
import torch.multiprocessing as mp
from IPython.display import set_matplotlib_formats
__plot_height = 9.119
matplotlib.rcParams['figure.figsize'] = (1.718*__plot_height, __plot_height)
set_matplotlib_formats('svg')

def analytic(problem):
    dictionary_single = {}
    expectation_values_single = SingleLayerQAOAExpectationValues(problem)
    expectation_values_single.optimize()
    energy_single = expectation_values_single.energy
    dictionary_single['energy']=energy_single
    dictionary_single['correlations'] = expectation_values_single.expect_val_dict.copy()

    return dictionary_single
            

def random_init(problem, ps, opt, **kwargs):
    if opt=='SGD':
        optimizer=torch.optim.SGD
    elif opt=='RMSprop':
        optimizer=torch.optim.RMSprop
    elif opt=='Adam':
        optimizer=torch.optim.Adam
        
    dictionary_random_init={}
    for p in ps:
        dictionary_random_init_sub={}

        if p==1:
            expectation_values_qtensor_random = QtensorQAOAExpectationValuesMAXCUT(problem, p, pbar=True)
            expectation_values_qtensor_random.optimize(Opt=optimizer, **kwargs)
            energy_qtensor_random = float(expectation_values_qtensor_random.energy)
            dictionary_random_init_sub['energy'] = energy_qtensor_random
            dictionary_random_init_sub['correlations'] = expectation_values_qtensor_random.expect_val_dict.copy()
            dictionary_random_init_sub['losses'] = expectation_values_qtensor_random.losses.copy()

        else:
            for j in range(p):
                expectation_values_qtensor_random = QtensorQAOAExpectationValuesMAXCUT(problem, p, pbar=True)
                expectation_values_qtensor_random.optimize(Opt=optimizer, **kwargs)
                energy_qtensor_random = float(expectation_values_qtensor_random.energy)

                if j==0:
                    energy_min_random=energy_qtensor_random
                    correlations_min_random=expectation_values_qtensor_random.expect_val_dict.copy()
                    losses_min_random=expectation_values_qtensor_random.losses.copy()
                if energy_qtensor_random < energy_min_random:
                    energy_min_random=energy_qtensor_random
                    correlations_min_random=expectation_values_qtensor_random.expect_val_dict.copy()
                    losses_min_random=expectation_values_qtensor_random.losses.copy()

            dictionary_random_init_sub['energy'] = energy_min_random
            dictionary_random_init_sub['correlations'] = correlations_min_random.copy()
            dictionary_random_init_sub['losses'] = losses_min_random.copy()
        
        dictionary_random_init[f'p={p}'] = dictionary_random_init_sub.copy()

    
    return dictionary_random_init


def transition_states(problem, ps, opt, **kwargs):
    if opt=='SGD':
        optimizer=torch.optim.SGD
    elif opt=='RMSprop':
        optimizer=torch.optim.RMSprop
    elif opt=='Adam':
        optimizer=torch.optim.Adam
    dictionary_transition_states={}
    for p in ps: 
        dictionary_transition_states_sub={}

        if p==1:
            expectation_values_single_transition = SingleLayerQAOAExpectationValues(problem)
            expectation_values_single_transition.optimize()
            gamma = [expectation_values_single_transition.gamma]
            beta = [expectation_values_single_transition.beta]
            energy_single_transition = expectation_values_single_transition.energy
            dictionary_transition_states_sub['energy'] = energy_single_transition
            dictionary_transition_states_sub['correlations'] = expectation_values_single_transition.expect_val_dict.copy()

        else:
            for j in range(p):
                gamma_ts = gamma.copy()
                beta_ts = beta.copy()
                gamma_ts.insert(j, 0)
                beta_ts.insert(j, 0)
                expectation_values_qtensor_transition = QtensorQAOAExpectationValuesMAXCUT(problem, p, gamma=gamma_ts, beta=beta_ts, pbar=True)
                expectation_values_qtensor_transition.optimize(Opt=optimizer, **kwargs)
                energy_qtensor_transition = float(expectation_values_qtensor_transition.energy)

                if j==0:
                    energy_min = energy_qtensor_transition
                    gamma_min = [float(i) for i in expectation_values_qtensor_transition.gamma]
                    beta_min = [float(i) for i in expectation_values_qtensor_transition.beta]
                    correlations_min = expectation_values_qtensor_transition.expect_val_dict.copy()
                    losses_min = expectation_values_qtensor_transition.losses.copy()

                if energy_qtensor_transition < energy_min:
                    energy_min = energy_qtensor_transition
                    gamma_min = [float(i) for i in expectation_values_qtensor_transition.gamma]
                    beta_min = [float(i) for i in expectation_values_qtensor_transition.beta]
                    correlations_min = expectation_values_qtensor_transition.expect_val_dict.copy()
                    losses_min = expectation_values_qtensor_transition.losses.copy()

            dictionary_transition_states_sub['energy'] = energy_min
            dictionary_transition_states_sub['correlations'] = correlations_min.copy()
            dictionary_transition_states_sub['losses'] = losses_min.copy()
            gamma=gamma_min.copy()
            beta=beta_min.copy()

        dictionary_transition_states[f'p={p}'] = dictionary_transition_states_sub
    
    return dictionary_transition_states


def fixed_angles(problem, regularity, ps):
    with open('angles_regular_graphs.json', 'r') as file:
        data = json.load(file)
    dictionary_fixed_angles = {}
    for p in ps:
        dictionary_fixed_angles_sub = {}
        gamma, beta = data[f"{regularity}"][f"{p}"]["gamma"], data[f"{regularity}"][f"{p}"]["beta"]
        gamma, beta = [value/(-2*np.pi) for value in gamma], [value/(2*np.pi) for value in beta]
        expectation_values_qtensor_fixed = QtensorQAOAExpectationValuesMAXCUT(problem, p, gamma=gamma, beta=beta, pbar=True)
        expectation_values_qtensor_fixed.calc_expect_val()
        energy_qtensor_fixed = float(expectation_values_qtensor_fixed.loss)
        dictionary_fixed_angles_sub['energy'] = energy_qtensor_fixed
        dictionary_fixed_angles_sub['correlations'] = expectation_values_qtensor_fixed.expect_val_dict.copy()
        dictionary_fixed_angles[f'p={p}'] = dictionary_fixed_angles_sub.copy()
    
    return dictionary_fixed_angles


def fixed_angles_optimization(problem, regularity, ps, opt, **kwargs):
    if opt=='SGD':
        optimizer=torch.optim.SGD
    elif opt=='RMSprop':
        optimizer=torch.optim.RMSprop
    elif opt=='Adam':
        optimizer=torch.optim.Adam

    with open('angles_regular_graphs.json', 'r') as file:
        data = json.load(file)
    dictionary_fixed_angles_optimization = {}
    for p in ps:
        dictionary_fixed_angles_optimization_sub = {}
        gamma, beta = data[f"{regularity}"][f"{p}"]["gamma"], data[f"{regularity}"][f"{p}"]["beta"]
        gamma, beta = [value/(-2*np.pi) for value in gamma], [value/(2*np.pi) for value in beta]
        expectation_values_qtensor_fixed_optim = QtensorQAOAExpectationValuesMAXCUT(problem, p, gamma=gamma, beta=beta, pbar=True)
        expectation_values_qtensor_fixed_optim.optimize(Opt=optimizer, **kwargs)
        energy_qtensor_fixed_optim = float(expectation_values_qtensor_fixed_optim.energy)
        dictionary_fixed_angles_optimization_sub['energy'] = energy_qtensor_fixed_optim
        dictionary_fixed_angles_optimization_sub['correlations'] = expectation_values_qtensor_fixed_optim.expect_val_dict.copy()
        dictionary_fixed_angles_optimization_sub['losses'] = expectation_values_qtensor_fixed_optim.losses.copy()
        dictionary_fixed_angles_optimization[f'p={p}'] = dictionary_fixed_angles_optimization_sub
    
    return dictionary_fixed_angles_optimization


def individual_MAXCUT_QAOA_optimization_single_initialization(G_num, n, regularity, max_p, initialization, opt, learning_rate):
    my_path = os.path.dirname(__file__)
    my_path = os.path.dirname(my_path)
    
    ps = list(range(1, max_p+1))
    with open(f'100_regular_graphs_nodes_{n}_reg_{regularity}.pkl', 'rb') as file:
        data = pickle.load(file)


    G = data[G_num]
    problem = Generator.MAXCUT(G)

    if initialization == 'analytic':
        dictionary_single = analytic(problem)
        pickle.dump(dictionary_single, open(my_path + f"/data/nodes_{n}_reg_{regularity}_graph_{G_num}_initialization_{initialization}.pkl", 'wb'))
        return dictionary_single.copy()
    
    elif initialization == 'random':
        dictionary_random = random_init(problem, ps, opt, lr=learning_rate)
        pickle.dump(dictionary_random, open(my_path + f"/data/nodes_{n}_reg_{regularity}_graph_{G_num}_initialization_{initialization}_opt_{opt}_lr_{learning_rate}.pkl", 'wb'))
        return dictionary_random.copy()
    
    elif initialization == 'transition_states':
        dictionary_transition_states = transition_states(problem, ps, opt, lr=learning_rate)
        pickle.dump(dictionary_transition_states, open(my_path + f"/data/nodes_{n}_reg_{regularity}_graph_{G_num}_initialization_{initialization}_opt_{opt}_lr_{learning_rate}.pkl", 'wb'))
        return dictionary_transition_states.copy()
    
    elif initialization =='fixed_angles':
        dictionary_fixed_angles = fixed_angles(problem, regularity, ps)
        pickle.dump(dictionary_fixed_angles, open(my_path + f"/data/nodes_{n}_reg_{regularity}_graph_{G_num}_initialization_{initialization}.pkl", 'wb'))
        return dictionary_fixed_angles.copy()
    
    elif initialization == 'fixed_angles_optimization':
        dictionary_fixed_angles_optimization = fixed_angles_optimization(problem, regularity, ps, opt, lr=learning_rate)
        pickle.dump(dictionary_fixed_angles_optimization, open(my_path + f"data/nodes_{n}_reg_{regularity}_graph_{G_num}_initialization_{initialization}_opt_{opt}_lr_{learning_rate}.pkl", 'wb'))
        return dictionary_fixed_angles_optimization   
    

def individual_MAXCUT_QAOA_optimization_all_initializations(G_num, n, regularity, max_p, opt, learning_rate):
    my_path = os.path.dirname(__file__)
    my_path = os.path.dirname(my_path)
    
    dictionary={}
    ps = list(range(1, max_p+1))
    with open(f'100_regular_graphs_nodes_{n}_reg_{regularity}.pkl', 'rb') as file:
        data = pickle.load(file)

    G = data[G_num]
    problem = Generator.MAXCUT(G)
    dictionary_single = analytic(problem)
    dictionary_random = random_init(problem, ps, opt, lr=learning_rate)
    dictionary_transition_states = transition_states(problem, ps, opt, lr=learning_rate)
    dictionary_fixed_angles = fixed_angles(problem, regularity, ps)
    dictionary_fixed_angles_optimization = fixed_angles_optimization(problem, regularity, ps, opt, lr=learning_rate)      
    dictionary['graph']=G_num
    dictionary['analytic_single_p']=dictionary_single
    dictionary['transition_states']=dictionary_transition_states
    dictionary['random_init']=dictionary_random
    dictionary['fixed_angles']=dictionary_fixed_angles
    dictionary['fixed_angles_optimization']=dictionary_fixed_angles_optimization

    #print(dictionary)
    pickle.dump(dictionary, open(my_path + f"/data/nodes_{n}_reg_{regularity}_graph_{G_num}_opt_{opt}_lr_{learning_rate}.pkl", 'wb'))
    print(f'Sucessfully saved nodes_{n}_reg_{regularity}_graph_{G_num}_opt_{opt}_lr_{learning_rate}.pkl"')


    




    

#individual_MAXCUT_QAOA_optimization(2, 4, 3, 2, 'random', 'RMSprop', 0.01)
