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
import itertools
import matplotlib.pyplot as plt
import matplotlib
import torch.multiprocessing as mp
from IPython.display import set_matplotlib_formats
__plot_height = 9.119
matplotlib.rcParams['figure.figsize'] = (1.718*__plot_height, __plot_height)
set_matplotlib_formats('svg')

def transition_states(problem, ps, opt, learning_rate):
    if opt=='SGD':
        optimizer=torch.optim.SGD
    elif opt=='RMSprop':
        optimizer=torch.optim.RMSprop
    elif opt=='Adam':
        optimizer=torch.optim.RMSprop
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
            dictionary_transition_states_sub['correlations'] = expectation_values_single_transition.expect_val_dict

        else:
            for j in range(p):
                gamma_ts = gamma.copy()
                beta_ts = beta.copy()
                gamma_ts.insert(j, 0)
                beta_ts.insert(j, 0)
                expectation_values_qtensor_transition = QtensorQAOAExpectationValuesMAXCUT(problem, p, gamma=gamma_ts, beta=beta_ts)
                expectation_values_qtensor_transition.optimize(Opt=optimizer, lr=learning_rate)
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