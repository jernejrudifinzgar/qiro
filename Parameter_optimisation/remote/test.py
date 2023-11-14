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
from Calculation import *




reg = 3
n = 200
seed = 666
G = nx.random_regular_graph(reg, n, seed = seed)
problem = Generator.MAXCUT(G)
ps=[1, 2, 3]

arguments_list=[]
for i in range(10):
    arguments_list.append((problem, ps, 'Adam', 0.0001))

print(arguments_list)

pool = mp.Pool(10) 
dictionaries_list = pool.starmap(transition_states, arguments_list)
