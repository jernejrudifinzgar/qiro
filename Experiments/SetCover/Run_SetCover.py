import sys 
sys.path.append("../../Qtensor")
sys.path.append("../../Qtensor/qtree_git")
sys.path.append("../../Qtensor/qtree_git/qtree")
sys.path.append("../../classical_benchmarks")
sys.path.append("..")
sys.path.append("../..")

from qtensor import ZZQtreeQAOAComposer, ZZQtreeQAOAComposer_MIS, ZZQtreeQAOAComposer_MAXCUT
from qtensor import QAOAQtreeSimulator, QAOAQtreeSimulator_MIS, QAOAQtreeSimulator_MAXCUT
from qtensor.contraction_backends import TorchBackend
import Generating_Problems as Generator
from Calculating_Expectation_Values import SingleLayerQAOAExpectationValues, QtensorQAOAExpectationValuesMIS,QtensorQAOAExpectationValuesMAXCUT,QtensorQAOAExpectationValuesQUBO
from QIRO import QIRO_MIS, MAXQ_MIS, MINQ_MIS, QIRO_SetCover
import RQAOA
import torch
import qtensor
import networkx as nx
import numpy as np
from scipy.optimize import minimize
import tqdm
from scipy.optimize import Bounds
import pprint
from functools import partial
import pickle
import random
import json
import matplotlib.pyplot as plt
from greedy_mis import min_greedy_mis, max_greedy_mis


U = [0, 1, 2, 3, 4, 5]
V = [[0, 1, 4], [1, 2, 4, 5, 6], [2, 3, 5], [0, 1], [1, 2], [2, 3], [4], [4, 5], [5]]
variation = 'MMQ'
problem = Generator.SetCover(U, V, A=2, B=1)
#print(problem.graph)
expectation_value_qtensor = QtensorQAOAExpectationValuesQUBO(problem, p=2, variation=variation, opt=torch.optim.RMSprop, initialization = 'interpolation', opt_kwargs=dict(lr=0.001))
QIRO_qtensor = QIRO_SetCover(1, expectation_value_qtensor, variation=variation)
QIRO_qtensor.execute()

print(QIRO_qtensor.energies_list)

for i in range(50):
    print(QIRO_qtensor.losses_list[0][i])