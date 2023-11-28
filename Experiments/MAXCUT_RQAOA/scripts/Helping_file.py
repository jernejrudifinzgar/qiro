import sys 
import os
import resource
#from memory_profiler import profile

sys.path.append("../../../Qtensor")
sys.path.append("../../../Qtensor/qtree_git")
sys.path.append("../../..")

#print(os.path.abspath(os.curdir))
#print(os.chdir("../../Qtensor"))
#print(os.path.abspath(os.curdir))

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
from scipy.optimize import Bounds
import pprint
from functools import partial
import random
import json
import matplotlib.pyplot as plt
from RQAOA import RQAOA
import torch.multiprocessing as mp
from time import time

def execute_RQAOA_single_instance(n, p, run):
    my_path = os.path.dirname(__file__)
    my_path = os.path.dirname(my_path)
    reg = 3
    seed = 666
    random.seed()
    G = nx.random_regular_graph(reg, n)
    problem = Generator.MAXCUT(G)
    expectation_values_qtensor = QtensorQAOAExpectationValuesQUBO(problem, p, initialization='fixed_angles_initialization', opt=torch.optim.SGD, opt_kwargs=dict(lr=0.0001))
    RQAOA_qtensor = RQAOA(expectation_values_qtensor, 3, type_of_problem="MAXCUT")
    time_start = time()
    cuts_qtensor, solution_qtensor = RQAOA_qtensor.execute()
    time_end = time()
    required_time = time_end-time_start
    
    if p==1:
        problem = Generator.MAXCUT(G)
        expectation_values_single = SingleLayerQAOAExpectationValues(problem)
        RQAOA_single = RQAOA(expectation_values_single, 3, type_of_problem="MAXCUT")
        cuts_single, solution_single = RQAOA_single.execute()

    f = open(my_path + f"/data/results_test_run_{run}_n_{n}_p_{p}.txt", "w+")
    f.write(f"\nRequired time in seconds for RQAOA: {required_time}")
    f.write(f"\nRequired time in minutes for RQAOA: {required_time/60}")
    f.write(f"\nRequired time in hours for RQAOA: {required_time/3600}")
    f.write(f"\nCalculated number of cuts with tensor networks: {cuts_qtensor}")
    f.write(f"\nCalculated solution with tensor networks: {solution_qtensor}")
    if p==1:
        f.write(f"\nCalculated number of cuts with analytic method:: {cuts_single}")
        f.write(f"\nCalculated solution with analytic method: {solution_single}")
    f.close()

    return cuts_qtensor, solution_qtensor

#@profile
def execute_RQAOA_multiple_instances(ns, ps, num_runs):
    arguments_list = []
    for n in ns:
        for p in ps: 
            for run in range(num_runs):
                arguments_list.append((n, p, run))

    num_processes = len(arguments_list)
    pool = mp.Pool(num_processes)
    results = pool.starmap(execute_RQAOA_single_instance, arguments_list)

    return results