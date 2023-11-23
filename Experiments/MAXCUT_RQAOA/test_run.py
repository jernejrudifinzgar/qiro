import sys 
sys.path.append("../../Qtensor")
sys.path.append("../../Qtensor/qtree_git")
sys.path.append("../..")

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

def execute_RQAOA_single_instance(n, p):
    reg = 3
    seed = 666
    G = nx.random_regular_graph(reg, n, seed=seed)
    problem = Generator.MAXCUT(G)
    expectation_values_qtensor = QtensorQAOAExpectationValuesQUBO(problem, p)
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

    f = open(f"results_test_run_n_{n}_p_{p}.txt", "w+")
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

def execute_RQAOA_multiple_instances(ns, ps):
    arguments_list = []
    for n in ns:
        for p in ps: 
            arguments_list.append((n, p))

    num_processes = len(arguments_list)
    pool = mp.Pool(num_processes)
    results = pool.starmap(execute_RQAOA_single_instance, arguments_list)

    return results
        
if __name__ == '__main__':
    reg = 3
    ns = [60, 80, 100, 120, 140, 160, 180, 200]
    seed = 666
    ps= [1, 2, 3]
    
    execute_RQAOA_multiple_instances(ns, ps)

    