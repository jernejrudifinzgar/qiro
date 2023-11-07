import sys 
sys.path.append("./")
sys.path.append("./Qtensor")
sys.path.append("./Qtensor/qtree_git")
import numpy as np
import torch
import networkx as nx
import Generating_Problems as Generator
from Calculating_Expectation_Values import SingleLayerQAOAExpectationValues, QtensorQAOAExpectationValuesMIS
from QIRO import QIRO_MIS
from classical_benchmarks.greedy_mis import greedy_mis, random_greedy_mis
from time import time
import random

def calculate_single_solution(G, nc, reg, n, ps, pbar, output_steps, i):
    random.seed()
    output_dict={}

    #for regular graphs:
    #G = nx.random_regular_graph(reg, n)

    #for Erdos Renyi graphs:
    #prob = reg/(n-1) 
    #G = nx.erdos_renyi_graph(n, prob)

    output_dict["graph"] = G

    size_indep_set_min_greedy = greedy_mis(G)
    output_dict["size_indep_set_min_greedy"] = size_indep_set_min_greedy

    size_indep_set_random_greedy = random_greedy_mis(G)
    output_dict["size_indep_set_random_greedy"] = size_indep_set_random_greedy
    
    problem = Generator.MIS(G)

    expectation_values_single = SingleLayerQAOAExpectationValues(problem)
    qiro_single = QIRO_MIS(nc, expectation_values_single)
    #qiro_single.execute()
    qiro_single.execute()
    solution_single = qiro_single.solution
    size_indep_set_qiro_single = np.sum(solution_single >= 0)
    output_dict["output_single_p"] = [size_indep_set_qiro_single, solution_single]

    results_qtensor = {}

    for p in ps:
        print(f"\nRight now calculating p={p}")
        expectation_values_Qtensor = QtensorQAOAExpectationValuesMIS(problem, p, pbar=pbar)
        qiro_qtensor = QIRO_MIS(nc, expectation_values_Qtensor, output_steps=output_steps)
        start_time = time()
        #qiro_qtensor.execute()
        qiro_qtensor.execute()
        end_time = time()
        solution_qtensor = qiro_qtensor.solution
        size_indep_set_qiro_qtensor = np.sum(solution_qtensor >= 0)  
        results_qtensor[f'p={p}'] = [size_indep_set_qiro_qtensor, solution_qtensor]

    output_dict["output_qtensor"] = results_qtensor
    
    return output_dict