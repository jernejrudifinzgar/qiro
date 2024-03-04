import sys 
import os
import resource
#from memory_profiler import profile

sys.path.append("../../../Qtensor")
sys.path.append("../../../Qtensor/qtree_git")
sys.path.append("../../..")
sys.path.append("../../../classical_benchmarks")

#print(os.path.abspath(os.curdir))
#print(os.chdir("../../Qtensor"))
#print(os.path.abspath(os.curdir))
from greedy_mis import greedy_mis

from qtensor import ZZQtreeQAOAComposer, ZZQtreeQAOAComposer_MIS, ZZQtreeQAOAComposer_MAXCUT
from qtensor import QAOAQtreeSimulator, QAOAQtreeSimulator_MIS, QAOAQtreeSimulator_MAXCUT
from qtensor.contraction_backends import TorchBackend
import Generating_Problems as Generator
from Calculating_Expectation_Values import SingleLayerQAOAExpectationValues, QtensorQAOAExpectationValuesMIS,QtensorQAOAExpectationValuesMAXCUT, QtensorQAOAExpectationValuesQUBO
from QIRO import QIRO_MIS, QIRO_MIS_QMIN
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
import pickle
import matplotlib.pyplot as plt
from RQAOA import RQAOA, RQAOA_recalculate
import torch.multiprocessing as mp
from time import time

def execute_QIRO_single_instance_137_nodes(p, run, version, initialization, variation='standard', output_results=False, gamma=None, beta=None):
    my_path = os.path.dirname(__file__)
    my_path = os.path.dirname(my_path)
    reg = 3
    seed = 666

    G = nx.read_adjlist(my_path + f'/graphs/{run}-graph.adjlist', nodetype=int)
    problem = Generator.MIS(G)

    expectation_values_qtensor = QtensorQAOAExpectationValuesQUBO(problem, p, opt=torch.optim.RMSprop, initialization = initialization, opt_kwargs=dict(lr=0.005), gamma=gamma, beta=beta)
    QIRO_qtensor = QIRO_MIS(5, expectation_values_qtensor, variation=variation)

    time_start = time()
    QIRO_qtensor.execute()
    time_end = time()
    required_time = time_end-time_start

    solution_qtensor = QIRO_qtensor.solution
    size_indep_set_qiro_qtensor = np.sum(solution_qtensor >= 0)  
    solution_dict = {}
    solution_dict['size_solution_qtensor'] = size_indep_set_qiro_qtensor
    solution_dict['solution_qtensor'] = solution_qtensor
    solution_dict['energies_qtensor'] = QIRO_qtensor.energies_list
    solution_dict['losses_qtensor'] = QIRO_qtensor.losses_list
    solution_dict['num_nodes_qtensor'] = QIRO_qtensor.num_nodes
    
    
    if p==1:
        size_greedy = greedy_mis(G)
        solution_dict['size_solution_greedy'] = size_greedy

        problem = Generator.MIS(G)
        expectation_values_single = SingleLayerQAOAExpectationValues(problem)
        QIRO_single = QIRO_MIS(5, expectation_values_single)
        QIRO_single.execute()
        solution_single = QIRO_single.solution
        size_indep_set_qiro_single = np.sum(solution_single >= 0)
        solution_dict['size_solution_single'] = size_indep_set_qiro_single
        solution_dict['solution_single'] = solution_single
        solution_dict['energies_single'] = QIRO_single.energies_list
        solution_dict['num_nodes_single'] = QIRO_single.num_nodes

    # f = open(my_path + f"/data/results_test_run_{run}_n_{n}_p_{p}_version_{version}.txt", "w+")
    # f.write(f"\nRequired time in seconds for RQAOA: {required_time}")
    # f.write(f"\nRequired time in minutes for RQAOA: {required_time/60}")
    # f.write(f"\nRequired time in hours for RQAOA: {required_time/3600}")
    # f.write(f"\nCalculated number of cuts with tensor networks: {cuts_qtensor}")
    # f.write(f"\nCalculated solution with tensor networks: {solution_qtensor}")
    # if p==1:
    #     f.write(f"\nCalculated number of cuts with analytic method:: {cuts_single}")
    #     f.write(f"\nCalculated solution with analytic method: {solution_single}")
    # f.close()
    #print(solution_dict)
    pickle.dump(solution_dict, open(my_path + f"/data/results_run_{run}_n_{137}_p_{p}_initialization_{initialization}_variation_{variation}_version_{version}.pkl", 'wb'))

    if output_results:
        print('MIS size qtensor:', size_indep_set_qiro_qtensor)
        if p==1: 
            print('MIS size single:', size_indep_set_qiro_single)

    return size_indep_set_qiro_qtensor, solution_qtensor


def execute_QIRO_single_instance(n, p, run, version, initialization, variation='standard', output_results=False, gamma=None, beta=None):
    my_path = os.path.dirname(__file__)
    my_path = os.path.dirname(my_path)
    reg = 3
    seed = 666

    ns_graphs_rudi = list(range(60, 220, 20))

    ns_graphs_maxi = [30, 40, 50]

    if n in ns_graphs_rudi:
        with open(my_path + f'/graphs/rudis_100_regular_graphs_nodes_{n}_reg_{3}.pkl', 'rb') as file:
            data = pickle.load(file)
        G = data[run]
    elif n in ns_graphs_maxi:
        with open(my_path + f'/graphs/100_regular_graphs_nodes_{n}_reg_{3}.pkl', 'rb') as file:
            data = pickle.load(file)
        G = data[run]
    else: 
        #random.seed()
        G = nx.random_regular_graph(reg, n, seed=seed)

        #for Erdos Renyi graphs:
        #prob = reg/(n-1) 
        #G = nx.erdos_renyi_graph(n, prob)

    problem = Generator.MIS(G)

    expectation_values_qtensor = QtensorQAOAExpectationValuesQUBO(problem, p, opt=torch.optim.RMSprop, initialization = initialization, opt_kwargs=dict(lr=0.005), gamma=gamma, beta=beta)
    QIRO_qtensor = QIRO_MIS(5, expectation_values_qtensor, variation=variation)

    time_start = time()
    QIRO_qtensor.execute()
    time_end = time()
    required_time = time_end-time_start

    solution_qtensor = QIRO_qtensor.solution
    size_indep_set_qiro_qtensor = np.sum(solution_qtensor >= 0)  
    solution_dict = {}
    solution_dict['size_solution_qtensor'] = size_indep_set_qiro_qtensor
    solution_dict['solution_qtensor'] = solution_qtensor
    solution_dict['energies_qtensor'] = QIRO_qtensor.energies_list
    solution_dict['losses_qtensor'] = QIRO_qtensor.losses_list
    solution_dict['num_nodes_qtensor'] = QIRO_qtensor.num_nodes
    
    
    if p==1:
        size_greedy = greedy_mis(G)
        solution_dict['size_solution_greedy'] = size_greedy

        problem = Generator.MIS(G)
        expectation_values_single = SingleLayerQAOAExpectationValues(problem)
        QIRO_single = QIRO_MIS(5, expectation_values_single, variation=variation)
        QIRO_single.execute()
        solution_single = QIRO_single.solution
        size_indep_set_qiro_single = np.sum(solution_single >= 0)
        solution_dict['size_solution_single'] = size_indep_set_qiro_single
        solution_dict['solution_single'] = solution_single
        solution_dict['energies_single'] = QIRO_single.energies_list
        solution_dict['num_nodes_single'] = QIRO_single.num_nodes

    # f = open(my_path + f"/data/results_test_run_{run}_n_{n}_p_{p}_version_{version}.txt", "w+")
    # f.write(f"\nRequired time in seconds for RQAOA: {required_time}")
    # f.write(f"\nRequired time in minutes for RQAOA: {required_time/60}")
    # f.write(f"\nRequired time in hours for RQAOA: {required_time/3600}")
    # f.write(f"\nCalculated number of cuts with tensor networks: {cuts_qtensor}")
    # f.write(f"\nCalculated solution with tensor networks: {solution_qtensor}")
    # if p==1:
    #     f.write(f"\nCalculated number of cuts with analytic method:: {cuts_single}")
    #     f.write(f"\nCalculated solution with analytic method: {solution_single}")
    # f.close()
    #print(solution_dict)
    pickle.dump(solution_dict, open(my_path + f"/data/results_run_{run}_n_{n}_p_{p}_initialization_{initialization}_variation_{variation}_version_{version}.pkl", 'wb'))

    if output_results:
        print('MIS size qtensor:', size_indep_set_qiro_qtensor)
        if p==1: 
            print('MIS size single:', size_indep_set_qiro_single)

    return size_indep_set_qiro_qtensor, solution_qtensor


def execute_QIRO_single_instance_2(n, p, run, version, initialization, variation='standard', output_results=False, gamma=None, beta=None):
    my_path = os.path.dirname(__file__)
    my_path = os.path.dirname(my_path)
    reg = 3
    seed = 666

    ns_graphs_rudi = list(range(60, 220, 20))

    ns_graphs_maxi = [30, 40, 50]

    if n in ns_graphs_rudi:
        with open(my_path + f'/graphs/rudis_100_regular_graphs_nodes_{n}_reg_{3}.pkl', 'rb') as file:
            data = pickle.load(file)
        G = data[run]
    elif n in ns_graphs_maxi:
        with open(my_path + f'/graphs/100_regular_graphs_nodes_{n}_reg_{3}.pkl', 'rb') as file:
            data = pickle.load(file)
        G = data[run]
    else: 
        #random.seed()
        G = nx.random_regular_graph(reg, n, seed=seed)

        #for Erdos Renyi graphs:
        #prob = reg/(n-1) 
        #G = nx.erdos_renyi_graph(n, prob)

    problem = Generator.MIS(G)
    solution_dict = {}

    if p==1:
        size_greedy = greedy_mis(G)
        solution_dict['size_solution_greedy'] = size_greedy

        problem = Generator.MIS(G)
        expectation_values_single = SingleLayerQAOAExpectationValues(problem)
        QIRO_single = QIRO_MIS(5, expectation_values_single, variation=variation)
        QIRO_single.execute()
        solution_single = QIRO_single.solution
        size_indep_set_qiro_single = np.sum(solution_single >= 0)
        solution_dict['size_solution_single'] = size_indep_set_qiro_single
        solution_dict['solution_single'] = solution_single
        solution_dict['energies_single'] = QIRO_single.energies_list
        solution_dict['num_nodes_single'] = QIRO_single.num_nodes

    # f = open(my_path + f"/data/results_test_run_{run}_n_{n}_p_{p}_version_{version}.txt", "w+")
    # f.write(f"\nRequired time in seconds for RQAOA: {required_time}")
    # f.write(f"\nRequired time in minutes for RQAOA: {required_time/60}")
    # f.write(f"\nRequired time in hours for RQAOA: {required_time/3600}")
    # f.write(f"\nCalculated number of cuts with tensor networks: {cuts_qtensor}")
    # f.write(f"\nCalculated solution with tensor networks: {solution_qtensor}")
    # if p==1:
    #     f.write(f"\nCalculated number of cuts with analytic method:: {cuts_single}")
    #     f.write(f"\nCalculated solution with analytic method: {solution_single}")
    # f.close()
    #print(solution_dict)
    pickle.dump(solution_dict, open(my_path + f"/data/results_run_{run}_n_{n}_p_{p}_initialization_{initialization}_variation_{variation}_version_{version}.pkl", 'wb'))

#@profile
def execute_QIRO_multiple_instances(ns, ps, num_runs):
    arguments_list = []
    for n in ns:
        for p in ps: 
            for run in range(num_runs):
                arguments_list.append((n, p, run))

    num_processes = len(arguments_list)
    pool = mp.Pool(num_processes)
    results = pool.starmap(execute_QIRO_single_instance, arguments_list)

    return results


##########################################

#TODO adjust to QIRO
"""def execute_RQAOA_single_instance_recalculation(n, p, run, recalculation, version, output_results=False):
    my_path = os.path.dirname(__file__)
    my_path = os.path.dirname(my_path)
    reg = 3
    seed = 666

    ns_graphs_rudi = list(range(60, 220, 20))

    ns_graphs_maxi = [30]

    if n in ns_graphs_rudi:
        with open(f'rudis_100_regular_graphs_nodes_{n}_reg_{3}.pkl', 'rb') as file:
            data = pickle.load(file)
        G = data[run]
    elif n in ns_graphs_maxi:
        with open(f'100_regular_graphs_nodes_{n}_reg_3.pkl', 'rb') as file:
            data = pickle.load(file)
        G = data[run]
    else: 
        #random.seed()
        G = nx.random_regular_graph(reg, n)

    problem = Generator.MAXCUT(G)
    expectation_values_qtensor = QtensorQAOAExpectationValuesQUBO(problem, p, initialization='fixed_angles_optimization', opt=torch.optim.SGD, opt_kwargs=dict(lr=0.0001))
    RQAOA_qtensor = RQAOA_recalculate(expectation_values_qtensor, 5, recalculations=recalculation, type_of_problem="MAXCUT")
    time_start = time()
    cuts_qtensor, solution_qtensor = RQAOA_qtensor.execute()
    time_end = time()
    required_time = time_end-time_start
    solution_dict = {}
    solution_dict['cuts_qtensor']=cuts_qtensor
    solution_dict['solution_qtensor']= solution_qtensor
    
    if p==1:
        problem = Generator.MAXCUT(G)
        expectation_values_single = SingleLayerQAOAExpectationValues(problem)
        RQAOA_single = RQAOA(expectation_values_single, 5, type_of_problem="MAXCUT")
        cuts_single, solution_single = RQAOA_single.execute()
        solution_dict['cuts_single']=cuts_single
        solution_dict['solution_single']=solution_single
    f = open(my_path + f"/data/results_test_run_{run}_n_{n}_p_{p}_recalc_{recalculation}_version_{version}.txt", "w+")
    f.write(f"\nRequired time in seconds for RQAOA: {required_time}")
    f.write(f"\nRequired time in minutes for RQAOA: {required_time/60}")
    f.write(f"\nRequired time in hours for RQAOA: {required_time/3600}")
    f.write(f"\nCalculated number of cuts with tensor networks: {cuts_qtensor}")
    f.write(f"\nCalculated solution with tensor networks: {solution_qtensor}")
    if p==1:
        f.write(f"\nCalculated number of cuts with analytic method:: {cuts_single}")
        f.write(f"\nCalculated solution with analytic method: {solution_single}")
    f.close()

    if output_results:
        print('Cuts:', cuts_qtensor)

    pickle.dump(solution_dict, open(my_path + f"/data/results_run_{run}_n_{n}_p_{p}_recalc_{recalculation}_version_{version}.pkl", 'wb'))

    return cuts_qtensor, solution_qtensor

def execute_RQAOA_multiple_instances_recalculation(ns, ps, num_runs, recalculation):
    arguments_list = []
    for n in ns:
        for p in ps: 
            for run in range(num_runs):
                arguments_list.append((n, p, run, recalculation))

    num_processes = len(arguments_list)
    pool = mp.Pool(num_processes)
    results = pool.starmap(execute_RQAOA_single_instance_recalculation, arguments_list)

    return results


def execute_RQAOA_multiple_instances_different_n(ns, p, run, recalculation, version):
    for n in ns: 
        execute_RQAOA_single_instance_recalculation(n, p, run, recalculation, version)


def execute_RQAOA_parallel_recalculation(ns, ps, runs, recalculation, version):
    arguments_list = []
    for p in ps:
        for run in runs:
            arguments_list.append((ns, p, run, recalculation, version))
    
    pool = mp.Pool(len(arguments_list))
    pool.starmap(execute_RQAOA_multiple_instances_different_n, arguments_list)



def execute_RQAOA_multiple_instances_recalculation(ns, ps, num_runs, recalculation):
    arguments_list = []
    for n in ns:
        for p in ps: 
            for run in range(num_runs):
                arguments_list.append((n, p, run, recalculation))

    num_processes = len(arguments_list)
    pool = mp.Pool(num_processes)
    results = pool.starmap(execute_RQAOA_single_instance_recalculation, arguments_list)

    return results
"""


def give_hessian(n, p, run, version, initialization, output_results=False, gamma=None, beta=None):
    my_path = os.path.dirname(__file__)
    my_path = os.path.dirname(my_path)
    reg = 3
    seed = 666

    ns_graphs_rudi = list(range(60, 220, 20))

    ns_graphs_maxi = [30, 50]

    if n in ns_graphs_rudi:
        with open(f'rudis_100_regular_graphs_nodes_{n}_reg_{3}.pkl', 'rb') as file:
            data = pickle.load(file)
        G = data[run]
    elif n in ns_graphs_maxi:
        with open(f'100_regular_graphs_nodes_{n}_reg_{3}.pkl', 'rb') as file:
            data = pickle.load(file)
        G = data[run]
    else: 
        #random.seed()
        G = nx.random_regular_graph(reg, n)

        #for Erdos Renyi graphs:
        #prob = reg/(n-1) 
        #G = nx.erdos_renyi_graph(n, prob)

    problem = Generator.MIS(G)

    expectation_values_qtensor = QtensorQAOAExpectationValuesQUBO(problem, p, opt=torch.optim.RMSprop, initialization = initialization, opt_kwargs=dict(lr=0.005), gamma=gamma, beta=beta)
    return expectation_values_qtensor.optimize()
    
  








def execute_QIRO_multiple_instances_different_n(ns, p, run, version, initialization, variation):
    for n in ns: 
        
        execute_QIRO_single_instance(n, p, run, version, initialization, variation=variation)


def execute_QIRO_parallel(ns, ps, runs, version, initialization='random', variations=['standard']):
    arguments_list = []
    for p in ps:
        for variation in variations:
            for run in runs:
                arguments_list.append((ns, p, run, version, initialization, variation))
    
    pool = mp.Pool(len(arguments_list))
    pool.starmap(execute_QIRO_multiple_instances_different_n, arguments_list)
    





def execute_QIRO_multiple_instances_different_n_2(ns, p, run, version, initialization, variation):
    for n in ns: 
        
        execute_QIRO_single_instance(n, p, run, version, initialization, variation=variation)


def execute_QIRO_parallel_2(ns, ps, runs, version, initialization='random', variations=['standard']):
    arguments_list = []
    for p in ps:
        for variation in variations:
            for run in runs:
                arguments_list.append((ns, p, run, version, initialization, variation))
    
    pool = mp.Pool(len(arguments_list))
    pool.starmap(execute_QIRO_multiple_instances_different_n, arguments_list)