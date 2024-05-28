import sys 
sys.path.append("../../../Qtensor")
sys.path.append("../../../Qtensor/qtree_git")
sys.path.append("../../../Qtensor/qtree_git/qtree")
sys.path.append("../../../classical_benchmarks")
sys.path.append("../..")
sys.path.append("../../..")
sys.path.append("..")


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
import json
import random
import json
import matplotlib.pyplot as plt
from greedy_mis import min_greedy_mis, max_greedy_mis
from greedy_set_cover import greedy_set_cover
from Create_instances import create_set_cover
import multiprocessing as mp
import os

def create_and_solve_multiple(i, num_problems = 1, x=None, y=None, p=1):
    my_path = os.path.dirname(__file__)
    my_path = os.path.dirname(my_path)
    random.seed(i*100)
    solution_list = []
    for j in range(num_problems):
        x = random.randint(3, 12)
        y_limit = round(80/x)
        y = random.randint(3, y_limit)
        qubit_requirement = False
        while qubit_requirement==False:
            U, V = create_set_cover(x=x, y=y)
            problem = Generator.SetCover(U, V, A=2, B=1)   
            num_qubits = problem.num_variables
            print(x, y)
            if num_qubits<400:
                qubit_requirement = True
        variation = 'MMQ'
        expectation_value_qtensor = QtensorQAOAExpectationValuesQUBO(problem, p=p, variation=variation, opt=torch.optim.RMSprop, initialization = 'interpolation', opt_kwargs=dict(lr=0.001))
        MMQ = QIRO_SetCover(1, expectation_value_qtensor, variation=variation)
        shrinking_solution, shrinking_size = MMQ.execute()     

        greedy_solution, greedy_size = greedy_set_cover(U, V)

        solution_list.append({'Set': U, 'Subsets': V, 'shrinking_solution': list(shrinking_solution), 'shrinking_size': int(shrinking_size), 'greedy_solution': list(greedy_solution), 'greedy_size': int(greedy_size)})
        
        with open(my_path + f"/data/results_node{i}_run_2.json", 'w') as f:
            json.dump(solution_list, f)

    return solution_list

def create_and_solve_single(U, V, p=1):
    solution_list = []
    problem = Generator.SetCover(U, V, A=2, B=1)  
    variation = 'MMQ'
    expectation_value_qtensor = QtensorQAOAExpectationValuesQUBO(problem, p=p, variation=variation, opt=torch.optim.RMSprop, initialization = 'interpolation', opt_kwargs=dict(lr=0.001))
    MMQ = QIRO_SetCover(1, expectation_value_qtensor, variation=variation)
    shrinking_solution, shrinking_size = MMQ.execute()     

    greedy_solution, greedy_size = greedy_set_cover(U, V)

    solution_list.append({'Set': U, 'Subsets': V, 'shrinking_solution': shrinking_solution, 'shrinking_size': shrinking_size, 'greedy_solution': greedy_solution, 'greedy_size': greedy_size})

    return solution_list

def create_and_solve_multiple_parallel(num_problems = 1):
    random.seed(100)
    arguments_list = []
    for i in range(num_problems):
        x = random.randint(3, 12)
        y_limit = round(80/x)
        y = random.randint(3, y_limit)
        qubit_requirement = False
        while qubit_requirement==False:
            U, V = create_set_cover(x=x, y=y)
            problem = Generator.SetCover(U, V, A=2, B=1)   
            num_qubits = problem.num_variables
            print(x, y)
            if num_qubits<400:
                qubit_requirement = True
        arguments_list.append(U, V)
    
    pool = mp.Pool(len(arguments_list))
    pool.starmap(create_and_solve_single, arguments_list)
    

def main():
    random.seed(100)
    num_nodes = 100
    num_problems_per_node = 100

    arguments_list = []
    for i in range(num_nodes):
        arguments_list.append((i, num_problems_per_node))
    pool = mp.Pool(len(arguments_list))
    pool.starmap(create_and_solve_multiple, arguments_list)

def main_2():
    U = [0, 1, 2, 3, 4, 5]
    V = [[0, 1, 4], [1, 2, 4, 5], [2, 3, 5], [4, 5], [4], [5]]
    variation = 'MMQ'
    problem = Generator.SetCover(U, V, A=2, B=1)
    print('number of qubits:', problem.num_variables)
    expectation_value_qtensor = QtensorQAOAExpectationValuesQUBO(problem, p=3, variation=variation, opt=torch.optim.RMSprop, initialization = 'interpolation', opt_kwargs=dict(lr=0.001))
    QIRO_qtensor = QIRO_SetCover(1, expectation_value_qtensor, variation=variation)
    shrinking_solution, shrinking_size = QIRO_qtensor.execute()

    valid, rest_size = problem.solution_check(list(shrinking_solution))
    #solution_qtensor = QIRO_qtensor.solution
    print(valid, rest_size)

    losses = QIRO_qtensor.losses_list
    energies = QIRO_qtensor.energies_list
    dictionary = {'losses': losses, 'energies': energies}
    my_path = os.path.dirname(__file__)
    my_path = os.path.dirname(my_path)
    with open(my_path + f"/data/example_graph.json", 'w') as f:
        json.dump(dictionary, f)

if __name__ == '__main__':
    # num_problems = 5
    # solution = create_and_solve_multiple(num_problems=num_problems)

    # for i in range(num_problems):
    #     print('\nProblem number', i, '\nShrinking size:', solution[i]['shrinking_size'], '\nGreedy size:', solution[i]['greedy_size'])
    QIRO_qtensor = main_2()





        