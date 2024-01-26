import sys 
#print(sys.path)
sys.path.append("../../..")
sys.path.append("../../../Qtensor")
sys.path.append("../../../Qtensor/qtree_git")

import numpy as np
import networkx as nx
import Generating_Problems as Generator
from Calculating_Expectation_Values import SingleLayerQAOAExpectationValues, QtensorQAOAExpectationValuesMIS
from QIRO import QIRO_MIS
from classical_benchmarks.greedy_mis import greedy_mis, random_greedy_mis
from time import time
import multiprocessing as mp
from Helping_file import *
import json
import pickle
import random
import matplotlib.pyplot as plt

if __name__ == '__main__':

    reg = 3
    ns = [80]#, 140, 180]
    seed = 666
    ps= [1, 2, 3]
    #num_runs = 10
    runs=list(range(0, 20, 1))
    initialization = 'interpolation'
    version = 1
    #execute_RQAOA_single_instance(ns[0], ps[0], num_runs)
    #execute_RQAOA_multiple_instances(ns, ps, num_runs)
    #execute_QIRO_parallel(ns, ps, runs, version)

    #execute_QIRO_single_instance(30, 1, 0, 5)
    execute_QIRO_parallel(ns, ps, runs, version, initialization=initialization)
    
    
    
    execute_QIRO_parallel(ns, ps, runs, version, initialization='transition_states')
    # start_time = time()
    # give_hessian(30, 3, 0, 4, 'transition_states_try')
    # end_time = time()

    # time_try = end_time-start_time

    # start_time = time()
    # give_hessian(30, 3, 0, 5, 'transition_states_new')
    # end_time = time()

    # time_conventional = end_time-start_time

    #start_time = time()
    #give_hessian(30, 3, 0, 6, 'interpolation')
    #end_time = time()

    #time_interpolation = end_time-start_time

    #print('try time in hours:', time_try/3600)
    #print('conventional time in hours:', time_conventional/3600)
    #print('interpolation time:', time_interpolation)
 

    
    
