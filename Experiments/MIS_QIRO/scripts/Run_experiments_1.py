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
    ns = [60, 80, 100, 120]#, 140, 180]
    seed = 666
    ps= [1]
    #num_runs = 10
    runs=list(range(0, 20, 1))
    initialization = 'interpolation'
    variations=['standard', 'MINQ', 'MAXQ', 'MMQ']
    version = 2
    results_list=[]

    #execute_QIRO_parallel([12], [1, 2], [0, 1], version, initialization=initialization, variations=variations)



    execute_QIRO_parallel(ns, ps, runs, version, initialization=initialization, variations=variations)
    
    
    
    
    #execute_QIRO_parallel(ns, ps, runs, version, initialization='transition_states')
    
    
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
 

    
    
