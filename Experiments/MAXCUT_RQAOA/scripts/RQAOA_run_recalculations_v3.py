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
from Helping_file import *

        
if __name__ == '__main__':
    reg = 3
    ns = [50]#, 140, 180]
    seed = 666
    ps= [1, 2, 3]
    #num_runs = 10
    runs=list(range(20, 25, 1))
    version = 3
    iterations = [0, 1, 2, 3, 4]
    recalculation = 10
    #execute_RQAOA_single_instance_recalculation(50, 1, 7, 0, 5, 1, output_results=True)
        
    #execute_RQAOA_parallel_recalculation(ns, ps, runs, iterations, recalculation, version)
    execute_RQAOA_parallel_recalculation_nonsynchronous_iterations(ns, ps, runs, iterations, recalculation, version)

    # list_wo=[]
    # list_w=[]
    # for run in runs:
    #     cuts_wo, solution_wo = execute_RQAOA_single_instance(ns[0], ps[1], run, version, output_results=True)
    #     cuts_w, solution_w = execute_RQAOA_single_instance_recalculation(ns[0], ps[1], run, 1, version, output_results=True)
    #     list_wo.append(cuts_wo)
    #     list_w.append(cuts_w)
    #     print(list_wo)
    #     print(list_w)
    #execute_RQAOA_multiple_instances(ns, ps, num_runs)
    #execute_RQAOA_parallel_recalculation(ns, ps, runs, 1, version)
    #execute_RQAOA_parallel_recalculation(ns, ps, runs, 6, version)
    #print(list_wo)
    #print(list_w)
    #execute_RQAOA_single_instance(ns[0], ps[0], num_runs)
    #execute_RQAOA_multiple_instances(ns, ps, num_runs)
    #execute_RQAOA_parallel_recalculation(ns, ps, runs, 6, version)


    # begin_time = time()
    # execute_RQAOA_parallel(ns, ps, runs, version)
    # end_time = time()
    # required_time = end_time-begin_time

    # f = open(f"time_three.txt", "w+")
    # f.write(f"\nRequired time in seconds: {required_time}")
    # f.write(f"\nRequired time in minutes: {required_time/60}")
    # f.write(f"\nRequired time in hours: {required_time/3600}")
    # f.close()

    # print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # print(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
