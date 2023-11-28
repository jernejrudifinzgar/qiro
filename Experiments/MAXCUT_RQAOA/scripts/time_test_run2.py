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
    ns = [60]#[60, 80, 100, 120, 140, 160, 180, 200]
    seed = 666
    ps= [2]#, 3]#, 2]#, 3]
    num_runs = 1
    execute_RQAOA_single_instance(ns[0], ps[0], num_runs)

    #execute_RQAOA_multiple_instances(ns, ps, num_runs)

    print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    print(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)