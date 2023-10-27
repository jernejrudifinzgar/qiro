import numpy as np
import torch
import sys 
#sys.path.append("./")
sys.path.append("./Qtensor")
sys.path.append("./Qtensor/qtree_git")
import networkx as nx
import Generating_Problems as Generator
import networkx as nx
from Calculating_Expectation_Values import SingleLayerQAOAExpectationValues, QtensorQAOAExpectationValuesMIS
from QIRO import QIRO_MIS
from time import time


if __name__ == '__main__':
    p = 1
    reg = 3
    n = 4
    seed=666
    G = nx.random_regular_graph(reg, n, seed = seed)
    nc=3
    #G = nx.erdos_renyi_graph(n, 0.05, seed = seed)

    problem = Generator.MIS(G)
    expectation_values = QtensorQAOAExpectationValuesMIS(problem, p)
    qiro=QIRO_MIS(nc, expectation_values)

    start_time = time()
    qiro.execute()
    end_time = time()

    required_time = end_time-start_time

    if required_time <= 60:
        print('Execution time in seconds: ' + str(required_time))
    elif required_time <= 3600:
        print('Execution time in minutes: ' + str(required_time/60))
    else:
        print('Execution time in hours: ' + str(required_time/3600))

    expectation_values = SingleLayerQAOAExpectationValues(problem)
    qiro=QIRO_MIS(nc, expectation_values)
    
    start_time = time()
    qiro.execute()
    end_time = time()

    required_time = end_time-start_time

    if required_time <= 60:
        print('Execution time in seconds: ' + str(required_time))
    elif required_time <= 3600:
        print('Execution time in minutes: ' + str(required_time/60))
    else:
        print('Execution time in hours: ' + str(required_time/3600))


    """problem = Generator.MIS(G)

    exp_values = SingleLayerQAOAExpectationValues(problem)

    exp_value_coeff, exp_value_sign, max_exp_value = exp_values.optimize()

    print(exp_values.gamma)
    print(exp_values.beta)

    print(exp_values.max_exp_dict)
    print(exp_value_coeff)
    print(exp_value_sign)
    print(max_exp_value) """


    """exp_values = QtensorQAOAExpectationValuesMIS(problem)
    exp_value_coeff, exp_value_sign, max_exp_value = exp_values.optimize_parameters()

    print(exp_values.gamma)
    print(exp_values.beta)

    print(exp_values.max_exp_dict)
    print(exp_value_coeff)
    print(exp_value_sign)
    print(max_exp_value) """


    """ import sys 
    sys.path.append("/Users/q619238/qiro/Qtensor")
    sys.path.append("/Users/q619238/qiro/Qtensor/qtree_git")
    print(sys.path)

    from Qtensor.qtensor import QAOASimulator

    import test

    test.test_function() """