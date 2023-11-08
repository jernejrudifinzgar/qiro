import sys 
sys.path.append("./")
sys.path.append("./Qtensor")
sys.path.append("./Qtensor/qtree_git")
import numpy as np
import torch
import networkx as nx
import Generating_Problems as Generator
import networkx as nx
from Calculating_Expectation_Values import SingleLayerQAOAExpectationValues, QtensorQAOAExpectationValuesMIS
from QIRO import QIRO_MIS
from time import time
import matplotlib.pyplot as plt

if __name__ == '__main__':
    p = 1
    reg = 3
    ns = [50, 100, 150, 200]
    nc=3
    seed=666
    gamma = [0.1]
    beta = [0.3]
    number_of_runs=8

    colors=['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    fig, ax = plt.subplots()
    
    
    for n, color in zip(ns, colors):
        degree_list=[]
        ratio_list=[]


        for i in range(number_of_runs):

            G = nx.random_regular_graph(reg, n, seed = seed)
            
            regularity = reg/(n-1) 
            G = nx.erdos_renyi_graph(n, regularity)

            problem = Generator.MIS(G)
            expectation_values_qtensor = QtensorQAOAExpectationValuesMIS(problem, p)#, gamma=gamma, beta=beta)
            expectation_values_single = SingleLayerQAOAExpectationValues(problem)#, gamma=gamma, beta=beta)
            #qiro=QIRO_MIS(nc, expectation_values)

            #expectation_values_single.calc_expect_val()
            #expectation_values_qtensor.calc_expect_val()

            expectation_values_qtensor.optimize()
            expectation_values_single.optimize()

            diff_dict = {}
            ratio_dict = {}

            for i in expectation_values_qtensor.expect_val_dict.keys():
                diff = expectation_values_qtensor.expect_val_dict[i]-expectation_values_single.expect_val_dict[i] 
                diff_dict[i] = diff
                ratio = abs(diff/ expectation_values_qtensor.expect_val_dict[i])
                ratio_dict[i] = ratio

            #print(expectation_values_qtensor.expect_val_dict)
            print('Differences:')
            print(diff_dict)
            print('')
            print('Ratios:')
            print(dict(sorted(ratio_dict.items(), key=lambda item: item[1])))
            print('')
            print('Maximal ratio:')
            max_key = max(ratio_dict, key=ratio_dict.get)
            print(max_key, ratio_dict[max_key])

            
            for item in ratio_dict.items():
                for node in list(item[0]):
                    node=node-1
                    degree_list.append(G.degree(node))
                    ratio_list.append(item[1])
        
        ax.scatter(degree_list, ratio_list, c=color, s=5,label=f'n={n}')
        ax.set_title(f'Ratios of differences between expectation values')
    
    plt.legend()
    fig.savefig(f'n_{ns}.png')
    plt.show()
     






    #expectation_values_qtensor.optimize()
    #expectation_values_single.optimize()