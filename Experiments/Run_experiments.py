import sys 
print(sys.path)
sys.path.append("./")
sys.path.append("./Qtensor")
sys.path.append("./Qtensor/qtree_git")
import numpy as np
import networkx as nx
import Generating_Problems as Generator
from Calculating_Expectation_Values import SingleLayerQAOAExpectationValues, QtensorQAOAExpectationValuesMIS
from QIRO import QIRO_MIS
from classical_benchmarks.greedy_mis import greedy_mis, random_greedy_mis
from time import time
import multiprocessing as mp
from Calculation import calculate_single_solution
import json
import pickle
import random
import matplotlib.pyplot as plt

def give_results(G, nc, reg, n, ps, number_of_cases, parallel = False):
    if parallel==False:
        results_list = []
        for i in range(number_of_cases):
            result = calculate_single_solution(G, nc, reg, ns, ps, number_of_cases)
            results_list.append(result)

    elif parallel==True:
        
        pool = mp.Pool(mp.cpu_count())
        results_list = pool.starmap(calculate_single_solution, [(G, nc, reg, n, ps, i) for i in range(number_of_cases)])

    return results_list

if __name__ == '__main__':

    reg = [3]
    #seed = 666
    ps = [1, 2, 3]#, 4]
    nc = 3
    ns = [40]    #[20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
    number_of_cases = 8

    #results_dict_all = {}
    #results_dict_all["General info"]={'reg': reg, 'p': ps, 'n': ns, 'number of cases': number_of_cases}

    for regularity in reg: 
        for n in ns: 
            random.seed()
            output_dict={}

            #for regular graphs:
            G = nx.random_regular_graph(regularity, n)

            #for Erdos Renyi graphs:
            prob = regularity/(n-1) 
            G = nx.erdos_renyi_graph(n, prob)

            for p in ps:
                results_dict = {}
                results_dict["General info"]={'reg': reg, 'p': ps, 'n': ns, 'number of cases': number_of_cases}
                print(f"\nRight now calculating regularity={regularity}, n={n}")
                results=give_results(G, nc, regularity, n, ps, number_of_cases, parallel=True)
                results_dict[f"reg_{regularity}_n_{n}_numCases_{number_of_cases}"]= results
                #results_dict_all[f"reg_{regularity}_n_{n}_numCases_{number_of_cases}"]= results

                with open(f"results_run_1_reg_{regularity}_n_{n}_p_{p}_numCases_{number_of_cases}.pkl", 'wb') as f:
                    pickle.dump(results_dict, f)

                print(f'\nSuccessfully saved results_run_1_reg_{regularity}_n_{n}_p_{p}_numCases_{number_of_cases}.pkl')

    

    
    #graph1=results_dict["reg_3_n_6_numCases_2"][0]["graph"]
    #graph2=results_dict["reg_3_n_6_numCases_2"][1]["graph"]
    #print(graph1.edges())
    #print(graph2.edges())

    #print(results_dict_all)

    #with open(f"results_run_1_all.pkl", 'wb') as f:
    #    pickle.dump(results_dict_all, f)



    #plt.legend()
    #plt.show()
    
    



    
    