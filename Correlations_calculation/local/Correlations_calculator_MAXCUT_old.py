import sys 
#sys.path
sys.path.append("../../Qtensor")
sys.path.append("../../Qtensor/qtree_git")
sys.path.append("../../")
from qtensor import ZZQtreeQAOAComposer
from qtensor import QAOAQtreeSimulator, QtreeSimulator
from qtensor.contraction_backends import TorchBackend
import torch
import qtensor
import networkx as nx
import numpy as np
from scipy.optimize import minimize
import tqdm
from scipy.optimize import Bounds
import pprint
from functools import lru_cache
from time import time
import torch.multiprocessing as mp
import pickle
import json


def define_graph(n, reg):
    graph = nx.random_regular_graph(reg, n)
    return graph

def zz_correlations(G, gamma, beta):
    correlations_dict = {}
    qaoa_sim = QAOAQtreeSimulator(ZZQtreeQAOAComposer)

    E=0

    for edge in list(G.edges()):
        correlation = np.real(qaoa_sim._get_edge_energy(G, gamma, beta, edge))
        E+=correlation
        correlations_dict[edge]=correlation
        print(correlation)

    Ed = G.number_of_edges()
    E = (Ed - E)/2
    print(E)

    return correlations_dict

def ZZ_correlations_parallel(G, gamma, beta):
    correlations_dict = {}
    backend = TorchBackend()
    sim = qtensor.QtreeSimulator(backend=backend)
    composer=qtensor.TorchQAOAComposer(G, gamma=gamma, beta=beta)
    #composer=ZZQtreeQAOAComposer(G, gamma=gamma, beta=beta)
    #sim = QtreeSimulator()
    pool = mp.Pool(mp.cpu_count())
    correlations_edges = pool.starmap(zz_correlation, [(sim, composer, edge) for edge in G.edges()])
    
    E=0
    for edge, correlation in zip(list(G.edges()), correlations_edges):
        correlations_dict[edge]=correlation
        E+=correlation

    Ed = G.number_of_edges()
    E = (Ed - E)/2
    print(E)
    return correlations_dict

def zz_correlation(simulator, composer, edge):
    composer.energy_expectation_lightcone(edge)
    correlation = np.real(simulator.simulate(composer.circuit))  
    return correlation  

def create_weighted_graph(zz_correlations_dict):
    graph = nx.Graph()

    for edge, w in zz_correlations_dict.items():
        graph.add_edge(edge[0], edge[1], weight = w)
    return graph

def calculate_correlations(graph, gamma, beta, parallel=False):
    G = graph
    #backend = TorchBackend()
    #sim = qtensor.QtreeSimulator(backend=backend)
    if parallel == False:
        edge_correlations = zz_correlations(G, gamma, beta)
    else:
        edge_correlations = ZZ_correlations_parallel(G, gamma, beta)
        print(edge_correlations)

    final_graph=create_weighted_graph(edge_correlations)
    return final_graph



    
if __name__ == '__main__':

    with open('angles_regular_graphs.json', 'r') as file:
        data = json.load(file)
    p=1
    reg = 3
    n = [20]
    number_of_graphs = 1
    #seed=666

    for nodes in n:
        """ for layer in range(p):
            gamma, beta = data[f"{reg}"][f"{layer+1}"]["gamma"], data[f"{reg}"][f"{layer+1}"]["beta"]   
            #gamma, beta = [0.307766814546916], [0.3926720292447629]  
            gamma, beta = [value/(-2*np.pi) for value in gamma], [value/(2*np.pi) for value in beta]
            gamma, beta = torch.tensor(gamma, requires_grad=False), torch.tensor(beta, requires_grad=False)
            print("")
            #print(gamma)
            #print(beta)
        

            for i in range(number_of_graphs):
                if i == 0:
                    start = time()
                print('Energy of p = {}:'.format(layer+1))
                final_graph = calculate_correlations(reg, n, gamma, beta, parallel = True, seed=seed)
                if i ==0: 
                    end = time()
                    print("required time for one graph: " + str(end-start))
            
        print("")
        
        
        gamma, beta = [0.4220840819023261], [0.608757260014991]
        gamma, beta = [value/np.pi for value in gamma], [value/np.pi for value in beta]
        gamma, beta = torch.tensor(gamma, requires_grad=False), torch.tensor(beta, requires_grad=False)
        #seed = 666
        p=len(gamma)"""
        graphs = []
        for i in range(number_of_graphs):
            graphs.append(define_graph(nodes, reg))
            

        for layer in range(p):
            #gamma, beta = data[f"{reg}"][f"{layer+1}"]["gamma"], data[f"{reg}"][f"{layer+1}"]["beta"]
            #gamma, beta = [value/(-2*np.pi) for value in gamma], [value/(2*np.pi) for value in beta]
            gamma, beta = [0.3063146225854495/np.pi], [1.1852830222237083/np.pi]
            gamma, beta = torch.tensor(gamma, requires_grad=False), torch.tensor(beta, requires_grad=False)
            #print(gamma)
            #print(beta)
            weighted_graphs = []
            description_dict={"p": layer+1, "reg": reg, "n": nodes, "number_of_graphs": number_of_graphs, "gamma": gamma, "beta": beta}
            weighted_graphs.append(description_dict)
            for i in range(number_of_graphs):

                print('Right now calculating graph number {} out of {}'.format(i+1, number_of_graphs))
                print('Energy of p = {}:'.format(layer+1))
                final_graph = calculate_correlations(graphs[i], gamma, beta, parallel = False)
                weighted_graphs.append(final_graph)

                for edge in final_graph.edges():
                    print("correlation of edge {}: {}".format(edge, final_graph.get_edge_data(edge[0], edge[1])['weight']))
                    
            #pickle.dump(weighted_graphs, open("correlations_p_{}_reg_{}_nodes{}.pkl".format(layer+1, reg, nodes), 'wb'))
            #print('file saved successfully')