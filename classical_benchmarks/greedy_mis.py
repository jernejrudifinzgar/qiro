import networkx as nx
import numpy as np
import copy
import os

def greedy_mis(graph, seed=None, return_sol=False):
    """Computes the MIS greedily."""
    if seed is None:
        seed = np.random.randint(100000)
    g = copy.deepcopy(graph)
    rng = np.random.default_rng(seed)

    indep_set = []

    while g.number_of_nodes() > 0:
        deg_list = sorted(g.degree(), key=lambda x: x[1])
        # get the nodes that have the minimum degree
        selectable_nodes = [x for x, y in deg_list if y == deg_list[0][1]]
        # select a random node of minimal degree
        selected_node = rng.choice(selectable_nodes)
        g.remove_node(selected_node)
        g.remove_nodes_from(graph.neighbors(selected_node))
        indep_set.append(selected_node)
        
    if return_sol:
        return indep_set
    else:
        return len(indep_set)
    
def random_greedy_mis(graph, seed=None, return_sol=False):
    if seed is None:
        seed = np.random.randint(100000)
    g = copy.deepcopy(graph)
    rng = np.random.default_rng(seed)

    indep_set = []

    while g.number_of_nodes() > 0:
        # deg_list = sorted(g.degree(), key=lambda x: x[1])
        # get the nodes that have the minimum degree
        # selectable_nodes = [x for x, y in deg_list if y == deg_list[0][1]]
        # select a random node of minimal degree
        selected_node = rng.choice(list(g.nodes))
        g.remove_node(selected_node)
        g.remove_nodes_from(graph.neighbors(selected_node))
        indep_set.append(selected_node)
        
    if return_sol:
        return indep_set
    else:
        return len(indep_set)