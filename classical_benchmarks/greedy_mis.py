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

def solve_greedy(path, idx, nreps=1):
    data = []
    for i in range(45):
        print(i)
        f_in = os.listdir(path)[i * 100 + idx]
        graph = nx.read_adjlist(os.path.join(path, f_in), nodetype=int)
        labs = f_in[:-8].split("-")
        # tmp = [n, k, i]
        for _ in range(nreps):
            tmp = [int(labs[0]), int(labs[1]), int(labs[2])]
            mis = greedy_mis(graph)
            tmp.append(mis)
            data.append(tmp)

    return np.array(data)

def solve_random_greedy(path, idx, nreps=1):
    data = []
    for i in range(45):
        print(i)
        f_in = os.listdir(path)[i * 100 + idx]
        graph = nx.read_adjlist(os.path.join(path, f_in), nodetype=int)
        labs = f_in[:-8].split("-")
        # tmp = [n, k, i]
        for _ in range(nreps):
            tmp = [int(labs[0]), int(labs[1]), int(labs[2])]
            mis = random_greedy_mis(graph)
            tmp.append(mis)
            data.append(tmp)

    return np.array(data)