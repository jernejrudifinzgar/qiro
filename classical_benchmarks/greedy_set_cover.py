import networkx as nx
import numpy as np
import copy
import random
import os


#Greedy solver for the set cover problem
#It iteratively chooses the subset V with the highest cardinality and fixes it in the solution. Its elements are removed from all other subsets
def greedy_set_cover(U, V):
    """Greedy solver for the set cover problem"""
    solution = []
    U_copy = copy.deepcopy(U)
    V_copy = copy.deepcopy(V)

    while len(U_copy)>0:
        len_max = 0
        idx_max = []
        for i, subset in enumerate(V_copy):
            if len(subset) > len_max:
                len_max = len(subset)
                idx_max = [i]
            elif len(subset) == len_max:
                idx_max.append(i)
        index = random.choice(idx_max)
        chosen_subset = copy.deepcopy(V_copy[index])
        for i in chosen_subset:
            U_copy.remove(i)
            for subset in V_copy:
                if i in subset:
                    subset.remove(i)
        solution.append(chosen_subset)
        del V_copy[index]
    
    return solution, len(solution)


        
