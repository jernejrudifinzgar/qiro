from problem_generation.Generating_Problems import Problem
import copy
from utils.Matrix import Matrix
import numpy as np
from typing import Union
import networkx as nx
from itertools import chain, product, combinations

class MAXCUT(Problem):
    """MAXCUT problem generator."""
    def __init__(self, graph: nx.Graph):
        super().__init__(self)
        self.graph = copy.deepcopy(graph)
        self.var_list = None
        self.position_translater=None

        # compute the matrix (i.e., Hamiltonian) from the graph. Also sets the varlist!
        self.graph_to_matrix()
        self.remain_var_list = copy.deepcopy(self.var_list)
    
    def graph_to_matrix(self):
        """Transform the graph into its corresponding Hamiltonian matrix."""
        
        # matrix is one dimension larger due to the 0-th row and column, which are set to 0 by convention.
        self.matrixClass = Matrix(self.graph.number_of_nodes()+1)
        self.matrix = self.matrixClass.matrix

        # Transform graph nodes in ordered list of variables. These run from 1 -> n (instead of 0 -> n-1)
        variable_set = set()
        for node_shifted in self.graph.nodes:
            node = node_shifted + 1
            variable_set.add(node)
        variables = list(variable_set)
        variables.sort()
        self.var_list = copy.deepcopy(variables)

        # Filling the matrix (here the type of optimization problem is encoded, MIS in this case)
        # we skip the zeroth index, which is set to 0 by convention

        for correlation in self.graph.edges:
            # the first correlation + 1 comes from the fact that graph nodes run from 0...n-1
            # the fact that we add another +1 to the index is because the variables list runs 1....n 
            # and the indices in the matrix run 0...n
            idx1, idx2 = variables.index(correlation[0] + 1) + 1, variables.index(correlation[1] + 1) + 1
            self.matrixClass.add_off_element(idx1, idx2, 1)

        # we define the appropriate position translater
        self.position_translater = [0] + variables

    def evaluate_cuts(self, solution):
        """Returns the number of cuts found."""
        number_of_cuts = 0
        for n1, n2 in self.graph.edges:
            if solution[n1] * solution[n2] < 0.:
                number_of_cuts += 1
        
        return number_of_cuts