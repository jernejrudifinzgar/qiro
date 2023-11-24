from typing import Any
import numpy as np
import itertools as it
import copy
import random 
import time


class Matrix():
    def __init__(self, size):
        self.size = size
        self.matrix = np.zeros((size, size))
    
    def add_off_element(self, i, j, const):
        """Adds an off-diagonal element to the Hamiltonian matrix."""
        if np.abs(i) >= np.abs(j):
            self.matrix[np.abs(i), np.abs(j)] += np.sign(i) * np.sign(j) * const
        else:
            self.matrix[np.abs(j), np.abs(i)] += np.sign(i) * np.sign(j) * const

    def add_diag_element(self, i, const):
        """Adds a diagonal element to the Hamiltonian matrix."""
        self.matrix[np.abs(i), np.abs(i)] += np.sign(i) * const


class Problem():
    """General problem generator."""
    def __init__(self, seed=42):
        self.random_seed = seed

    def calc_energy(self, assignment, single_energy_vector, correl_energy_matrix):
        E_calc = np.sign(assignment).T @ correl_energy_matrix @ np.sign(assignment) + single_energy_vector.T @ np.sign(assignment)
        return E_calc
    

class MAX2SAT(Problem):
    """Max 2-SAT problem generator."""
    def __init__(self, num_var, num_clauses):
        self.num_var = num_var
        self.num_clauses = num_clauses
        self.num_per_clause = 2
        self.var_list = []
        self.cnf = None
        self.cnf_start = None
        self.generate_formula()
        self.SAT_to_Hamiltonian()
        self.matrix_start = copy.deepcopy(self.matrix)
        self.type = "M2S"

    def generate_formula(self):
        """This generates a random MAX-2-SAT formula."""
        variables = [i for i in range(1, self.num_var + 1)]
        self.cnf = []

        rg = np.random.default_rng(self.random_seed)

        for j in range(int(self.num_clauses)):
            var_in_clause = rg.choice(variables, self.num_per_clause, replace=True)
            neg_pos_in_clause = rg.choice([1, -1], self.num_per_clause, replace=True)
            for var in var_in_clause:
                if var not in self.var_list:
                    self.var_list.append(var)

            # check if the two variables in the clause are the same
            if var_in_clause[0] == var_in_clause[1]:
                # if variables have opposite polarity and the clause is therefore automatically fulfilled, it is not
                # added to the formula
                if neg_pos_in_clause[0] != neg_pos_in_clause[1]:
                    pass
                # the two variables have also the same polarity, therefore it is sufficient to add the variable only
                # once to the clause
                else:
                    clause = [var_in_clause[0] * neg_pos_in_clause[0]]
                    self.cnf.append(clause)
            # adding clause with two different variables ("standard case")
            else:
                clause = [var_in_clause[0] * neg_pos_in_clause[0], var_in_clause[1] * neg_pos_in_clause[1]]
                self.cnf.append(clause)

        # copy of cnf formula will be needed in the recursive algorithms
        self.cnf_start = copy.deepcopy(self.cnf)
        self.var_list.sort()
        self.remain_var_list = copy.deepcopy(self.var_list)

    def SAT_to_Hamiltonian(self):
        """Converts the MAX-2-SAT formula to a Hamiltonian matrix."""

        self.position_translater = {0}
        for clause in self.cnf:
            positive_clause = []
            for element in range(len(clause)):
                positive_clause.append(abs(clause[element]))
            self.position_translater = self.position_translater.union(set(positive_clause))
        self.position_translater = list(self.position_translater)
        self.position_translater.sort()

        self.matrixClass = Matrix(len(self.position_translater))
        self.matrix = self.matrixClass.matrix       
        
        for clause in self.cnf:
            if len(clause) > 1:
                ind_0 = self.position_translater.index(np.abs(clause[0])) * np.sign(clause[0])
                ind_1 = self.position_translater.index(np.abs(clause[1])) * np.sign(clause[1])
                self.matrixClass.add_off_element(i=int(ind_0), j=int(ind_1), const=1 / 4)
                self.matrixClass.add_diag_element(i=-int(ind_0), const=1 / 4)
                self.matrixClass.add_diag_element(i=-int(ind_1), const=1 / 4)
            elif len(clause) == 1:
                ind_0 = self.position_translater.index(np.abs(clause[0])) * np.sign(clause[0])
                self.matrixClass.add_diag_element(i=-int(ind_0), const=1 / 2)
    
    def calc_violated_clauses(self, solution):
        E = 0
        E_diff = 1
        for clause in self.cnf_start:
            if len(clause) != 1:
                for variable in solution:
                    if np.abs(clause[0]) == np.abs(variable):
                        if np.sign(clause[0]) == np.sign(variable):
                            E_diff = 0
                for variable in solution:
                    if np.abs(clause[1]) == np.abs(variable):
                        if np.sign(clause[1]) == np.sign(variable):
                            E_diff = 0
            else:
                for variable in solution:
                    if np.abs(clause[0]) == np.abs(variable):
                        if np.sign(clause[0]) == np.sign(variable):
                            E_diff = 0
            E = E + E_diff
            E_diff = 1
            E = E_diff + E
        return E


class MIS(Problem):
    """Maximum Independent Set problem generator."""
    def __init__(self, graph, alpha=1.1):
        """Init takes in a networkx graph object, and a penalty factor alpha."""
        Problem.__init__(self)
        self.graph = copy.deepcopy(graph)
        self.alpha = alpha
        self.var_list = None
        self.position_translater=None
        self.type = "MIS"

        # compute the matrix (i.e., Hamiltonian) from the graph. Also sets the varlist!
        self.graph_to_matrix()
        self.remain_var_list = copy.deepcopy(self.var_list)
    
    def graph_to_matrix(self):
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
        for variable in variables:
            idx = variables.index(variable) + 1
            self.matrixClass.add_diag_element(idx, -1/2)

        for correlation in self.graph.edges:
            # the first correlation + 1 comes from the fact that graph nodes run from 0...n-1
            # the fact that we add another +1 to the index is because the variables list runs 1....n 
            # and the indices in the matrix run 0...n
            idx1, idx2 = variables.index(correlation[0] + 1) + 1, variables.index(correlation[1] + 1) + 1
            self.matrixClass.add_off_element(idx1, idx2, self.alpha/4)
            self.matrixClass.add_diag_element(idx1, self.alpha/4)
            self.matrixClass.add_diag_element(idx2, self.alpha/4)

        # we define the appropriate position translater
        self.position_translater = [0] + variables

    def evaluate_solution(self, solution):
        """Returns the number of violations and the size of the set found."""
        number_of_violations = 0
        size_of_set = len(*np.where(np.array(solution) > 0.))
        for n1, n2 in self.graph.edges:
            if solution[n1] > 0. and solution[n2] > 0.:
                number_of_violations += 1
        
        return number_of_violations, size_of_set
    


class MAXCUT(Problem):
    """MAXCUT problem generator."""
    def __init__(self, graph, weighted=False):
        Problem.__init__(self)
        self.graph = copy.deepcopy(graph)
        self.original_graph = copy.deepcopy(graph)
        self.var_list = None
        self.position_translater=None
        self.weighted = weighted
        if self.weighted:
            self.type = "weighted_MAXCUT"
        else:
            self.type = "MAXCUT"

        # compute the matrix (i.e., Hamiltonian) from the graph. Also sets the varlist!
        self.graph_to_matrix()
        self.remain_var_list = copy.deepcopy(self.var_list)
    
    def graph_to_matrix(self):
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

        if self.weighted:
            weights = [-1, 1]
            random.seed(time.time())
            for (u,v) in self.graph.edges():
                self.graph.edges[u, v]['weight'] = random.choice(weights)
            self.original_graph = copy.deepcopy(self.graph)

        for correlation in self.graph.edges:
            # the first correlation + 1 comes from the fact that graph nodes run from 0...n-1
            # the fact that we add another +1 to the index is because the variables list runs 1....n 
            # and the indices in the matrix run 0...n
            idx1, idx2 = variables.index(correlation[0] + 1) + 1, variables.index(correlation[1] + 1) + 1
            if self.weighted:
                weight = self.graph.edges[correlation[0], correlation[1]]['weight']
                self.matrixClass.add_off_element(idx1, idx2, weight)
            else:
                self.matrixClass.add_off_element(idx1, idx2, 1)

        # we define the appropriate position translater
        self.position_translater = [0] + variables


    def evaluate_cuts(self, solution):
        """Returns the number of cuts found."""
        number_of_cuts = 0
        for n1, n2 in self.original_graph.edges:
            if solution[n1] * solution[n2] < 0.:
                number_of_cuts += 1
        
        return number_of_cuts

    


