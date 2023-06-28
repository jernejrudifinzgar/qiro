import numpy as np
import itertools as it
import copy


class MAX2SAT:
    def __init__(self, num_var, num_clauses):
        self.num_var = num_var
        self.num_clauses = num_clauses
        self.num_per_clause = 2
        self.var_list = []
        self.cnf = None
        self.cnf_start = None
        self.matrix = None
        self.position_translater = None
        self.generate_formula()
        self.SAT_to_Hamiltonian()
        self.matrix_start = copy.deepcopy(self.matrix)

    def generate_formula(self):
        variables = [i for i in range(1, self.num_var + 1)]
        self.cnf = []

        for j in range(int(self.num_clauses)):
            var_in_clause = np.random.choice(variables, self.num_per_clause, replace=True)
            neg_pos_in_clause = np.random.choice([1, -1], self.num_per_clause, replace=True)
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

    def add_off_element(self, i, j, const):
        if np.abs(i) >= np.abs(j):
            self.matrix[np.abs(i), np.abs(j)] += np.sign(i) * np.sign(j) * const
        else:
            self.matrix[np.abs(j), np.abs(i)] += np.sign(i) * np.sign(j) * const

    def add_diag_element(self, i, const):
        self.matrix[np.abs(i), np.abs(i)] += -np.sign(i) * const

    def SAT_to_Hamiltonian(self):
        self.single_dict = {}
        self.coupling_dict = {}

        self.position_translater = {0}
        for clause in self.cnf:
            positive_clause = []
            for element in range(len(clause)):
                positive_clause.append(abs(clause[element]))
            self.position_translater = self.position_translater.union(set(positive_clause))
        self.position_translater = list(self.position_translater)
        self.position_translater.sort()

        self.matrix = np.zeros((len(self.position_translater), len(self.position_translater)))

        for clause in self.cnf:
            if len(clause) > 1:
                ind_0 = self.position_translater.index(np.abs(clause[0])) * np.sign(clause[0])
                ind_1 = self.position_translater.index(np.abs(clause[1])) * np.sign(clause[1])
                self.add_off_element(i=int(ind_0), j=int(ind_1), const=1 / 4)
                self.add_diag_element(i=int(ind_0), const=1 / 4)
                self.add_diag_element(i=int(ind_1), const=1 / 4)
            elif len(clause) == 1:
                ind_0 = self.position_translater.index(np.abs(clause[0])) * np.sign(clause[0])
                self.add_diag_element(i=int(ind_0), const=1 / 2)

    def calc_energy(self, assignment, single_energy_vector, correl_energy_matrix):
        E_calc = np.sign(assignment).T @ correl_energy_matrix @ np.sign(assignment) + single_energy_vector.T @ np.sign(assignment)
        return E_calc

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
        return E




    ##### RUDI HERE YOU CAN ADD THE MIS PROBLEM GENERATOR
    ##### RUDI HERE YOU CAN ADD THE MIS PROBLEM GENERATOR
    ##### RUDI HERE YOU CAN ADD THE MIS PROBLEM GENERATOR


