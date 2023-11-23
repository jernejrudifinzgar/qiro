import numpy as np
import copy
from Calculating_Expectation_Values import QtensorQAOAExpectationValuesMIS, SingleLayerQAOAExpectationValues
from Generating_Problems import MIS
import networkx as nx
from aws_quera import find_mis
import matplotlib.pyplot as plt
import itertools as it


class QIRO():
    """General QIRO base class"""
    def __init__(self, nc, expectation_values):
        self.problem = expectation_values.problem
        self.nc = nc
        self.expectation_values = expectation_values
        self.assignment = []
        self.solution = [] 

    def brute_force(self) -> np.ndarray:
        """Calculate brute force solution."""

        x_in_dict = {}
        brute_forced_solution = {}
        count = 0
        single_energy_vector = copy.deepcopy(self.problem.matrix.diagonal())
        correl_energy_matrix = copy.deepcopy(self.problem.matrix)
        np.fill_diagonal(correl_energy_matrix, 0)

        for iter_var_list in it.product([-1, 1], repeat=(len(self.problem.position_translater)-1)):
            vec = np.array([0])
            vec = np.append(vec, iter_var_list)
            E_current = self.problem.calc_energy(vec, single_energy_vector, correl_energy_matrix)

            for i in range(1, len(vec)):
                x_in_dict[self.problem.position_translater[i]] = iter_var_list[i-1]
            if count == 0:
                E_best = copy.deepcopy(E_current)
                brute_forced_solution = copy.deepcopy(x_in_dict)
                count += 1
            if float(E_current) < float(E_best):
                brute_forced_solution = copy.deepcopy(x_in_dict)
                E_best = copy.deepcopy(E_current)

        return brute_forced_solution
        

class QIRO_MIS(QIRO):
    """
    :param problem_input: The problem object that shall be solved
    :param nc: size of remaining subproblems that are solved by brute force
    This class is responsible for the whole QIRO procedure; the output represents the optimized bitstring solution in the form
    of a dictionary as well as a list of optimal parameters from each elimination step
    """
    def __init__(self, nc_input, expectation_values_input, output_steps=True):
        super().__init__( nc = nc_input, expectation_values=expectation_values_input)
        # let us use the problem graph as the reference, and this current graph as the dynamic
        # object from which we will eliminate nodes:
        self.output_steps=output_steps
        self.graph = copy.deepcopy(self.problem.graph)
        
    def update_single(self, variable_index, max_expect_val_sign):
        """Updates Hamiltonian according to fixed single point correlation"""
        node = variable_index - 1
        fixing_list = []
        assignments = []
        # if the node is included in the IS we remove its neighbors
        if max_expect_val_sign == 1:
            ns = copy.deepcopy(self.graph.neighbors(node))
            for n in ns:
                self.graph.remove_node(n)
                fixing_list.append([n + 1])
                assignments.append(-1)
        
        # in any case we remove the node which was selected by correlations:
        self.graph.remove_node(node)
        fixing_list.append([variable_index])
        assignments.append(max_expect_val_sign)

        # reinitailize the problem object with the new, updated, graph:
        self.problem = MIS(self.graph, self.problem.alpha)

        if self.expectation_values.type == 'QtensorQAOAExpectationValuesMIS':
            self.expectation_values = QtensorQAOAExpectationValuesMIS(self.problem, self.expectation_values.p, pbar=self.expectation_values.pbar)
        elif self.expectation_values.type == 'SingleLayerQAOAExpectationValue':
            self.expectation_values = SingleLayerQAOAExpectationValues(self.problem)

        return fixing_list, assignments
    
    def update_correlation(self, variables, max_expect_val_sign):
        """Updates Hamiltonian according to fixed two point correlation -- RQAOA (for now)."""
        
        #     """This does the whole getting-of-coupled-vars mumbo-jumbo."""
        fixing_list = []
        assignments = []
        if max_expect_val_sign == 1:
            # if variables are correlated, then we set both to -1 
            # (as the independence constraint prohibits them from being +1 simultaneously). 
            for variable in variables:
                fixing_list.append([variable])
                assignments.append(-1)
                self.graph.remove_node(variable - 1)                
        else:
            if self.output_steps:
                print("Entered into anticorrelated case:")
            # we remove the things we need to remove are the ones connected to both node, which are not both node.
            mutual_neighbors = set(self.graph.neighbors(variables[0] - 1)) & set(self.graph.neighbors(variables[1] - 1))
            fixing_list = [[n + 1] for n in mutual_neighbors]
            assignments = [-1] * len(fixing_list)
            for n in mutual_neighbors:
                self.graph.remove_node(n)

        # reinitailize the problem object with the new, updated, graph:
        self.problem = MIS(self.graph, self.problem.alpha)

        if self.expectation_values.type == 'QtensorQAOAExpectationValuesMIS':
            self.expectation_values = QtensorQAOAExpectationValuesMIS(self.problem, self.expectation_values.p, pbar=self.expectation_values.pbar)
        elif self.expectation_values.type == 'SingleLayerQAOAExpectationValue':
            self.expectation_values = SingleLayerQAOAExpectationValues(self.problem)

        return fixing_list, assignments

    def prune_graph(self):
        """Prunes the graph by removing all connected components that have less than nc nodes. The assignments are determined
        to be the maximum independent sets of the connected components. The self.graph is updated correspondingly."""

        # get connected components
        connected_components = copy.deepcopy(list(nx.connected_components(self.graph)))
        prune_assignments = {}
        for component in connected_components:
            if len(component) < self.nc:
                subgraph = self.graph.subgraph(component)
                _, miss = find_mis(subgraph)
                prune_assignments.update({n: 1 if n in miss[0] else -1 for n in subgraph.nodes}) 

        # remove component from graph
        for node in prune_assignments.keys():
            self.graph.remove_node(node)

        self.problem = MIS(self.graph, self.problem.alpha)

        if self.expectation_values.type == 'QtensorQAOAExpectationValuesMIS':
            self.expectation_values = QtensorQAOAExpectationValuesMIS(self.problem, self.expectation_values.p, pbar=self.expectation_values.pbar)
        elif self.expectation_values.type == 'SingleLayerQAOAExpectationValue':
            self.expectation_values = SingleLayerQAOAExpectationValues(self.problem)

        fixing_list = [[n + 1] for n in sorted(prune_assignments.keys())]
        assignments = [prune_assignments[n] for n in sorted(prune_assignments.keys())]

        return fixing_list, assignments
    
    def execute(self):
        """Main QIRO function which produces the solution by applying the QIRO procedure."""
        self.opt_gamma = []
        self.opt_beta = []
        self.fixed_correlations = []
        step_nr = 0
       
        while self.graph.number_of_nodes() > 0:
            step_nr += 1
            print(f"Step: {step_nr}. Number of nodes: {self.graph.number_of_nodes()}.")
            fixed_variables = []
            
            self.expectation_values.optimize()

            #plt.plot(self.expectation_values.losses, label = self.problem.graph.number_of_nodes())
            #plt.draw()
         
            # sorts correlations in decreasing order. Ties are broken randomly.
            sorted_correlation_dict = sorted(self.expectation_values.expect_val_dict.items(), key=lambda item: (abs(item[1]), np.random.rand()), reverse=True)
            # we iterate until we remove a node
            which_correlation = 0
            while len(fixed_variables) == 0:
                max_expect_val_location, max_expect_val = sorted_correlation_dict[which_correlation]
                
                max_expect_val_location = [self.problem.position_translater[idx] for idx in max_expect_val_location]
                max_expect_val_sign = np.sign(max_expect_val).astype(int)

                if len(max_expect_val_location) == 1:
                    if self.output_steps:
                        print(f"single var {max_expect_val_location}. Sign: {max_expect_val_sign}")
                    fixed_variables, assignments = self.update_single(*max_expect_val_location, max_expect_val_sign)
                    for var, assignment in zip(fixed_variables, assignments):

                        if var is None:
                            raise Exception("Variable to be eliminated is None. WTF?")
                        self.fixed_correlations.append([var, int(assignment), max_expect_val])
                else:
                    if self.output_steps:
                        print(f'Correlation {max_expect_val_location}. Sign: {max_expect_val_sign}.')
                    fixed_variables, assignments = self.update_correlation(max_expect_val_location, max_expect_val_sign)
                    for var, assignment in zip(fixed_variables, assignments):
                        if var is None:
                            raise Exception("Variable to be eliminated is None. WTF?")
                        
                        self.fixed_correlations.append([var, int(assignment), max_expect_val])

                # perform pruning.
                pruned_variables, pruned_assignments = self.prune_graph()
                if self.output_steps:
                    print(f"Pruned {len(pruned_variables)} variables.")
                
                for var, assignment in zip(pruned_variables, pruned_assignments):
                    if var is None:
                        raise Exception("Variable to be eliminated is None. WTF?")
                    self.fixed_correlations.append([var, assignment, None])
                
                fixed_variables += pruned_variables
                which_correlation += 1
                
                if self.output_steps:
                    if len(fixed_variables) == 0:
                        print("No variables could be fixed.")
                        print(f"Attempting with the {which_correlation}. largest correlation.")
                    else:
                        print(f"We have fixed the following variables: {fixed_variables}. Moving on.")
        
        solution = [var[0] * assig for var, assig, _ in self.fixed_correlations]
        sorted_solution = sorted(solution, key=lambda x: abs(x))
        print(f"Solution: {sorted_solution}")
        self.solution = np.array(sorted_solution).astype(int)


class QIRO_MAX_2SAT(QIRO):
    """
    :param problem_input: The problem object that shall be solved
    :param nc: size of remaining problem that is solved by brute force
    This class is responsible for the whole QIRO procedure; the output represents the optimized bitstring solution in the form
    of a dictionary as well as a list of optimal parameters from each elimination step
    """
    def __init__(self, nc_input, expectation_values_input):
        """ create dictionaries, where single_dict contains all the coefficients of the single Z operators Z_i
        coupling_dict contains all coefficients of the terms Z_i*Z_j"""
        super().__init__(nc=nc_input, expectation_values=expectation_values_input)
        self.num_inference_appl = 1
        
    def inference_rules(self):
        self.num_inference_appl = 1
        # BnB.search_variables()
        while self.num_inference_appl != 0:
            self.num_inference_appl = 0
            self.pure_literal()
            self.dominating_unit_clause()
            self.almost_common_clause()
            self.complementary_unit_clause()
            if len(self.problem.remain_var_list) <= self.nc:
                break
        self.problem.SAT_to_Hamiltonian()

    def update_formula(self, exp_value_coeff, exp_value_sign):
        if len(exp_value_coeff) == 1:
            self.problem.position_translater.remove(exp_value_coeff[0])
            self.problem.remain_var_list.remove(exp_value_coeff[0])
            clause_index = 0
            for clause in copy.deepcopy(self.problem.cnf):
                variable_index = 0
                for variable in clause:
                    if np.abs(variable) == exp_value_coeff[0]:
                        if np.sign(variable) == exp_value_sign:
                            self.problem.cnf.remove(clause)
                            clause_index -= 1
                        elif (np.sign(variable) == -exp_value_sign) and (len(clause) == 1):
                            self.problem.cnf.remove(clause)
                            clause_index -= 1
                        else:
                            self.problem.cnf[clause_index].remove(variable)
                clause_index += 1
        else:
            self.problem.position_translater.remove(exp_value_coeff[1])
            self.problem.remain_var_list.remove(exp_value_coeff[1])
            clause_index = 0
            for clause in copy.deepcopy(self.problem.cnf):
                if len(clause) == 1:
                    if exp_value_coeff[1] in clause:
                        self.problem.cnf[clause_index].remove(exp_value_coeff[1])
                        self.problem.cnf[clause_index].append(exp_value_sign*exp_value_coeff[0])
                    elif (-exp_value_coeff[1]) in clause:
                        self.problem.cnf[clause_index].remove((-exp_value_coeff[1]))
                        self.problem.cnf[clause_index].append((-exp_value_sign*exp_value_coeff[0]))

                elif ((np.abs(clause[0]) == exp_value_coeff[0]) and (np.abs(clause[1]) == exp_value_coeff[1])) or ((np.abs(clause[0]) == exp_value_coeff[1]) and (np.abs(clause[1]) == exp_value_coeff[0])):
                    if np.sign(clause[0])*np.sign(clause[1]) != exp_value_sign:
                        self.problem.cnf.remove(clause)
                        clause_index -= 1
                    else:
                        if exp_value_coeff[1] in copy.deepcopy(self.problem.cnf[clause_index]):
                            self.problem.cnf[clause_index].remove(exp_value_coeff[1])
                        else:
                            self.problem.cnf[clause_index].remove((-exp_value_coeff[1]))
                elif exp_value_coeff[1] in clause:
                    self.problem.cnf[clause_index].remove(exp_value_coeff[1])
                    self.problem.cnf[clause_index].append(exp_value_sign*exp_value_coeff[0])
                elif (-exp_value_coeff[1]) in clause:
                    self.problem.cnf[clause_index].remove((-exp_value_coeff[1]))
                    self.problem.cnf[clause_index].append((-exp_value_sign * exp_value_coeff[0]))
                clause_index += 1

    def search_variables(self, formula, variable):
        pos_unit_clause_index = []
        neg_unit_clause_index = []
        pos_duo_clause_index = []
        neg_duo_clause_index = []
        clause_index = 0
        for clause in formula:
            if variable in clause:
                if len(clause) == 1:
                    pos_unit_clause_index.append(clause_index)
                else:
                    pos_duo_clause_index.append(clause_index)
            if (-variable) in clause:
                if len(clause) == 1:
                    neg_unit_clause_index.append(clause_index)
                else:
                    neg_duo_clause_index.append(clause_index)
            clause_index += 1
        return pos_unit_clause_index, neg_unit_clause_index, pos_duo_clause_index, neg_duo_clause_index


    def dominating_unit_clause_help(self, variable, dominating_unit_clause_index, dominating_duo_clause_index, submissive_unit_clause_index, submissive_duo_clause_index):
        self.assignment.append(variable)
        self.problem.remain_var_list.remove(np.abs(variable))

        for clause_index in submissive_duo_clause_index:
            self.problem.cnf[clause_index].remove((-variable))

        del_duo_list = []
        for index_clause in dominating_duo_clause_index:
            del_duo_list.append(self.problem.cnf[index_clause])
        for del_clause in del_duo_list:
            del_clause_copy = copy.deepcopy(del_clause)
            del_clause_copy.remove(variable)
            removed_variable = del_clause_copy[0]
            self.problem.cnf.remove(del_clause)
            pos_unit_clause_index, neg_unit_clause_index, pos_duo_clause_index, neg_duo_clause_index = self.search_variables(self.problem.cnf, np.abs(removed_variable))
            if (len(pos_unit_clause_index) == 0) and (len(neg_unit_clause_index) == 0) and (len(pos_duo_clause_index) == 0) and (len(neg_duo_clause_index) == 0):
                self.problem.remain_var_list.remove(np.abs(removed_variable))
                self.assignment.append(removed_variable)

        for count in range(len(dominating_unit_clause_index)):
            self.problem.cnf.remove([variable])
        for count in range(len(submissive_unit_clause_index)):
            self.problem.cnf.remove([-variable])
        self.num_inference_appl += 1

    def dominating_unit_clause(self):
        for variable in self.problem.remain_var_list:
            pos_unit_clause_index, neg_unit_clause_index, pos_duo_clause_index, neg_duo_clause_index = self.search_variables(self.problem.cnf, variable)
            if len(pos_unit_clause_index) >= (len(neg_unit_clause_index) + len(neg_duo_clause_index)):
                self.dominating_unit_clause_help(variable, pos_unit_clause_index, pos_duo_clause_index, neg_unit_clause_index, neg_duo_clause_index)
            elif len(neg_unit_clause_index) >= (len(pos_unit_clause_index) + len(pos_duo_clause_index)):
                self.dominating_unit_clause_help(-variable, neg_unit_clause_index, neg_duo_clause_index, pos_unit_clause_index, pos_duo_clause_index)

    def almost_common_clause(self):
        for variable in self.problem.remain_var_list:
            pos_unit_clause_index, neg_unit_clause_index, pos_duo_clause_index, neg_duo_clause_index = self.search_variables(self.problem.cnf, variable)
            delete_clause_list = []
            added_clause = []
            for clause_pos in pos_duo_clause_index:
                index_negative_deleting_clauses = []
                for clause_neg in neg_duo_clause_index:
                    pos_copy_clause = copy.deepcopy(self.problem.cnf[clause_pos])
                    pos_copy_clause.remove(variable)
                    neg_copy_clause = copy.deepcopy(self.problem.cnf[clause_neg])
                    neg_copy_clause.remove(-variable)
                    if pos_copy_clause == neg_copy_clause and clause_neg not in index_negative_deleting_clauses:
                        index_negative_deleting_clauses.append(clause_neg)
                        delete_clause_list.append(self.problem.cnf[clause_pos])
                        delete_clause_list.append(self.problem.cnf[clause_neg])
                        added_clause.append(pos_copy_clause)
                        pos_duo_clause_index.remove(clause_pos)
                        neg_duo_clause_index.remove(clause_neg)
                        self.num_inference_appl += 1
                        break
            for removing_index in range(len(delete_clause_list)):
                self.problem.cnf.remove(delete_clause_list[removing_index])
            for removing_index in range(len(added_clause)):
                self.problem.cnf.append(added_clause[removing_index])
            pos_unit_clause_index, neg_unit_clause_index, pos_duo_clause_index, neg_duo_clause_index = self.search_variables(self.problem.cnf, variable)
            if (len(pos_unit_clause_index) == 0) and (len(neg_unit_clause_index) == 0) and (len(pos_duo_clause_index) == 0) and (len(neg_duo_clause_index) == 0):
                self.problem.remain_var_list.remove(variable)
                self.assignment.append(variable)

    def pure_literal(self):
        for variable in self.problem.remain_var_list:
            pos_unit_clause_index, neg_unit_clause_index, pos_duo_clause_index, neg_duo_clause_index = self.search_variables(self.problem.cnf, variable)
            if len(pos_unit_clause_index + pos_duo_clause_index) == 0 and len(neg_unit_clause_index + neg_duo_clause_index) > 0:
                del_duo_list = []
                for index_clause in neg_duo_clause_index:
                    del_duo_list.append(self.problem.cnf[index_clause])
                for del_clause in del_duo_list:
                    self.problem.cnf.remove(del_clause)
                for count in range(len(neg_unit_clause_index)):
                    self.problem.cnf.remove([-variable])
                self.assignment.append(-variable)
                self.problem.remain_var_list.remove(variable)
                self.num_inference_appl += 1
            elif len(neg_unit_clause_index + neg_duo_clause_index) == 0 and len(pos_unit_clause_index + pos_duo_clause_index) > 0:
                del_duo_list = []
                for index_clause in pos_duo_clause_index:
                    del_duo_list.append(self.problem.cnf[index_clause])
                for del_clause in del_duo_list:
                    self.problem.cnf.remove(del_clause)
                for count in range(len(pos_unit_clause_index)):
                    self.problem.cnf.remove([variable])
                self.assignment.append(variable)
                self.problem.remain_var_list.remove(variable)
                self.num_inference_appl += 1

    def help_for_complementary(self, pos_unit_clause_index, neg_unit_clause_index, variable):
        for i in range(np.min((len(pos_unit_clause_index), len(neg_unit_clause_index)))):
            self.problem.cnf.remove([variable])
            self.problem.cnf.remove([-variable])
            self.num_inference_appl += 1

    def complementary_unit_clause(self):
        for variable in self.problem.remain_var_list:
            pos_unit_clause_index, neg_unit_clause_index, pos_duo_clause_index, neg_duo_clause_index = self.search_variables(self.problem.cnf, variable)
            self.help_for_complementary(pos_unit_clause_index, neg_unit_clause_index, variable)
            pos_unit_clause_index, neg_unit_clause_index, pos_duo_clause_index, neg_duo_clause_index = self.search_variables(self.problem.cnf, variable)
            if len(pos_duo_clause_index) == 0 and len(neg_duo_clause_index) == 0 and len(pos_unit_clause_index) == len(neg_unit_clause_index):
                self.problem.remain_var_list.remove(variable)
                self.assignment.append(variable)

    def calc_solution(self, x_start_sol):
        for key in x_start_sol:
            self.assignment.append(key*x_start_sol[key])
        for correl in reversed(self.fixed_correl):
            if correl[1] == 0:
                correl[1] = 1
            if len(correl[0]) == 1:
                self.assignment.append(correl[0][0]*correl[1])
            else:
                test_count = 0
                for variable in self.assignment:
                    if int(np.abs(variable)) == int(correl[0][0]):
                        sign = np.sign(variable)
                        test_count += 1
                # This can happen if the assignment of a variable doesn't matter since it only appears in clauses that
                # are fulfilled anyway
                if test_count == 0:
                    sign = 1
                self.assignment.append(correl[0][1] * sign * correl[1])
        for variable in self.problem.var_list:
            for assigned_variable in self.assignment:
                if variable == np.abs(assigned_variable):
                    self.solution.append(assigned_variable)
                if (variable not in self.assignment) and ((-variable) not in self.assignment):
                    self.solution.append(variable)
                    break

    #def execute():



"""def QIRO_BT_execute(nc, generated_problem, BT=True, QIRO=QIRO_MAX_2SAT):

        def calc_E(BnB):
            BnB.problem.SAT_to_Hamiltonian()
            brute_forced_sol = BnB.brute_force()
            BnB.calc_solution(brute_forced_sol)
            E = BnB.problem.calc_violated_clauses(BnB.solution)
            return E, BnB.solution

        def eliminations(BnB, first_run=False):
            backtrack_data_new = []

            count = 0
            while len(BnB.problem.remain_var_list) > BnB.nc:
                if not(first_run) or count != 0:
                    BnB.inference_rules()
                else:
                    BnB.problem.SAT_to_Hamiltonian()

                if len(BnB.problem.remain_var_list) <= BnB.nc:
                    return BnB, backtrack_data_new

                exp_value_coeff, exp_value_sign, max_exp_value = BnB.expectation_values.optimize()
                backtrack_data_new.append([max_exp_value, exp_value_coeff, exp_value_sign, copy.deepcopy(BnB)])
                BnB.update_formula(exp_value_coeff, exp_value_sign)
                count += 1
            return BnB, backtrack_data_new

        BnB = QIRO(problem_input=generated_problem, nc=nc)

        BnB_best, backtrack_data = eliminations(BnB=BnB, first_run=True)
        E_first, first_solution = calc_E(BnB_best)

        improved_backtrack_solution = copy.deepcopy(first_solution)
        E_best_backtrack = copy.deepcopy(E_first)

        if BT == True:
            for data in backtrack_data:
                if len(data) == 0:
                    break
                current_BnB = data[3]

                current_BnB.fixed_correl[-1][1] = current_BnB.fixed_correl[-1][1] * (-1)
                current_BnB.update_formula(data[1], -data[2])
                current_BnB.inference_rules()

                BnB_current, backtrack_data_to_add = eliminations(BnB=current_BnB)
                E, solution_try = calc_E(copy.deepcopy(BnB_current))

                if E < E_best_backtrack:
                    E_best_backtrack = E
                    improved_backtrack_solution = solution_try
        return E_best_backtrack, E_first, first_solution, improved_backtrack_solution"""