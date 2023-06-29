import numpy as np
import itertools as it
import copy
import Calculating_Expectation_Values as Expectation_Values

class QIRO(Expectation_Values.ExpectationValues):
    """
    :param problem_input: The problem object that shall be solved
    :param nc: size of remaining problem that is solved by brute force
    This class is responsible for the whole QIRO procedure; the output represents the optimized bitstring solution in the form
    of a dictionary as well as a list of optimal parameters from each elimination step
    """
    def __init__(self, problem_input, nc):
        """ create dictionaries, where single_dict contains all the coefficients of the single Z operators Z_i
        coupling_dict contains all coefficients of the terms Z_i*Z_j"""
        super().__init__(problem=problem_input)
        self.nc = nc
        self.assignment = []
        self.num_inference_appl = 1
        self.solution = []
        
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


def QIRO_BT_execute(nc, generated_problem, BT=True, QIRO=QIRO):

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

            exp_value_coeff, exp_value_sign, max_exp_value = BnB.optimize()
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
    return E_best_backtrack, E_first, first_solution, improved_backtrack_solution