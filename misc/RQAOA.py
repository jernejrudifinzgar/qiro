import numpy as np
import itertools as it
import copy
import Calculating_Expectation_Values as Expectation_Values


class RQAOA(Expectation_Values.ExpectationValues):
    """
    :param problem_input: The problem object that shall be solved
    :param nc: size of remaining problem that is solved by brute force
    :param type_of_problem: Type of the problem (MAX-2-SAT and MIS are implemented so far)
    This class is responsible for the whole RQAOA procedure"""
    def __init__(self, problem_input, nc, type_of_problem="MAX_2_SAT"):
        """ create dictionaries, where single_dict contains all the coefficients of the single Z operators Z_i
        coupling_dict contains all coefficients of the terms Z_i*Z_j"""
        super().__init__(problem=problem_input)
        self.type_of_problem = type_of_problem
        self.nc = nc
        self.nq = len(self.problem.var_list) - self.nc
        self.coupling_del_list = []
        self.single_del_list = []
        self.solution = []

    def update_single(self, exp_value_coeff, exp_value_sign):
        """Updates Hamiltonian according to fixed single point correlation"""
        coupled_values = self.problem.matrix[exp_value_coeff[0], :exp_value_coeff[0]+1]
        coupled_values = np.append(coupled_values, self.problem.matrix[(exp_value_coeff[0]+1):, exp_value_coeff[0]])
        self.problem.matrix[np.diag_indices_from(self.problem.matrix)] += (coupled_values * exp_value_sign)
        self.problem.matrix = np.delete(self.problem.matrix, exp_value_coeff[0], 0)
        self.problem.matrix = np.delete(self.problem.matrix, exp_value_coeff[0], 1)

    def update_correlation(self, exp_value_coeff, exp_value_sign):
        """Updates Hamiltonian according to fixed two point correlation"""
        self.problem.matrix[exp_value_coeff[0], exp_value_coeff[0]] += self.problem.matrix[exp_value_coeff[1], exp_value_coeff[1]] * exp_value_sign
        coupled_values = self.problem.matrix[exp_value_coeff[1], :exp_value_coeff[1] + 1]
        coupled_values = np.append(coupled_values, self.problem.matrix[(exp_value_coeff[1] + 1):, exp_value_coeff[1]])
        for pos in range(1, len(coupled_values)):
            if exp_value_coeff[0] > pos:
                self.problem.matrix[exp_value_coeff[0], pos] += coupled_values[pos] * exp_value_sign
            elif exp_value_coeff[0] < pos:
                self.problem.matrix[pos, exp_value_coeff[0]] += coupled_values[pos] * exp_value_sign
        self.problem.matrix = np.delete(self.problem.matrix, exp_value_coeff[1], 0)
        self.problem.matrix = np.delete(self.problem.matrix, exp_value_coeff[1], 1)

    def calc_complete_solution(self, brute_forced_solution):
        """Starting from the small brute forced solution the full complete solution of the problem is reconstructed
        via the rounded correlations (in fixed_correl list)"""
        for i in reversed(self.fixed_correl):
            if i[1] == 0:
                i[1] = 1
            if len(i[0]) == 1:
                brute_forced_solution[i[0][0]] = int(i[1])
            else:
                brute_forced_solution[i[0][1]] = int(brute_forced_solution[i[0][0]] * i[1])

        for variable in self.problem.var_list:
            for assigned_variable in brute_forced_solution:
                if variable == assigned_variable:
                    self.solution.append(assigned_variable * brute_forced_solution[assigned_variable])

    def execute(self):
        """Solves the given problem via RQAOA and returns the solution and solution quality"""
        for step in range(self.nq):
            print(f"RQAOA Step: {step + 1}")

            #self.num_elimination_step.append(i)
            exp_value_coeff, exp_value_sign, max_exp_value = self.optimize()

            if len(exp_value_coeff) == 1:
                self.update_single([self.problem.position_translater.index(exp_value_coeff[0])], exp_value_sign)
                self.problem.remain_var_list.remove(exp_value_coeff[0])
                self.problem.position_translater.remove(exp_value_coeff[0])
            else:
                self.update_correlation([self.problem.position_translater.index(exp_value_coeff[0]), self.problem.position_translater.index(exp_value_coeff[1])], exp_value_sign)
                self.problem.remain_var_list.remove(exp_value_coeff[1])
                self.problem.position_translater.remove(exp_value_coeff[1])

        brute_forced_solution = self.brute_force()
        self.calc_complete_solution(brute_forced_solution)
        if self.type_of_problem == "MAX_2_SAT":
            E = self.problem.calc_violated_clauses(self.solution)

        elif self.type_of_problem == "MIS":
            # Calculate here size of independent set or whatever measure of quality you want to return
            E = self.problem.evaluate_solution(self.solution)


        return E, self.solution










