import timeit
import numpy as np
import itertools as it
import copy
from scipy.optimize import fsolve
import Generating_Problems as Generator

class ExpectationValues():
    """
    :param problem: input problem
    this class is responsible for the whole RQAOA procedure
    """
    def __init__(self, problem):
        self.problem = problem
        self.a = None
        self.b = None
        self.c = None
        self.energy = None
        self.best_energy = None
        self.gamma = 0
        self.beta = 0
        self.fixed_correl = []

    """the single_cos und coupling_cos functions are sub-functions which are called in the calculation of the
    expectation values"""
    def single_cos(self, i, gamma):
        """sub-function"""
        a = 1
        vec_i = self.problem.matrix[i, 1:i]
        vec_i = np.append(vec_i, self.problem.matrix[i+1:, i])
        vec_i = vec_i[vec_i != 0]
        vec = np.cos(2 * gamma * (vec_i))
        a = np.prod(vec)
        return a

    def coupling_cos_0(self, i, j, gamma):
        """sub-function, careful it's not symmetric, the first index tells us the row of the matrix"""
        a = 1
        index_small = np.min((i, j))
        index_large = np.max((i, j))

        vec_i = self.problem.matrix[i, 1:index_small]
        if index_small == i:
            vec_i = np.append(vec_i, self.problem.matrix[index_small + 1:index_large, i])
        else:
            vec_i = np.append(vec_i, self.problem.matrix[i, index_small + 1:index_large])
        vec_i = np.append(vec_i, self.problem.matrix[index_large + 1:, i])

        vec_zeros = vec_i
        vec_non_zeros = vec_zeros[vec_zeros != 0]
        vec = np.cos(2 * gamma * (vec_non_zeros))
        a = np.prod(vec)

        return a

    def coupling_cos_plus(self, i, j, gamma):
        """sub-function"""
        index_small = np.min((i, j))
        index_large = np.max((i, j))

        vec_i = self.problem.matrix[index_small, 1:index_small]
        vec_i = np.append(vec_i, self.problem.matrix[index_small + 1:index_large, index_small])
        vec_i = np.append(vec_i, self.problem.matrix[index_large + 1:, index_small])

        vec_j = self.problem.matrix[index_large, 1:index_small]
        vec_j = np.append(vec_j, self.problem.matrix[index_large, index_small + 1:index_large])
        vec_j = np.append(vec_j, self.problem.matrix[index_large + 1:, index_large])
        vec_zeros = vec_i + vec_j
        vec_non_zeros = vec_zeros[vec_zeros != 0]

        vec = np.cos(2 * gamma * (vec_non_zeros))
        a = np.prod(vec)

        return a

    def coupling_cos_minus(self, index_large, index_small, gamma):
        """sub-function"""
        vec_i = self.problem.matrix[index_small, 1:index_small]
        vec_i = np.append(vec_i, self.problem.matrix[index_small + 1:index_large, index_small])
        vec_i = np.append(vec_i, self.problem.matrix[index_large + 1:, index_small])

        vec_j = self.problem.matrix[index_large, 1:index_small]
        vec_j = np.append(vec_j, self.problem.matrix[index_large, index_small + 1:index_large])
        vec_j = np.append(vec_j, self.problem.matrix[index_large + 1:, index_large])
        vec_zeros = vec_i - vec_j
        vec_non_zeros = vec_zeros[vec_zeros != 0]

        vec = np.cos(2 * gamma * (vec_non_zeros))
        a = np.prod(vec)
        return a

    def calc_single_terms(self, gamma, index):
        """This is just a help function to compute lengthy terms of the two-point expectation values and stores them as
        constants"""
        a_part_term = np.sin(2 * gamma * self.problem.matrix[index, index]) * self.single_cos(index, gamma)
        return a_part_term

    def calc_coupling_terms(self, gamma, index_large, index_small):
        """This is just a help function to compute lengthy terms of the two-point expectation values and stores them as
        constants"""

        b_part_term = (1/2) * np.sin(2 * gamma * self.problem.matrix[index_large, index_small]) * \
                    (
                            np.cos(2 * gamma * self.problem.matrix[index_large, index_large]) * self.coupling_cos_0(index_large, index_small, gamma) + np.cos(2 * gamma * self.problem.matrix[index_small, index_small]) * self.coupling_cos_0(index_small, index_large, gamma)
                    )
        c_0 = 1/2
        c_1 = np.cos(2 * gamma * (self.problem.matrix[index_large, index_large] + self.problem.matrix[index_small, index_small])) * self.coupling_cos_plus(index_large, index_small, gamma)
        c_2 = np.cos(2 * gamma * (self.problem.matrix[index_large, index_large] - self.problem.matrix[index_small, index_small])) * self.coupling_cos_minus(index_large, index_small, gamma)
        c_part_term = c_0 * (c_1 - c_2)

        return b_part_term, c_part_term

    def calc_const(self, gamma):
        """Calculates the constant terms for the step in which optimal beta is being calculated"""
        a = 0
        b = 0
        c = 0

        for index in range(1, len(self.problem.matrix)):
            a_term = self.problem.matrix[index, index] * self.calc_single_terms(gamma, index)
            a = a + a_term

        timi = 0
        count = 0
        for index_large in range(1, len(self.problem.matrix)):
            for index_small in range(1, index_large):

                if self.problem.matrix[index_large, index_small] != 0:
                    start = timeit.default_timer()
                    b_part_term, c_part_term = self.calc_coupling_terms(gamma, index_large, index_small)
                    stop = timeit.default_timer()
                    timi += stop - start
                    b_term = self.problem.matrix[index_large, index_small] * b_part_term
                    c_term = self.problem.matrix[index_large, index_small] * c_part_term
                    b = b + b_term
                    c = c + c_term
                    count += 1
        self.a = a
        self.b = b
        self.c = c

    def calc_max_exp_value(self):
        """Calculate all one- and two-point correlation expectation values and return the one with highest absolute value"""

        Z = np.sin(2 * self.beta) * self.calc_single_terms(gamma=self.gamma, index=1)
        if np.abs(Z) > 0:
            rounding_list = [[[self.problem.position_translater[1]], np.sign(Z), np.abs(Z)]]
            max_exp_value = np.abs(Z)
        else:
            rounding_list = [[[self.problem.position_translater[1]], 1, 0]]
            max_exp_value = 0

        for index in range(2, len(self.problem.matrix)):
            Z = np.sin(2 * self.beta) * self.calc_single_terms(gamma=self.gamma, index=index)
            if np.abs(Z) > max_exp_value:
                rounding_list = [[[self.problem.position_translater[index]], np.sign(Z), np.abs(Z)]]
                max_exp_value = np.abs(Z)
            elif np.abs(Z) == max_exp_value:
                rounding_list.append([[self.problem.position_translater[index]], np.sign(Z), np.abs(Z)])

        for index_large in range(1, len(self.problem.matrix)):
            for index_small in range(1, index_large):
                if self.problem.matrix[index_large, index_small] != 0:
                    b_part_term, c_part_term = self.calc_coupling_terms(gamma=self.gamma, index_large=index_large, index_small=index_small)
                    ZZ = np.sin(4 * self.beta) * b_part_term - ((np.sin(2 * self.beta)) ** 2) * c_part_term
                    if np.abs(ZZ) > max_exp_value:
                        rounding_list = [[[self.problem.position_translater[index_large], self.problem.position_translater[index_small]], np.sign(ZZ), np.abs(ZZ)]]
                        max_exp_value = np.abs(ZZ)
                    elif np.abs(ZZ) == max_exp_value:
                        rounding_list.append([[self.problem.position_translater[index_large], self.problem.position_translater[index_small]], np.sign(ZZ), np.abs(ZZ)])

        random_index = np.random.randint(len(rounding_list))
        roundling_element = rounding_list[random_index]
        exp_value_coeff = roundling_element[0]
        exp_value_sign = roundling_element[1]
        max_exp_value = roundling_element[2]
        return exp_value_coeff, exp_value_sign, max_exp_value

    def calc_beta_energy(self, gamma):
        """Calculate the optimal value of beta regarding the energy, dependent on the input gamma"""
        self.calc_const(gamma)

        def f(x):
            """energy derivative"""
            return 2 * self.a * np.cos(2 * x) + 4 * self.b * np.cos(4 * x) - 4 * self.c * np.sin(2 * x) * np.cos(2 * x)

        beta = float(fsolve(f, 0.01, xtol=0.000001))
        energy = self.a * np.sin(2 * beta) + self.b * np.sin(4 * beta) - self.c * ((np.sin(2 * beta)) ** 2)

        # running solver for calculating the root of the derivative of the energy function
        for i in range(10):
            start_point = (np.pi / 10) * (i + 1)
            beta_try = float(fsolve(f, start_point, xtol=0.000001))
            energy_try = self.a * np.sin(2 * beta_try) + self.b * np.sin(4 * beta_try) - self.c * ((np.sin(2 * beta_try)) ** 2)
            if energy_try < energy:
                energy = energy_try
                beta = beta_try

        # this is basically a backup code paragraph which does a rough grid search for the beta parameters in case
        # our solver gets stuck in a local minima then this value of beta will be used
        beta_grid = 0
        energy_grid = self.a * np.sin(2 * beta_grid) + self.b * np.sin(4 * beta_grid) - self.c * ((np.sin(2 * beta_grid)) ** 2)
        for beta_try in np.linspace(0, np.pi, 50):
            energy_try = self.a * np.sin(2 * beta_try) + self.b * np.sin(4 * beta_try) - self.c * ((np.sin(2 * beta_try)) ** 2)
            if energy_try < energy_grid:
                energy_grid = energy_try
                beta_grid = beta_try

        if energy > energy_grid:
            energy = energy_grid
            beta = beta_grid

        return beta, energy

    def calc_best_gamma(self, lb=0, ub=np.pi, steps=30):
        """Calculates best angles in a 30 points grid between a lower and upper bound"""
        for gamma in np.linspace(lb, ub, steps):
            beta, energy = self.calc_beta_energy(gamma)

            if energy < self.energy:
                self.gamma = gamma
                self.beta = beta
                self.energy = energy

    def optimize(self):
        self.gamma = 0
        self.beta, self.energy = self.calc_beta_energy(self.gamma)

        # rough grid search
        steps = 30
        self.calc_best_gamma(lb=0, ub=np.pi, steps=steps)

        # refined grid search
        lb = self.gamma - (np.pi / (steps - 1))
        ub = self.gamma + (np.pi / (steps - 1))
        self.calc_best_gamma(lb=lb, ub=ub, steps=steps)

        exp_value_coeff, exp_value_sign, max_exp_value = self.calc_max_exp_value()
        self.fixed_correl.append([exp_value_coeff, exp_value_sign, max_exp_value])

        return exp_value_coeff, exp_value_sign, max_exp_value


    def brute_force(self):
        """calculate optimal solution of the remaining variables (according to the remaining
        optimization problem) brute force"""
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
