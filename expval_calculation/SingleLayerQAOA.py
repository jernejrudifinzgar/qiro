import numpy as np
from scipy.optimize import fsolve

from expval_calculation.ExpVal import ExpectationValues
from problem_generation.Generating_Problems import Problem


class SingleLayerQAOAExpectationValues(ExpectationValues):
    """
    :param problem: input problem
    This class computes the p=1 QAOA expectation values, as given
    by the formulae in Ozaeta et al.
    """

    # Initialization method
    def __init__(self, problem: Problem, gamma: float = 0.0, beta: float = 0.0):
        # Call the parent class constructor
        super().__init__(problem)

        # Initialize instance variables
        self.a = None
        self.b = None
        self.c = None

        # Type checking and value assignment for gamma
        if isinstance(gamma, (float, int)):
            self.gamma = float(gamma) * np.pi
        elif isinstance(gamma, (list, np.ndarray)) and len(gamma) == 1:
            self.gamma = float(gamma[0]) * np.pi
        else:
            raise ValueError(
                "Gamma should be a float, int, or a single-element list or np.array."
            )

        # Type checking and value assignment for beta
        if isinstance(beta, (float, int)):
            self.beta = float(beta) * np.pi
        elif isinstance(beta, (list, np.ndarray)) and len(beta) == 1:
            self.beta = float(beta[0]) * np.pi
        else:
            raise ValueError(
                "Beta should be a float, int, or a single-element list or np.array."
            )

        # Set the type of the expectation value
        self.type = "SingleLayerQAOAExpectationValue"

    def calc_expect_val(self) -> (list, int, float):
        """Calculate all one- and two-point correlation expectation values and return the one with highest absolute value."""

        # initialize dictionary for saving the correlations
        self.expect_val_dict = {}

        # this first part takes care of the case where all correlations are 0.
        Z = np.sin(2 * self.beta) * self._calc_single_terms(gamma=self.gamma, index=1)
        if np.abs(Z) > 0.:
            rounding_list = [
                [[self.problem.position_translater[1]], np.sign(Z), np.abs(Z)]
            ]
            max_expect_val = np.abs(Z)
        else:
            rounding_list = [[[self.problem.position_translater[1]], 1, 0.]]
            max_expect_val = 0.

        self.expect_val_dict[frozenset({1})] = Z

        # iterating through single-body terms
        for index in range(1, len(self.problem.matrix)):
            Z = np.sin(2 * self.beta) * self._calc_single_terms(
                gamma=self.gamma, index=index
            )
            self.expect_val_dict[frozenset({index})] = Z
            if np.abs(Z) > max_expect_val:
                rounding_list = [
                    [[self.problem.position_translater[index]], np.sign(Z), np.abs(Z)]
                ]
                max_expect_val = np.abs(Z)
            elif np.abs(Z) == max_expect_val:
                rounding_list.append(
                    [[self.problem.position_translater[index]], np.sign(Z), np.abs(Z)]
                )

        # iterating through two-body terms; on
        for index_large in range(1, len(self.problem.matrix)):
            for index_small in range(1, index_large):
                # we only compute correlations if the coupling coefficient is not 0. between variables index_large and index_small
                if self.problem.matrix[index_large, index_small] != 0:
                    b_part_term, c_part_term = self._calc_coupling_terms(
                        gamma=self.gamma,
                        index_large=index_large,
                        index_small=index_small,
                    )
                    ZZ = (
                        np.sin(4 * self.beta) * b_part_term
                        - ((np.sin(2 * self.beta)) ** 2) * c_part_term
                    )
                    self.expect_val_dict[frozenset({index_large, index_small})] = ZZ
                    if np.abs(ZZ) > max_expect_val:
                        rounding_list = [
                            [
                                [
                                    self.problem.position_translater[index_large],
                                    self.problem.position_translater[index_small],
                                ],
                                np.sign(ZZ),
                                np.abs(ZZ),
                            ]
                        ]
                        max_expect_val = np.abs(ZZ)
                    elif np.abs(ZZ) == max_expect_val:
                        rounding_list.append(
                            [
                                [
                                    self.problem.position_translater[index_large],
                                    self.problem.position_translater[index_small],
                                ],
                                np.sign(ZZ),
                                np.abs(ZZ),
                            ]
                        )

        # random tie-breaking of the largest correlation.
        random_index = np.random.randint(len(rounding_list))
        rounding_element = rounding_list[random_index]
        max_expect_val_location = rounding_element[0]
        max_expect_val_sign = rounding_element[1]
        max_expect_val = rounding_element[2]

        return max_expect_val_location, int(max_expect_val_sign), max_expect_val

    def optimize(self):
        """This function optimizes the QAOA parameters to minimize the energy."""
        self.gamma = 0
        self.beta, self.energy = self._calc_beta_energy(self.gamma)

        # rough grid search
        steps = 30
        self._calc_best_gamma(lb=0, ub=np.pi, steps=steps)

        # refined grid search
        lb = self.gamma - (np.pi / (steps - 1))
        ub = self.gamma + (np.pi / (steps - 1))
        self._calc_best_gamma(lb=lb, ub=ub, steps=steps)
        # computing the correlatios at the optimal parameters
        (
            max_expect_val_location,
            max_expect_val_sign,
            max_expect_val,
        ) = self.calc_expect_val()
        self.fixed_correl.append(
            [max_expect_val_location, max_expect_val_sign, max_expect_val]
        )

        return max_expect_val_location, max_expect_val_sign, max_expect_val

    ####################################################################################
    # Beginning of block with helper functions that calculate the numerics.            #
    ####################################################################################
    def _single_cos(self, i: int, gamma: float) -> float:
        """
        Calculate the cosine term in the single qubit expectation value.
        """
        # Extract relevant elements from row i of the problem matrix, excluding zeros
        vec_i = self.problem.matrix[i, 1:i]
        vec_i = np.append(vec_i, self.problem.matrix[i + 1 :, i])
        vec_i = vec_i[vec_i != 0]

        # Calculate the product of cosines and return
        return np.prod(np.cos(2 * gamma * vec_i))

    def _coupling_cos_0(self, i: int, j: int, gamma: float) -> float:
        """
        Calculate the cosine term in the two-qubit coupling term for expectation value.
        Careful: this function is not symmetric w.r.t. its first two arguments.
        """
        # Determine the smaller and larger of the two indices
        index_small = np.min((i, j))
        index_large = np.max((i, j))

        # Extract relevant elements based on the smaller and larger index
        vec_i = self.problem.matrix[i, 1:index_small]
        if index_small == i:
            vec_i = np.append(
                vec_i, self.problem.matrix[index_small + 1 : index_large, i]
            )
        else:
            vec_i = np.append(
                vec_i, self.problem.matrix[i, index_small + 1 : index_large]
            )
        vec_i = np.append(vec_i, self.problem.matrix[index_large + 1 :, i])

        # Remove zero elements and calculate the product of cosines
        return np.prod(np.cos(2 * gamma * vec_i[vec_i != 0]))

    def _coupling_cos_plus(self, i: int, j: int, gamma: float) -> float:
        """
        Calculate the 'plus' version of the cosine term for the two-qubit coupling in expectation values.
        """
        # Identify the smaller and larger indices
        index_small = np.min((i, j))
        index_large = np.max((i, j))

        # Extract relevant elements for smaller index
        vec_i = self.problem.matrix[index_small, 1:index_small]
        vec_i = np.append(
            vec_i, self.problem.matrix[index_small + 1 : index_large, index_small]
        )
        vec_i = np.append(vec_i, self.problem.matrix[index_large + 1 :, index_small])

        # Extract relevant elements for larger index
        vec_j = self.problem.matrix[index_large, 1:index_small]
        vec_j = np.append(
            vec_j, self.problem.matrix[index_large, index_small + 1 : index_large]
        )
        vec_j = np.append(vec_j, self.problem.matrix[index_large + 1 :, index_large])

        # Combine and filter out zero elements
        vec_non_zeros = (vec_i + vec_j)[vec_i + vec_j != 0]

        # Calculate the product of cosines
        return np.prod(np.cos(2 * gamma * vec_non_zeros))

    def _coupling_cos_minus(
        self, index_large: int, index_small: int, gamma: float
    ) -> float:
        """
        Calculate the 'minus' version of the cosine term for the two-qubit coupling in expectation values.
        """
        # Extract relevant elements for smaller index
        vec_i = self.problem.matrix[index_small, 1:index_small]
        vec_i = np.append(
            vec_i, self.problem.matrix[index_small + 1 : index_large, index_small]
        )
        vec_i = np.append(vec_i, self.problem.matrix[index_large + 1 :, index_small])

        # Extract relevant elements for larger index
        vec_j = self.problem.matrix[index_large, 1:index_small]
        vec_j = np.append(
            vec_j, self.problem.matrix[index_large, index_small + 1 : index_large]
        )
        vec_j = np.append(vec_j, self.problem.matrix[index_large + 1 :, index_large])

        # Combine, filter out zero elements, and take the difference
        vec_non_zeros = (vec_i - vec_j)[vec_i - vec_j != 0]

        # Calculate the product of cosines
        return np.prod(np.cos(2 * gamma * vec_non_zeros))

    def _calc_single_terms(self, gamma: float, index: int) -> float:
        """Helper function to compute terms for the two-point expectation values."""
        # Calculate the single-term part with sin and cos functions
        a_part_term = np.sin(
            2 * gamma * self.problem.matrix[index, index]
        ) * self._single_cos(index, gamma)
        return a_part_term

    def _calc_coupling_terms(
        self, gamma: float, index_large: int, index_small: int
    ) -> (float, float):
        """Helper function to compute coupling terms for the two-point expectation values."""

        # Calculate the b_part_term using sin and cos functions
        b_part_term = (
            0.5
            * np.sin(2 * gamma * self.problem.matrix[index_large, index_small])
            * (
                np.cos(2 * gamma * self.problem.matrix[index_large, index_large])
                * self._coupling_cos_0(index_large, index_small, gamma)
                + np.cos(2 * gamma * self.problem.matrix[index_small, index_small])
                * self._coupling_cos_0(index_small, index_large, gamma)
            )
        )

        # Constants and terms for calculating c_part_term
        c_0 = 0.5
        c_1 = np.cos(
            2
            * gamma
            * (
                self.problem.matrix[index_large, index_large]
                + self.problem.matrix[index_small, index_small]
            )
        ) * self._coupling_cos_plus(index_large, index_small, gamma)
        c_2 = np.cos(
            2
            * gamma
            * (
                self.problem.matrix[index_large, index_large]
                - self.problem.matrix[index_small, index_small]
            )
        ) * self._coupling_cos_minus(index_large, index_small, gamma)
        c_part_term = c_0 * (c_1 - c_2)

        return b_part_term, c_part_term

    def _calc_const(self, gamma):
        """Calculate the constant terms for optimizing beta."""
        a, b, c = 0, 0, 0

        # Calculate the 'a' constant term
        for index in range(1, len(self.problem.matrix)):
            a_term = self.problem.matrix[index, index] * self._calc_single_terms(
                gamma, index
            )
            a += a_term

        # timi = 0  # Variable for measuring time
        count = 0  # Variable for counting non-zero entries

        # Calculate the 'b' and 'c' constant terms
        for index_large in range(1, len(self.problem.matrix)):
            for index_small in range(1, index_large):
                if self.problem.matrix[index_large, index_small] != 0:
                    # start = timeit.default_timer()

                    b_part_term, c_part_term = self._calc_coupling_terms(
                        gamma, index_large, index_small
                    )

                    # stop = timeit.default_timer()
                    # timi += stop - start

                    b_term = self.problem.matrix[index_large, index_small] * b_part_term
                    c_term = self.problem.matrix[index_large, index_small] * c_part_term

                    b += b_term
                    c += c_term
                    count += 1
        self.a = a
        self.b = b
        self.c = c

    ####################################################################################
    # Ending of block with helper functions that calculate the numerics.               #
    ####################################################################################

    ####################################################################################
    # Beginning of block with helper functions to optimize the parameters              #
    ####################################################################################

    def _calc_beta_energy(self, gamma: float):
        """Calculate the optimal value of beta regarding the energy, dependent on the input gamma"""
        self._calc_const(gamma)

        # derivative helper function
        def __f(x):
            """Derivative of the energy."""
            return (
                2 * self.a * np.cos(2 * x)
                + 4 * self.b * np.cos(4 * x)
                - 4 * self.c * np.sin(2 * x) * np.cos(2 * x)
            )

        beta = float(fsolve(__f, 0.01, xtol=0.000001))
        energy = (
            self.a * np.sin(2 * beta)
            + self.b * np.sin(4 * beta)
            - self.c * ((np.sin(2 * beta)) ** 2)
        )

        # running solver for calculating the root of the derivative of the energy function
        for i in range(10):
            start_point = (np.pi / 10) * (i + 1)
            beta_try = float(fsolve(__f, start_point, xtol=0.000001))
            energy_try = (
                self.a * np.sin(2 * beta_try)
                + self.b * np.sin(4 * beta_try)
                - self.c * ((np.sin(2 * beta_try)) ** 2)
            )
            if energy_try < energy:
                energy = energy_try
                beta = beta_try

        # this is basically a backup code block which does a rough grid search
        # for the beta parameters in case
        # our solver gets stuck in a local minimum then this value of beta will be used
        beta_grid = 0
        energy_grid = (
            self.a * np.sin(2 * beta_grid)
            + self.b * np.sin(4 * beta_grid)
            - self.c * ((np.sin(2 * beta_grid)) ** 2)
        )
        for beta_try in np.linspace(0, np.pi, 50):
            energy_try = (
                self.a * np.sin(2 * beta_try)
                + self.b * np.sin(4 * beta_try)
                - self.c * ((np.sin(2 * beta_try)) ** 2)
            )
            if energy_try < energy_grid:
                energy_grid = energy_try
                beta_grid = beta_try

        if energy > energy_grid:
            energy = energy_grid
            beta = beta_grid

        return beta, energy

    def _calc_best_gamma(self, lb=0, ub=np.pi, steps=30):
        """Calculates best angles in a 30 points grid between a lower and upper bound"""
        for gamma in np.linspace(lb, ub, steps):
            beta, energy = self._calc_beta_energy(gamma)

            if energy < self.energy:
                self.gamma = gamma
                self.beta = beta
                self.energy = energy

    ####################################################################################
    # Ending of block with helper functions to optimize the parameters                 #
    ####################################################################################


# deprecated of now.
# def brute_force(self):
# """calculate optimal solution of the remaining variables (according to the remaining
# optimization problem) brute force"""
# x_in_dict = {}
# brute_forced_solution = {}
# count = 0
# single_energy_vector = copy.deepcopy(self.problem.matrix.diagonal())
# correl_energy_matrix = copy.deepcopy(self.problem.matrix)
# np.fill_diagonal(correl_energy_matrix, 0)

# for iter_var_list in it.product(
#     [-1, 1], repeat=(len(self.problem.position_translater) - 1)
# ):
#     vec = np.array([0])
#     vec = np.append(vec, iter_var_list)
#     E_current = self.problem.calc_energy(
#         vec, single_energy_vector, correl_energy_matrix
#     )

#     for i in range(1, len(vec)):
#         x_in_dict[self.problem.position_translater[i]] = iter_var_list[i - 1]
#     if count == 0:
#         E_best = copy.deepcopy(E_current)
#         brute_forced_solution = copy.deepcopy(x_in_dict)
#         count += 1
#     if float(E_current) < float(E_best):
#         brute_forced_solution = copy.deepcopy(x_in_dict)
#         E_best = copy.deepcopy(E_current)
# return brute_forced_solution
