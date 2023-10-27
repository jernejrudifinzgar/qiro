import numpy as np
import copy
import networkx as nx
from utils.Matrix import Matrix
from typing import List, Union


import numpy as np


class Problem:
    """
    General problem generator base class.
    """

    def __init__(self, seed: int) -> None:
        """
        Initialize the problem.
        """
        self.random_seed = seed

    def calc_energy(
        self,
        assignment: np.ndarray,
        single_energy_vector: np.ndarray,
        correl_energy_matrix: np.ndarray,
    ) -> float:
        """
        Calculates the energy of a given assignment.

        Parameters:
            assignment: np.ndarray - The variable assignment.
            single_energy_vector: np.ndarray - The single-variable energy contributions.
            correl_energy_matrix: np.ndarray - The correlation energy matrix.

        Returns:
            float: The calculated energy.
        """
        E_calc = np.sign(assignment).T @ correl_energy_matrix @ np.sign(
            assignment
        ) + single_energy_vector.T @ np.sign(assignment)
        return E_calc


class MAX2SAT(Problem):
    """Max 2-SAT problem generator."""

    def __init__(self, num_var: int, num_clauses: int, seed: int = 42) -> None:
        super().__init__(seed=seed)
        self.num_var = num_var
        self.num_clauses = num_clauses
        self.num_per_clause = 2
        self.var_list = []
        self.cnf = None
        self.cnf_start = None
        self.generate_formula()
        self.SAT_to_Hamiltonian()
        self.matrix_start = copy.deepcopy(self.matrix)

    def generate_formula(self) -> None:
        """Generates a random MAX-2-SAT formula."""

        # Initialize the list of variables
        variables = list(range(1, self.num_var + 1))
        self.cnf = []

        # Initialize the random generator
        rg = np.random.default_rng(self.random_seed)

        # Generate clauses for the MAX-2-SAT problem
        for _ in range(self.num_clauses):
            # Randomly choose variables for this clause
            var_in_clause = rg.choice(variables, self.num_per_clause, replace=True)

            # Randomly assign them to be either positive or negative
            neg_pos_in_clause = rg.choice([1, -1], self.num_per_clause, replace=True)

            # Keep track of the unique variables involved
            for var in var_in_clause:
                if var not in self.var_list:
                    self.var_list.append(var)

            # Handle the case where the two chosen variables for the clause are the same
            if var_in_clause[0] == var_in_clause[1]:
                # If polarities are opposite, the clause is automatically true; skip adding it
                if neg_pos_in_clause[0] != neg_pos_in_clause[1]:
                    pass

                # If polarities are the same, then it's sufficient to add one variable to the clause
                else:
                    clause = [var_in_clause[0] * neg_pos_in_clause[0]]
                    self.cnf.append(clause)

            # Handle the standard case where the two chosen variables for the clause are different
            else:
                clause = [
                    var_in_clause[0] * neg_pos_in_clause[0],
                    var_in_clause[1] * neg_pos_in_clause[1],
                ]
                self.cnf.append(clause)

        # Save a copy of the original CNF formula for use in recursive algorithms
        self.cnf_start = copy.deepcopy(self.cnf)

        # Sort variable list for easier handling later
        self.var_list.sort()

        # Make a copy of the remaining variable list for tracking
        self.remain_var_list = copy.deepcopy(self.var_list)

    def SAT_to_Hamiltonian(self) -> None:
        """Converts the MAX-2-SAT formula to a Hamiltonian matrix."""

        # Initialize a set to keep track of the unique positions in the Hamiltonian matrix
        self.position_translater = {0}

        # Loop through each clause to populate 'position_translater'
        for clause in self.cnf:
            positive_clause = []
            for element in range(len(clause)):
                # Take the absolute value of each element in the clause
                positive_clause.append(abs(clause[element]))
            # Add unique variables to 'position_translater'
            self.position_translater = self.position_translater.union(
                set(positive_clause)
            )

        # Convert the set to a sorted list
        self.position_translater = list(self.position_translater)
        self.position_translater.sort()

        # Initialize the Hamiltonian matrix with the appropriate size
        self.matrixClass = Matrix(len(self.position_translater))
        self.matrix = self.matrixClass.matrix

        # Populate the Hamiltonian matrix based on the clauses
        for clause in self.cnf:
            # If the clause has more than one variable
            if len(clause) > 1:
                # Convert the variables to their corresponding indices in the Hamiltonian matrix
                ind_0 = self.position_translater.index(np.abs(clause[0])) * np.sign(
                    clause[0]
                )
                ind_1 = self.position_translater.index(np.abs(clause[1])) * np.sign(
                    clause[1]
                )

                # Add off-diagonal elements for these two variables
                self.matrixClass.add_off_element(
                    i=int(ind_0), j=int(ind_1), const=1 / 4
                )

                # Add diagonal elements for each variable in the clause
                self.matrixClass.add_diag_element(i=-int(ind_0), const=1 / 4)
                self.matrixClass.add_diag_element(i=-int(ind_1), const=1 / 4)

            # If the clause has only one variable
            elif len(clause) == 1:
                # Convert the variable to its corresponding index in the Hamiltonian matrix
                ind_0 = self.position_translater.index(np.abs(clause[0])) * np.sign(
                    clause[0]
                )

                # Add a diagonal element for this variable
                self.matrixClass.add_diag_element(i=-int(ind_0), const=1 / 2)

    def calc_violated_clauses(self, solution: Union[np.ndarray, List[int]]):
        """Calculates the number of violations."""

        # Check the type of the solution
        if isinstance(solution, list) and not all(isinstance(x, int) for x in solution):
            raise TypeError("Solution must be a numpy array or list of integers")
            # Initialize E to store the number of clause violations
        E = 0

        # Use a set for faster look-up of variables in the solution
        solution_set = {(abs(var), np.sign(var)) for var in solution}

        # Loop through each original clause to calculate the number of violations
        for clause in self.cnf_start:
            # Assume clause is violated, then try to prove otherwise
            is_clause_violated = True

            # Loop through each variable in the clause
            for var in clause:
                abs_var, sign_var = abs(var), np.sign(var)
                if (abs_var, sign_var) in solution_set:
                    # Clause is satisfied, break out of inner loop
                    is_clause_violated = False
                    break

            # Increment the violation count if the clause was not satisfied
            E += is_clause_violated

        # Return the total number of clause violations
        return E


class MIS(Problem):
    """Maximum Independent Set problem generator."""

    def __init__(self, graph: nx.Graph, alpha: float = 1.1, seed: int = 42) -> None:
        """Initialize with a networkx graph object, and a penalty factor alpha."""
        super().__init__(seed=seed)
        self.graph = copy.deepcopy(graph)
        self.alpha = alpha
        self.var_list = None
        self.position_translater = None

        # compute the matrix (i.e., Hamiltonian) from the graph. Also sets the varlist!
        self.graph_to_matrix()
        self.remain_var_list = copy.deepcopy(self.var_list)

    def graph_to_matrix(self) -> None:
        """Transform the graph into its corresponding Hamiltonian matrix."""

        # matrix is one dimension larger due to the 0-th row and column, which are set to 0 by convention.
        self.matrixClass = Matrix(self.graph.number_of_nodes() + 1)
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
            self.matrixClass.add_diag_element(idx, -1 / 2)

        for correlation in self.graph.edges:
            # the first correlation + 1 comes from the fact that graph nodes run from 0...n-1
            # the fact that we add another +1 to the index is because the variables list runs 1....n
            # and the indices in the matrix run 0...n
            idx1, idx2 = (
                variables.index(correlation[0] + 1) + 1,
                variables.index(correlation[1] + 1) + 1,
            )
            self.matrixClass.add_off_element(idx1, idx2, self.alpha / 4)
            self.matrixClass.add_diag_element(idx1, self.alpha / 4)
            self.matrixClass.add_diag_element(idx2, self.alpha / 4)

        # we define the appropriate position translater
        self.position_translater = [0] + variables

    def evaluate_solution(self, solution: Union[np.ndarray, list]) -> (int, int):
        """Returns the number of violations and the size of the set found."""
        # Type check for solution
        if isinstance(solution, list) and not all(
            isinstance(x, (int, float)) for x in solution
        ):
            raise TypeError("Solution must be a numpy array or list of integers/floats")
        assert len(solution) == self.graph.number_of_nodes(), (
            f"Solution length {len(solution)} does not match"
            f" the number of nodes in the graph {self.graph.number_of_nodes()}."
        )
        number_of_violations = 0
        size_of_set = len(*np.where(np.array(solution) > 0.0))
        for n1, n2 in self.graph.edges:
            if solution[n1] > 0.0 and solution[n2] > 0.0:
                number_of_violations += 1

        return number_of_violations, size_of_set
