from problem_generation.Generating_Problems import Problem
import copy
from utils.Matrix import Matrix
import numpy as np
from typing import Union
import networkx as nx
from itertools import chain, product, combinations


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


# Misc. helper functions


def find_mis(graph: nx.Graph, maximum=True) -> (int, list):
    """Finds a maximAL independent set (IS) of a graph, and returns its bitstrings. If
    maximum is set to True, then we return the maximUM such IS."""
    if not maximum:
        colored_nodes = nx.maximal_independent_set(graph)
        return len(colored_nodes), colored_nodes
    else:
        solutions = []
        maximum = 0
        for subset in powerset(graph.nodes):
            if is_independent(graph, subset):
                if len(subset) > maximum:
                    solutions = [subset]
                    maximum = len(subset)
                elif len(subset) == maximum:
                    solutions.append(subset)
        return maximum, solutions


def powerset(iterable):
    """Returns the powerset of an iterable."""
    s = list(iterable)
    return chain.from_iterable(combinations(s, q) for q in range(len(s) + 1))


def is_independent(graph, solution):
    """Checks if a given solution is independent."""
    if isinstance(solution, dict):
        sorted_keys = sorted(solution.keys())
        solution = np.where([int(solution[i] > 0) for i in sorted_keys])[0]
    for edge in product(solution, repeat=2):
        if graph.has_edge(*edge):
            return False
    return True
