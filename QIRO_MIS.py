from QIRO import QIRO
from expval_calculation.SingleLayerQAOA import SingleLayerQAOAExpectationValues
from problem_generation.Generate_MIS import MIS, find_mis
import copy
import networkx as nx
import numpy as np


class QIRO_MIS(QIRO):
    """
    :param problem_input: The problem object that shall be solved
    :param nc: size of remaining subproblems that are solved by brute force
    This class is responsible for the whole QIRO procedure; the output represents the optimized bitstring solution in the form
    of a dictionary as well as a list of optimal parameters from each elimination step
    """

    def __init__(self, nc_input, expectation_values_input):
        super().__init__(nc=nc_input, expectation_values=expectation_values_input)
        # let us use the problem graph as the reference, and this current graph as the dynamic
        # object from which we will eliminate nodes:
        self.graph = copy.deepcopy(self.problem.graph)

    def execute(self) -> None:
        """Main QIRO function which produces the solution by applying the QIRO procedure."""
        self.opt_gamma = []
        self.opt_beta = []
        self.fixed_correlations = []
        step_nr = 0

        while self.graph.number_of_nodes() > 0:
            step_nr += 1
            print(f"Step: {step_nr}. Number of nodes: {self.graph.number_of_nodes()}.")
            fixed_variables = []
            # optimize the correlations
            self.expectation_values.optimize()

            # sorts correlations in decreasing order. Ties are broken randomly.
            sorted_correlation_dict = sorted(
                self.expectation_values.expect_val_dict.items(),
                key=lambda item: (abs(item[1]), np.random.rand()),
                reverse=True,
            )
            # we iterate until we remove a node
            which_correlation = 0
            while len(fixed_variables) == 0:
                max_expect_val_location, max_expect_val = sorted_correlation_dict[
                    which_correlation
                ]

                max_expect_val_location = [
                    self.problem.position_translater[idx]
                    for idx in max_expect_val_location
                ]
                max_expect_val_sign = np.sign(max_expect_val).astype(int)

                # Differentiate the case of 1- and 2- point correlations:
                if len(max_expect_val_location) == 1:
                    # One point correlation case.
                    print(
                        f"single var {max_expect_val_location}. Sign: {max_expect_val_sign}"
                    )
                    fixed_variables, assignments = self._update_single(
                        *max_expect_val_location, max_expect_val_sign
                    )
                    for var, assignment in zip(fixed_variables, assignments):
                        if var is None:
                            raise Exception("Variable to be eliminated is None?")
                        self.fixed_correlations.append(
                            [var, int(assignment), max_expect_val]
                        )
                else:
                    # Two point correlation case
                    print(
                        f"Correlation {max_expect_val_location}. Sign: {max_expect_val_sign}."
                    )
                    fixed_variables, assignments = self._update_correlation(
                        max_expect_val_location, max_expect_val_sign
                    )
                    for var, assignment in zip(fixed_variables, assignments):
                        if var is None:
                            raise Exception("Variable to be eliminated is None. WTF?")

                        self.fixed_correlations.append(
                            [var, int(assignment), max_expect_val]
                        )

                # perform pruning; i.e., brute-forcing small connected components.
                pruned_variables, pruned_assignments = self._prune_graph()
                print(f"Pruned {len(pruned_variables)} variables.")
                # save assignments performed by pruning.
                for var, assignment in zip(pruned_variables, pruned_assignments):
                    if var is None:
                        raise Exception("Variable to be eliminated is None?")
                    self.fixed_correlations.append([var, assignment, None])

                fixed_variables += pruned_variables
                which_correlation += 1

                if len(fixed_variables) == 0:
                    print("No variables could be fixed.")
                    print(
                        f"Attempting with the {which_correlation}. largest correlation."
                    )
                else:
                    print(
                        f"We have fixed the following variables: {fixed_variables}. Moving on."
                    )
        # extract the solution into the correct format
        solution = [var[0] * assig for var, assig, _ in self.fixed_correlations]
        sorted_solution = sorted(solution, key=lambda x: abs(x))
        print(f"Solution: {sorted_solution}")
        self.solution = np.array(sorted_solution).astype(int)

    ################################################################################
    # Helper functions.                                                            #
    ################################################################################

    def _update_single(self, variable_index, max_expect_val_sign):
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

 
        if self.expectation_values.type == "SingleLayerQAOAExpectationValue":
            self.expectation_values = SingleLayerQAOAExpectationValues(self.problem)
        # elif self.expectation_values.type == "QtensorQAOAExpectationValuesMIS":
        #     self.expectation_values = QtensorQAOAExpectationValuesMIS(
        #         self.problem, self.expectation_values.p
        #     )

        return fixing_list, assignments

    def _update_correlation(self, variables, max_expect_val_sign):
        """Updates Hamiltonian according to fixed two point correlation -- RQAOA (for now)."""

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
            # we remove the things we need to remove are the ones connected to both node, which are not both node.
            mutual_neighbors = set(self.graph.neighbors(variables[0] - 1)) & set(
                self.graph.neighbors(variables[1] - 1)
            )
            fixing_list = [[n + 1] for n in mutual_neighbors]
            assignments = [-1] * len(fixing_list)
            for n in mutual_neighbors:
                self.graph.remove_node(n)

        # reinitailize the problem object with the new, updated, graph:
        self.problem = MIS(self.graph, self.problem.alpha)

        if self.expectation_values.type == "SingleLayerQAOAExpectationValue":
            self.expectation_values = SingleLayerQAOAExpectationValues(self.problem)
        # elif self.expectation_values.type == "QtensorQAOAExpectationValuesMIS":
        #     self.expectation_values = QtensorQAOAExpectationValuesMIS(
        #         self.problem, self.expectation_values.p
        #     )

        return fixing_list, assignments

    def _prune_graph(self):
        """Prunes the graph by removing all connected components that have less than nc nodes. The assignments are determined
        to be the maximum independent sets of the connected components. The self.graph is updated correspondingly.
        """

        # get connected components
        connected_components = copy.deepcopy(list(nx.connected_components(self.graph)))
        prune_assignments = {}
        for component in connected_components:
            if len(component) < self.nc:
                subgraph = self.graph.subgraph(component)
                _, miss = find_mis(subgraph)
                prune_assignments.update(
                    {n: 1 if n in miss[0] else -1 for n in subgraph.nodes}
                )

        # remove component from graph
        for node in prune_assignments.keys():
            self.graph.remove_node(node)

        self.problem = MIS(self.graph, self.problem.alpha)

        if self.expectation_values.type == "SingleLayerQAOAExpectationValue":
            self.expectation_values = SingleLayerQAOAExpectationValues(self.problem)
        # elif self.expectation_values.type == "QtensorQAOAExpectationValuesMIS":
        #     self.expectation_values = QtensorQAOAExpectationValuesMIS(
        #         self.problem, self.expectation_values.p
        #     )

        fixing_list = [[n + 1] for n in sorted(prune_assignments.keys())]
        assignments = [prune_assignments[n] for n in sorted(prune_assignments.keys())]

        return fixing_list, assignments
