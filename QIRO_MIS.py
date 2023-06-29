import numpy as np
import itertools as it
import copy
import Calculating_Expectation_Values as Expectation_Values
from Generating_Problems import MIS
import networkx as nx
from aws_quera import find_mis

class QIRO(Expectation_Values.ExpectationValues):
    """
    :param problem_input: The problem object that shall be solved
    :param nc: size of remaining subproblems that are solved by brute force
    This class is responsible for the whole QIRO procedure; the output represents the optimized bitstring solution in the form
    of a dictionary as well as a list of optimal parameters from each elimination step
    """

    def __init__(self, problem_input, nc):
        super().__init__(problem=problem_input)
        # let us use the problem graph as the reference, and this current graph as the dynamic
        # object from which we will eliminate nodes:
        self.graph = copy.deepcopy(self.problem.graph)
        self.nc = nc
        self.assignment = []
        self.solution = []

    
    def update_single(self, variable_index, exp_value_sign):
        """Updates Hamiltonian according to fixed single point correlation"""
        node = variable_index - 1
        fixing_list = []
        assignments = []
        # if the node is included in the IS we remove its neighbors
        if exp_value_sign == 1:
            ns = copy.deepcopy(self.graph.neighbors(node))
            for n in ns:
                self.graph.remove_node(n)
                fixing_list.append([n + 1])
                assignments.append(-1)
        
        # in any case we remove the node which was selected by correlations:
        self.graph.remove_node(node)
        fixing_list.append([variable_index])
        assignments.append(exp_value_sign)

        # reinitailize the problem object with the new, updated, graph:
        self.problem = MIS(self.graph, self.problem.alpha)

        return fixing_list, assignments
    
    def update_correlation(self, variables, exp_value_sign):
        """Updates Hamiltonian according to fixed two point correlation -- RQAOA (for now)."""
        
        #     """This does the whole getting-of-coupled-vars mumbo-jumbo."""
        fixing_list = []
        assignments = []
        if exp_value_sign == 1:
            # if variables are correlated, then we set both to -1 
            # (as the independence constraint prohibits them from being +1 simultaneously). 
            for variable in variables:
                fixing_list.append([variable])
                assignments.append(-1)
                self.graph.remove_node(variable - 1)                
        else:
            print("Entered into anticorrelated case:")
            # we remove the things we need to remove are the ones connected to both node, which are not both node.
            mutual_neighbors = set(self.graph.neighbors(variables[0] - 1)) & set(self.graph.neighbors(variables[1] - 1))
            fixing_list = [[n + 1] for n in mutual_neighbors]
            assignments = [-1] * len(fixing_list)
            for n in mutual_neighbors:
                self.graph.remove_node(n)

        # reinitailize the problem object with the new, updated, graph:
        self.problem = MIS(self.graph, self.problem.alpha)

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

        fixing_list = [[n + 1] for n in sorted(prune_assignments.keys())]
        assignments = [prune_assignments[n] for n in sorted(prune_assignments.keys())]

        return fixing_list, assignments
    

    def execute(self, energy='best'):
        """Main QIRO function which produces the solution by applying the QIRO procedure."""
        self.opt_gamma = []
        self.opt_beta = []
        self.fixed_correlations = []
        step_nr = 0

        while self.graph.number_of_nodes() > 0:
            step_nr += 1
            print(f"Step: {step_nr}. Number of nodes: {self.graph.number_of_nodes()}.")
            fixed_variables = []
            if energy == 'best':
                self.optimize()
            else:
                assert isinstance(energy, (float, int))
                # get random parameters, as defined in the RQAOA class
                raise NotImplementedError("This is not implemented yet.")

                
            # sorts correlations in decreasing order. Ties are broken randomly.
            sorted_correlation_dict = sorted(self.max_exp_dict.items(), key=lambda item: (abs(item[1]), np.random.rand()), reverse=True)
            # we iterate until we remove a node
            which_correlation = 0
            while len(fixed_variables) == 0:
                exp_value_coeff, max_exp_value = sorted_correlation_dict[which_correlation]
                exp_value_coeff = [self.problem.position_translater[idx] for idx in exp_value_coeff]
                exp_value_sign = np.sign(max_exp_value).astype(int)

                if len(exp_value_coeff) == 1:
                    print(f"single var {exp_value_coeff}. Sign: {exp_value_sign}")
                    fixed_variables, assignments = self.update_single(*exp_value_coeff, exp_value_sign)
                    for var, assignment in zip(fixed_variables, assignments):

                        if var is None:
                            raise Exception("Variable to be eliminated is None. WTF?")
                        self.fixed_correlations.append([var, int(assignment), max_exp_value])


                else:
                    print(f'Correlation {exp_value_coeff}. Sign: {exp_value_sign}.')
                    fixed_variables, assignments = self.update_correlation(exp_value_coeff, exp_value_sign)
                    for var, assignment in zip(fixed_variables, assignments):
                        if var is None:
                            raise Exception("Variable to be eliminated is None. WTF?")
                        
                        self.fixed_correlations.append([var, int(assignment), max_exp_value])

                # perform pruning.
                pruned_variables, pruned_assignments = self.prune_graph()
                print(f"Pruned {len(pruned_variables)} variables.")
                
                for var, assignment in zip(pruned_variables, pruned_assignments):
                    if var is None:
                        raise Exception("Variable to be eliminated is None. WTF?")
                    self.fixed_correlations.append([var, assignment, None])
                
                fixed_variables += pruned_variables
                which_correlation += 1
                

                if len(fixed_variables) == 0:
                    print("No variables could be fixed.")
                    print(f"Attempting with the {which_correlation}. largest correlation.")
                else:
                    print(f"We have fixed the following variables: {fixed_variables}. Moving on.")
        
        solution = [var[0] * assig for var, assig, _ in self.fixed_correlations]
        sorted_solution = sorted(solution, key=lambda x: abs(x))
        print(f"Solution: {sorted_solution}")
        self.solution = np.array(sorted_solution).astype(int)

