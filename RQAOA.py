import numpy as np
from QIRO import QIRO
import copy
import networkx as nx
from operator import itemgetter
import os
import random

class RQAOA(QIRO):
    """
    RQAOA is a subclass of QIRO, designed to implement the Recursive Quantum Approximate Optimization Algorithm for combinatorial optimization problems.
    
    Attributes:
        nq (int): The number of qubits used for the quantum part of the algorithm.
        coupling_del_list (list): A list to store information related to the couplings between qubits.
        single_del_list (list): A list to store information related to the individual qubits.
        solution (list): A list to store the final solution of the optimization problem.
        fixed_correlations (list): A list to store correlations that are fixed during the optimization process.
        type_of_problem (str): A string indicating the type of the optimization problem being solved.
    
    Args:
        expectation_values_input (ExpectationValues): An object that can compute expectation values, which are needed to evaluate the cost function for different quantum states.
        nc (int): The size of the remaining problem that will be solved classically, using brute force.
        type_of_problem (str, optional): The type of the optimization problem to solve. Currently supports "MAX_2_SAT" and "MIS" (Maximum Independent Set). Default is "MAX_2_SAT".
    """

    def __init__(self, expectation_values_input, nc, type_of_problem="MAX_2_SAT", connectivity_output=False):
        """
        Initializes the RQAOA instance.
        
        The constructor initializes various attributes required for the RQAOA algorithm and inherits from the QIRO class.
        """
        # Call the constructor of the parent class QIRO
        super().__init__(nc=nc, expectation_values=expectation_values_input)
        
        # Initializing attributes specific to RQAOA
        self.nq = len(self.problem.var_list) - self.nc
        self.coupling_del_list = []
        self.single_del_list = []
        self.solution = []
        self.fixed_correlations = []
        self.type_of_problem = type_of_problem
        self.energies_list = []
        self.connectivity_output = connectivity_output

    
    def execute(self):
        """
        Solves the given problem via RQAOA and returns the solution and solution quality.

        Returns:
            E (int or float): The quality of the found solution, specific to the problem type.
            self.solution (numpy.ndarray): The found solution as an array of variable assignments.
        """
        # Iterate through nq steps of RQAOA, where nq is the number of quantum variables
        energies_list = []
        for step in range(self.nq):
            print(f"RQAOA Step: {step + 1}")

            # Optimize to find the expectation values, and determine which correlation to fix
            exp_value_coeff, exp_value_sign, max_exp_value = self.expectation_values.optimize()
            self.energies_list.append(self.expectation_values.energy)
            # Store the fixed correlation for later use in solution reconstruction
            self.fixed_correlations.append([exp_value_coeff, int(exp_value_sign), max_exp_value])

            # If we're fixing a single-point correlation
            if len(exp_value_coeff) == 1:
                # Update the Hamiltonian to reflect the fixed single-point correlation
                self.update_single([self.problem.position_translater.index(exp_value_coeff[0])], exp_value_sign)
                # Update lists to reflect that a variable has been fixed
                self.problem.remain_var_list.remove(exp_value_coeff[0])
                self.problem.position_translater.remove(exp_value_coeff[0])
            # If we're fixing a two-point correlation
            else:
                # Update the Hamiltonian to reflect the fixed two-point correlation
                self.update_correlation([self.problem.position_translater.index(exp_value_coeff[0]), self.problem.position_translater.index(exp_value_coeff[1])], exp_value_sign)
                # Update lists to reflect that a variable has been fixed
                self.problem.remain_var_list.remove(exp_value_coeff[1])

                self.problem.position_translater.remove(exp_value_coeff[1])
                
        # Solve the reduced problem using brute force
        brute_forced_solution = self.brute_force()
        # Reconstruct the complete solution from the brute-forced solution and fixed correlations
        self.calc_complete_solution(brute_forced_solution)
        # Convert the solution to a numpy array of integers
        self.solution = np.array(self.solution).astype(int)

        # Evaluate the quality of the found solution based on the problem type
        if self.type_of_problem == "MAX_2_SAT":
            E = self.problem.calc_violated_clauses(self.solution)
        elif self.type_of_problem == "MIS":
            E = self.problem.evaluate_solution(self.solution)
        elif self.type_of_problem == "MAXCUT":
            E = self.problem.evaluate_cuts(self.solution)
        else:
            raise Exception(f"Problem type {self.type_of_problem} unknown.")

        return E, self.solution
    
    ####################################################################################
    # Beginning of block with helper functions that perform the updates.               #
    ####################################################################################

    def update_single(self, exp_value_coeff, exp_value_sign):
        """
        Updates the Hamiltonian of the quantum problem according to fixed single-point correlations.
        
        Args:
            exp_value_coeff (tuple or list): A pair (i, j) representing the indices of the Hamiltonian matrix element to be updated.
            exp_value_sign (float): The sign of the expectation value to be used for updating the Hamiltonian.
            
        Description:
            This method modifies the problem Hamiltonian based on the single-point correlations that have been determined and fixed during the RQAOA process. 
            It adjusts the diagonal elements of the Hamiltonian matrix, reflecting the contributions from the fixed correlations to the remaining problem. 
            The method then removes the row and column associated with the fixed variable from the Hamiltonian matrix.
        """
        # Extract the coupling coefficients associated with the variable being fixed
        coupled_values = self.problem.matrix[exp_value_coeff[0], :exp_value_coeff[0]+1]
        coupled_values = np.append(coupled_values, self.problem.matrix[(exp_value_coeff[0]+1):, exp_value_coeff[0]])
        
        # Update the diagonal elements of the Hamiltonian matrix
        self.problem.matrix[np.diag_indices_from(self.problem.matrix)] += (coupled_values * exp_value_sign)
        
        # Remove the row and column associated with the fixed variable from the Hamiltonian matrix
        self.problem.matrix = np.delete(self.problem.matrix, exp_value_coeff[0], 0)
        self.problem.matrix = np.delete(self.problem.matrix, exp_value_coeff[0], 1)
       

    def update_correlation(self, exp_value_coeff, exp_value_sign):
        """
        Updates the Hamiltonian of the quantum problem according to fixed two-point correlations.
        
        Args:
            exp_value_coeff (tuple or list): A pair (i, j) representing the indices of the Hamiltonian matrix elements to be updated.
            exp_value_sign (float): The sign of the expectation value to be used for updating the Hamiltonian.
            
        Description:
            This method modifies the problem Hamiltonian based on the two-point correlations that have been determined and fixed during the RQAOA process. 
            It adjusts the diagonal and off-diagonal elements of the Hamiltonian matrix, reflecting the contributions from the fixed correlations 
            to the remaining problem. The method then removes the row and column associated with one of the variables involved in the two-point 
            correlation from the Hamiltonian matrix.
        """
        # Update the diagonal element of the Hamiltonian matrix for the variable not being removed
        self.problem.matrix[exp_value_coeff[0], exp_value_coeff[0]] += self.problem.matrix[exp_value_coeff[1], exp_value_coeff[1]] * exp_value_sign
        
        # Extract the coupling coefficients associated with the variable being removed
        coupled_values = self.problem.matrix[exp_value_coeff[1], :exp_value_coeff[1] + 1]
        coupled_values = np.append(coupled_values, self.problem.matrix[(exp_value_coeff[1] + 1):, exp_value_coeff[1]])
        
        # Update the Hamiltonian matrix with contributions from the variable being removed
        for pos in range(1, len(coupled_values)):
            if exp_value_coeff[0] > pos:
                self.problem.matrix[exp_value_coeff[0], pos] += coupled_values[pos] * exp_value_sign
            elif exp_value_coeff[0] < pos:
                self.problem.matrix[pos, exp_value_coeff[0]] += coupled_values[pos] * exp_value_sign
        
        # Remove the row and column associated with the variable being removed from the Hamiltonian matrix
        self.problem.matrix = np.delete(self.problem.matrix, exp_value_coeff[1], 0)
        self.problem.matrix = np.delete(self.problem.matrix, exp_value_coeff[1], 1)
        

        # Remove the node associated with the variable being removed from the graph and create new edges
        ns = copy.deepcopy(self.problem.graph.neighbors(self.problem.position_translater[exp_value_coeff[1]]-1))
        self.problem.graph.remove_node(self.problem.position_translater[exp_value_coeff[1]]-1)
        
        for n in ns: 
            if n != self.problem.position_translater[exp_value_coeff[0]]-1:
                self.problem.graph.add_edge(n, self.problem.position_translater[exp_value_coeff[0]]-1)

        max_connectivity = 0
        list_connectivity = []
        for node in self.problem.graph.nodes():
            neighbors = self.problem.graph.neighbors(node)
            connectivity = len(list(neighbors))
            list_connectivity.append(connectivity)
            if connectivity>max_connectivity:
                max_connectivity=connectivity
        
        if self.connectivity_output:
            print(f'Max connectivity of graph with {len(self.problem.matrix)-1} nodes: ', max_connectivity)
            print('Connectivity:', list_connectivity)
        
            f = open(f"connectivity_n_{len(self.problem.original_graph.nodes())}_p_{self.expectation_values.p}.txt", "a")
            f.write(f"\n\nMax connectivity of graph with {len(self.problem.matrix)-1} nodes: {max_connectivity}")
            f.write(f"\nConnectivity all nodes: {list_connectivity}")
            f.close()


    def calc_complete_solution(self, brute_forced_solution):
        """
        Starting from the small brute forced solution, the full complete solution of the problem is reconstructed
        via the rounded correlations (in fixed_correlations list).

        Args:
            brute_forced_solution (dict): The initial solution obtained by brute force for a smaller subset of the problem.
                                          The keys are variable indices, and the values are their assigned values (-1 or 1).

        Note:
            This method updates the solution attribute with the complete solution of the problem.
        """
        # Iterate through the fixed correlations in reverse order
        for corr in reversed(self.fixed_correlations):
            # If the correlation value is 0, change it to 1 to avoid multiplying by 0 in subsequent calculations
            if corr[1] == 0:
                corr[1] = 1

            # If the correlation involves a single variable, directly assign its value
            if len(corr[0]) == 1:
                brute_forced_solution[corr[0][0]] = int(corr[1])
            # If the correlation involves two variables, deduce the value of one variable based on the other
            else:
                brute_forced_solution[corr[0][1]] = int(brute_forced_solution[corr[0][0]] * corr[1])
       
        # Iterate through all variables in the problem
        for variable in self.problem.var_list:
            # Check if each variable is in the brute_forced_solution, and if so, add it to the solution
            for assigned_variable in brute_forced_solution:
                if variable == assigned_variable:
                    self.solution.append(assigned_variable * brute_forced_solution[assigned_variable])

####################################################

class RQAOA_recalculate(QIRO):
    """
    RQAOA is a subclass of QIRO, designed to implement the Recursive Quantum Approximate Optimization Algorithm for combinatorial optimization problems.
    
    Attributes:
        nq (int): The number of qubits used for the quantum part of the algorithm.
        coupling_del_list (list): A list to store information related to the couplings between qubits.
        single_del_list (list): A list to store information related to the individual qubits.
        solution (list): A list to store the final solution of the optimization problem.
        fixed_correlations (list): A list to store correlations that are fixed during the optimization process.
        type_of_problem (str): A string indicating the type of the optimization problem being solved.
    
    Args:
        expectation_values_input (ExpectationValues): An object that can compute expectation values, which are needed to evaluate the cost function for different quantum states.
        nc (int): The size of the remaining problem that will be solved classically, using brute force.
        type_of_problem (str, optional): The type of the optimization problem to solve. Currently supports "MAX_2_SAT" and "MIS" (Maximum Independent Set). Default is "MAX_2_SAT".
    """

    def __init__(self, expectation_values_input, nc, recalculations=6, type_of_problem="MAX_2_SAT"):
        """
        Initializes the RQAOA instance.
        
        The constructor initializes various attributes required for the RQAOA algorithm and inherits from the QIRO class.
        """
        # Call the constructor of the parent class QIRO
        super().__init__(nc=nc, expectation_values=expectation_values_input)
        
        # Initializing attributes specific to RQAOA
        self.nq = len(self.problem.var_list) - self.nc
        self.coupling_del_list = []
        self.single_del_list = []
        self.solution = []
        self.fixed_correlations = []
        self.type_of_problem = type_of_problem
        self.energies_list = []
        self.recalculations = recalculations
    
    def execute(self):
        """
        Solves the given problem via RQAOA and returns the solution and solution quality.

        Returns:
            E (int or float): The quality of the found solution, specific to the problem type.
            self.solution (numpy.ndarray): The found solution as an array of variable assignments.
        """
        # Iterate through nq steps of RQAOA, where nq is the number of quantum variables
        energies_list = []
        for step in range(self.nq):
            print(f"RQAOA Step: {step + 1}")

            if step%self.recalculations==0:

                # Optimize to find the expectation values, and determine which correlation to fix
                exp_value_coeff, exp_value_sign, max_exp_value = self.expectation_values.optimize()
                self.energies_list.append(self.expectation_values.energy)
                expect_val_dict = self.expectation_values.expect_val_dict.copy()
                #highest_values_dict = dict(sorted(expect_val_dict.items(), key=itemgetter(1), reverse=True)[:6]) 
                #highest_values_dict=dict(sorted(highest_values_dict.items() , key=self.get_max , reverse = True))
                #print(highest_values_dict)
    

            #print(highest_values_dict)
            #exp_value_coeff = list(highest_values_dict.keys())[0]
            #print(exp_value_coeff)
            #exp_value_sign = np.sign(highest_values_dict[exp_value_coeff])
            #max_exp_value = abs(highest_values_dict[exp_value_coeff])
            
            #exp_value_coeff = max(expect_val_dict, key=expect_val_dict.get)
            #print(exp_value_coeff)


            
            max_value = max(expect_val_dict.values())
            exp_value_coeff_list = [k for k, v in expect_val_dict.items() if v == max_value]

            index = random.randrange(len(exp_value_coeff_list))
            exp_value_coeff = exp_value_coeff_list[index]

            exp_value_sign = np.sign(expect_val_dict[exp_value_coeff])
            max_exp_value = abs(expect_val_dict[exp_value_coeff])
            exp_value_coeff_translated = ([self.expectation_values.problem.position_translater[min(list(exp_value_coeff))], self.problem.position_translater[max(list(exp_value_coeff))]])

            # Store the fixed correlation for later use in solution reconstruction
            self.fixed_correlations.append([exp_value_coeff_translated, int(exp_value_sign), max_exp_value])

            # If we're fixing a single-point correlation
            if len(exp_value_coeff) == 1:
                # Update the Hamiltonian to reflect the fixed single-point correlation
                self.update_single([self.problem.position_translater.index(exp_value_coeff_translated[0])], exp_value_sign)
                # Update lists to reflect that a variable has been fixed
                self.problem.remain_var_list.remove(exp_value_coeff_translated[0])
                self.problem.position_translater.remove(exp_value_coeff_translated[0])
            # If we're fixing a two-point correlation
            else:
                # Update the Hamiltonian to reflect the fixed two-point correlation
                self.update_correlation([self.problem.position_translater.index(exp_value_coeff_translated[0]), self.problem.position_translater.index(exp_value_coeff_translated[1])], exp_value_sign)
                # Update lists to reflect that a variable has been fixed
                self.problem.remain_var_list.remove(exp_value_coeff_translated[1])

                self.problem.position_translater.remove(exp_value_coeff_translated[1])

                # del expect_val_dict[exp_value_coeff]
                # for correlation in expect_val_dict.copy().keys():
                #     if min(list(correlation))==max(list(exp_value_coeff)) or max(list(correlation))==max(list(exp_value_coeff)):
                #         del expect_val_dict[correlation]
                #     elif min(list(correlation))>max(list(exp_value_coeff)):
                #         expect_val_dict[frozenset({min(list(correlation))-1, max(list(correlation))-1})] = expect_val_dict[correlation]
                #         del expect_val_dict[correlation]
                #     elif max(list(correlation))>max(list(exp_value_coeff)):
                #         expect_val_dict[frozenset({min(list(correlation)), max(list(correlation))-1})] = expect_val_dict[correlation]
                #         del expect_val_dict[correlation]




                    
                    




                # del highest_values_dict[exp_value_coeff]    
                # for correlation in highest_values_dict.copy().keys():
                #     if list(correlation)[0]==max(list(exp_value_coeff)):
                #         highest_values_dict[frozenset({min(list(exp_value_coeff)), list(correlation)[1]})] = highest_values_dict[correlation]
                #         del highest_values_dict[correlation]
                #         print('added', highest_values_dict)
                #     if list(correlation)[1]==max(list(exp_value_coeff)):
                #         highest_values_dict[frozenset({list(correlation)[0], min(list(exp_value_coeff))})] = highest_values_dict[correlation]
                #         del highest_values_dict[correlation]
                            
        # Solve the reduced problem using brute force
        brute_forced_solution = self.brute_force()
        # Reconstruct the complete solution from the brute-forced solution and fixed correlations
        self.calc_complete_solution(brute_forced_solution)
        # Convert the solution to a numpy array of integers
        self.solution = np.array(self.solution).astype(int)

        # Evaluate the quality of the found solution based on the problem type
        if self.type_of_problem == "MAX_2_SAT":
            E = self.problem.calc_violated_clauses(self.solution)
        elif self.type_of_problem == "MIS":
            E = self.problem.evaluate_solution(self.solution)
        elif self.type_of_problem == "MAXCUT":
            E = self.problem.evaluate_cuts(self.solution)
        else:
            raise Exception(f"Problem type {self.type_of_problem} unknown.")

        return E, self.solution
    
    ####################################################################################
    # Beginning of block with helper functions that perform the updates.               #
    ####################################################################################

    def update_single(self, exp_value_coeff, exp_value_sign):
        """
        Updates the Hamiltonian of the quantum problem according to fixed single-point correlations.
        
        Args:
            exp_value_coeff (tuple or list): A pair (i, j) representing the indices of the Hamiltonian matrix element to be updated.
            exp_value_sign (float): The sign of the expectation value to be used for updating the Hamiltonian.
            
        Description:
            This method modifies the problem Hamiltonian based on the single-point correlations that have been determined and fixed during the RQAOA process. 
            It adjusts the diagonal elements of the Hamiltonian matrix, reflecting the contributions from the fixed correlations to the remaining problem. 
            The method then removes the row and column associated with the fixed variable from the Hamiltonian matrix.
        """
        # Extract the coupling coefficients associated with the variable being fixed
        coupled_values = self.problem.matrix[exp_value_coeff[0], :exp_value_coeff[0]+1]
        coupled_values = np.append(coupled_values, self.problem.matrix[(exp_value_coeff[0]+1):, exp_value_coeff[0]])
        
        # Update the diagonal elements of the Hamiltonian matrix
        self.problem.matrix[np.diag_indices_from(self.problem.matrix)] += (coupled_values * exp_value_sign)
        
        # Remove the row and column associated with the fixed variable from the Hamiltonian matrix
        self.problem.matrix = np.delete(self.problem.matrix, exp_value_coeff[0], 0)
        self.problem.matrix = np.delete(self.problem.matrix, exp_value_coeff[0], 1)
       

    def update_correlation(self, exp_value_coeff, exp_value_sign):
        """
        Updates the Hamiltonian of the quantum problem according to fixed two-point correlations.
        
        Args:
            exp_value_coeff (tuple or list): A pair (i, j) representing the indices of the Hamiltonian matrix elements to be updated.
            exp_value_sign (float): The sign of the expectation value to be used for updating the Hamiltonian.
            
        Description:
            This method modifies the problem Hamiltonian based on the two-point correlations that have been determined and fixed during the RQAOA process. 
            It adjusts the diagonal and off-diagonal elements of the Hamiltonian matrix, reflecting the contributions from the fixed correlations 
            to the remaining problem. The method then removes the row and column associated with one of the variables involved in the two-point 
            correlation from the Hamiltonian matrix.
        """
        # Update the diagonal element of the Hamiltonian matrix for the variable not being removed
        self.problem.matrix[exp_value_coeff[0], exp_value_coeff[0]] += self.problem.matrix[exp_value_coeff[1], exp_value_coeff[1]] * exp_value_sign
        
        # Extract the coupling coefficients associated with the variable being removed
        coupled_values = self.problem.matrix[exp_value_coeff[1], :exp_value_coeff[1] + 1]
        coupled_values = np.append(coupled_values, self.problem.matrix[(exp_value_coeff[1] + 1):, exp_value_coeff[1]])
        
        # Update the Hamiltonian matrix with contributions from the variable being removed
        for pos in range(1, len(coupled_values)):
            if exp_value_coeff[0] > pos:
                self.problem.matrix[exp_value_coeff[0], pos] += coupled_values[pos] * exp_value_sign
            elif exp_value_coeff[0] < pos:
                self.problem.matrix[pos, exp_value_coeff[0]] += coupled_values[pos] * exp_value_sign
        
        # Remove the row and column associated with the variable being removed from the Hamiltonian matrix
        self.problem.matrix = np.delete(self.problem.matrix, exp_value_coeff[1], 0)
        self.problem.matrix = np.delete(self.problem.matrix, exp_value_coeff[1], 1)
        

        # Remove the node associated with the variable being removed from the graph and create new edges
        ns = copy.deepcopy(self.problem.graph.neighbors(self.problem.position_translater[exp_value_coeff[1]]-1))
        self.problem.graph.remove_node(self.problem.position_translater[exp_value_coeff[1]]-1)
        
        for n in ns: 
            if n != self.problem.position_translater[exp_value_coeff[0]]-1:
                self.problem.graph.add_edge(n, self.problem.position_translater[exp_value_coeff[0]]-1)

        max_connectivity = 0
        list_connectivity = []
        for node in self.problem.graph.nodes():
            neighbors = self.problem.graph.neighbors(node)
            connectivity = len(list(neighbors))
            list_connectivity.append(connectivity)
            if connectivity>max_connectivity:
                max_connectivity=connectivity
        
        print(f'Max connectivity of graph with {len(self.problem.matrix)-1} nodes: ', max_connectivity)
        print('Connectivity:', list_connectivity)


    def calc_complete_solution(self, brute_forced_solution):
        """
        Starting from the small brute forced solution, the full complete solution of the problem is reconstructed
        via the rounded correlations (in fixed_correlations list).

        Args:
            brute_forced_solution (dict): The initial solution obtained by brute force for a smaller subset of the problem.
                                          The keys are variable indices, and the values are their assigned values (-1 or 1).

        Note:
            This method updates the solution attribute with the complete solution of the problem.
        """
        # Iterate through the fixed correlations in reverse order
        for corr in reversed(self.fixed_correlations):
            # If the correlation value is 0, change it to 1 to avoid multiplying by 0 in subsequent calculations
            if corr[1] == 0:
                corr[1] = 1

            # If the correlation involves a single variable, directly assign its value
            if len(corr[0]) == 1:
                brute_forced_solution[corr[0][0]] = int(corr[1])
            # If the correlation involves two variables, deduce the value of one variable based on the other
            else:
                brute_forced_solution[corr[0][1]] = int(brute_forced_solution[corr[0][0]] * corr[1])
       
        # Iterate through all variables in the problem
        for variable in self.problem.var_list:
            # Check if each variable is in the brute_forced_solution, and if so, add it to the solution
            for assigned_variable in brute_forced_solution:
                if variable == assigned_variable:
                    self.solution.append(assigned_variable * brute_forced_solution[assigned_variable])
                    
                    



################################################################################

class RQAOA_shrink(QIRO):
    """
    RQAOA is a subclass of QIRO, designed to implement the Recursive Quantum Approximate Optimization Algorithm for combinatorial optimization problems.
    
    Attributes:
        nq (int): The number of qubits used for the quantum part of the algorithm.
        coupling_del_list (list): A list to store information related to the couplings between qubits.
        single_del_list (list): A list to store information related to the individual qubits.
        solution (list): A list to store the final solution of the optimization problem.
        fixed_correlations (list): A list to store correlations that are fixed during the optimization process.
        type_of_problem (str): A string indicating the type of the optimization problem being solved.
    
    Args:
        expectation_values_input (ExpectationValues): An object that can compute expectation values, which are needed to evaluate the cost function for different quantum states.
        nc (int): The size of the remaining problem that will be solved classically, using brute force.
        type_of_problem (str, optional): The type of the optimization problem to solve. Currently supports "MAX_2_SAT" and "MIS" (Maximum Independent Set). Default is "MAX_2_SAT".
    """

    def __init__(self, expectation_values_input, nc, type_of_problem="MAX_2_SAT", connectivity_output=False):
        """
        Initializes the RQAOA instance.
        
        The constructor initializes various attributes required for the RQAOA algorithm and inherits from the QIRO class.
        """
        # Call the constructor of the parent class QIRO
        super().__init__(nc=nc, expectation_values=expectation_values_input)
        
        # Initializing attributes specific to RQAOA
        self.nq = len(self.problem.var_list) - self.nc
        self.coupling_del_list = []
        self.single_del_list = []
        self.solution = []
        self.fixed_correlations = []
        self.type_of_problem = type_of_problem
        self.energies_list = []
        self.connectivity_output = connectivity_output

    
    def execute(self):
        """
        Solves the given problem via RQAOA and returns the solution and solution quality.

        Returns:
            E (int or float): The quality of the found solution, specific to the problem type.
            self.solution (numpy.ndarray): The found solution as an array of variable assignments.
        """
        # Iterate through nq steps of RQAOA, where nq is the number of quantum variables
        energies_list = []
        for step in range(10):
            print(f"RQAOA Step: {step + 1}")

            # Optimize to find the expectation values, and determine which correlation to fix
            exp_value_coeff, exp_value_sign, max_exp_value = self.expectation_values.optimize()
            self.energies_list.append(self.expectation_values.energy)
            # Store the fixed correlation for later use in solution reconstruction
            self.fixed_correlations.append([exp_value_coeff, int(exp_value_sign), max_exp_value])

            # If we're fixing a single-point correlation
            if len(exp_value_coeff) == 1:
                # Update the Hamiltonian to reflect the fixed single-point correlation
                self.update_single([self.problem.position_translater.index(exp_value_coeff[0])], exp_value_sign)
                # Update lists to reflect that a variable has been fixed
                self.problem.remain_var_list.remove(exp_value_coeff[0])
                self.problem.position_translater.remove(exp_value_coeff[0])
            # If we're fixing a two-point correlation
            else:
                # Update the Hamiltonian to reflect the fixed two-point correlation
                self.update_correlation([self.problem.position_translater.index(exp_value_coeff[0]), self.problem.position_translater.index(exp_value_coeff[1])], exp_value_sign)
                # Update lists to reflect that a variable has been fixed 
                self.problem.remain_var_list.remove(exp_value_coeff[1])

                self.problem.position_translater.remove(exp_value_coeff[1])
            

        return self.problem
    
    ####################################################################################
    # Beginning of block with helper functions that perform the updates.               #
    ####################################################################################

    def update_single(self, exp_value_coeff, exp_value_sign):
        """
        Updates the Hamiltonian of the quantum problem according to fixed single-point correlations.
        
        Args:
            exp_value_coeff (tuple or list): A pair (i, j) representing the indices of the Hamiltonian matrix element to be updated.
            exp_value_sign (float): The sign of the expectation value to be used for updating the Hamiltonian.
            
        Description:
            This method modifies the problem Hamiltonian based on the single-point correlations that have been determined and fixed during the RQAOA process. 
            It adjusts the diagonal elements of the Hamiltonian matrix, reflecting the contributions from the fixed correlations to the remaining problem. 
            The method then removes the row and column associated with the fixed variable from the Hamiltonian matrix.
        """
        # Extract the coupling coefficients associated with the variable being fixed
        coupled_values = self.problem.matrix[exp_value_coeff[0], :exp_value_coeff[0]+1]
        coupled_values = np.append(coupled_values, self.problem.matrix[(exp_value_coeff[0]+1):, exp_value_coeff[0]])
        
        # Update the diagonal elements of the Hamiltonian matrix
        self.problem.matrix[np.diag_indices_from(self.problem.matrix)] += (coupled_values * exp_value_sign)
        
        # Remove the row and column associated with the fixed variable from the Hamiltonian matrix
        self.problem.matrix = np.delete(self.problem.matrix, exp_value_coeff[0], 0)
        self.problem.matrix = np.delete(self.problem.matrix, exp_value_coeff[0], 1)
       

    def update_correlation(self, exp_value_coeff, exp_value_sign):
        """
        Updates the Hamiltonian of the quantum problem according to fixed two-point correlations.
        
        Args:
            exp_value_coeff (tuple or list): A pair (i, j) representing the indices of the Hamiltonian matrix elements to be updated.
            exp_value_sign (float): The sign of the expectation value to be used for updating the Hamiltonian.
            
        Description:
            This method modifies the problem Hamiltonian based on the two-point correlations that have been determined and fixed during the RQAOA process. 
            It adjusts the diagonal and off-diagonal elements of the Hamiltonian matrix, reflecting the contributions from the fixed correlations 
            to the remaining problem. The method then removes the row and column associated with one of the variables involved in the two-point 
            correlation from the Hamiltonian matrix.
        """
        # Update the diagonal element of the Hamiltonian matrix for the variable not being removed
        self.problem.matrix[exp_value_coeff[0], exp_value_coeff[0]] += self.problem.matrix[exp_value_coeff[1], exp_value_coeff[1]] * exp_value_sign
        
        # Extract the coupling coefficients associated with the variable being removed
        coupled_values = self.problem.matrix[exp_value_coeff[1], :exp_value_coeff[1] + 1]
        coupled_values = np.append(coupled_values, self.problem.matrix[(exp_value_coeff[1] + 1):, exp_value_coeff[1]])
        
        # Update the Hamiltonian matrix with contributions from the variable being removed
        for pos in range(1, len(coupled_values)):
            if exp_value_coeff[0] > pos:
                self.problem.matrix[exp_value_coeff[0], pos] += coupled_values[pos] * exp_value_sign
            elif exp_value_coeff[0] < pos:
                self.problem.matrix[pos, exp_value_coeff[0]] += coupled_values[pos] * exp_value_sign
        
        # Remove the row and column associated with the variable being removed from the Hamiltonian matrix
        self.problem.matrix = np.delete(self.problem.matrix, exp_value_coeff[1], 0)
        self.problem.matrix = np.delete(self.problem.matrix, exp_value_coeff[1], 1)
        

        # Remove the node associated with the variable being removed from the graph and create new edges
        ns = copy.deepcopy(self.problem.graph.neighbors(self.problem.position_translater[exp_value_coeff[1]]-1))
        self.problem.graph.remove_node(self.problem.position_translater[exp_value_coeff[1]]-1)
        
        for n in ns: 
            if n != self.problem.position_translater[exp_value_coeff[0]]-1:
                self.problem.graph.add_edge(n, self.problem.position_translater[exp_value_coeff[0]]-1)

        max_connectivity = 0
        list_connectivity = []
        for node in self.problem.graph.nodes():
            neighbors = self.problem.graph.neighbors(node)
            connectivity = len(list(neighbors))
            list_connectivity.append(connectivity)
            if connectivity>max_connectivity:
                max_connectivity=connectivity
        
        if self.connectivity_output:
            print(f'Max connectivity of graph with {len(self.problem.matrix)-1} nodes: ', max_connectivity)
            print('Connectivity:', list_connectivity)
        
            # f = open(f"connectivity_n_{len(self.problem.original_graph.nodes())}_p_{self.expectation_values.p}.txt", "a")
            # f.write(f"\n\nMax connectivity of graph with {len(self.problem.matrix)-1} nodes: {max_connectivity}")
            # f.write(f"\nConnectivity all nodes: {list_connectivity}")
            # f.close()


    def calc_complete_solution(self, brute_forced_solution):
        """
        Starting from the small brute forced solution, the full complete solution of the problem is reconstructed
        via the rounded correlations (in fixed_correlations list).

        Args:
            brute_forced_solution (dict): The initial solution obtained by brute force for a smaller subset of the problem.
                                          The keys are variable indices, and the values are their assigned values (-1 or 1).

        Note:
            This method updates the solution attribute with the complete solution of the problem.
        """
        # Iterate through the fixed correlations in reverse order
        for corr in reversed(self.fixed_correlations):
            # If the correlation value is 0, change it to 1 to avoid multiplying by 0 in subsequent calculations
            if corr[1] == 0:
                corr[1] = 1

            # If the correlation involves a single variable, directly assign its value
            if len(corr[0]) == 1:
                brute_forced_solution[corr[0][0]] = int(corr[1])
            # If the correlation involves two variables, deduce the value of one variable based on the other
            else:
                brute_forced_solution[corr[0][1]] = int(brute_forced_solution[corr[0][0]] * corr[1])
       
        # Iterate through all variables in the problem
        for variable in self.problem.var_list:
            # Check if each variable is in the brute_forced_solution, and if so, add it to the solution
            for assigned_variable in brute_forced_solution:
                if variable == assigned_variable:
                    self.solution.append(assigned_variable * brute_forced_solution[assigned_variable])






