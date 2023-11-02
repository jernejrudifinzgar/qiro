from QIRO import QIRO
import copy
import numpy as np


class QIRO_MAX_2SAT(QIRO):
    """
    QIRO class for the Max-2-SAT problem.
    """

    def __init__(self, nc_input, expectation_values_input):
        super().__init__(nc=nc_input, expectation_values=expectation_values_input)
        self.num_inference_appl = 1

    def execute(self, backtracking=False):
        """Wrapper for the execution function below. backtracking variable determines
        whether or not backtracking is performed."""

        return QIRO_BT_execute(self, backtracking=backtracking)

    ####################################################################################
    # Beginning of block with helper functions that perform the updates.               #
    ####################################################################################

    def inference_rules(self):
        """Applies inference rules to the problem."""
        self.num_inference_appl = 1
        # perform as long as inference rules actually simplify something.
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
        """Updates formula given a QAOA (or another correlation)."""
        # one point correlation
        if len(exp_value_coeff) == 1:
            self.problem.position_translater.remove(exp_value_coeff[0])
            self.problem.remain_var_list.remove(exp_value_coeff[0])
            clause_index = 0
            # iterate over clauses
            for clause in copy.deepcopy(self.problem.cnf):
                for variable in clause:
                    # if assignment agrees with clause:
                    if np.abs(variable) == exp_value_coeff[0]:
                        if np.sign(variable) == exp_value_sign:
                            self.problem.cnf.remove(clause)
                            clause_index -= 1
                        # if it does not agree, and this variable was the last variable left, we
                        # set the clause as unsatisfied
                        elif (np.sign(variable) == -exp_value_sign) and (
                            len(clause) == 1
                        ):
                            self.problem.cnf.remove(clause)
                            clause_index -= 1
                        else:
                            self.problem.cnf[clause_index].remove(variable)
                clause_index += 1
        else:
            # two-point correlation; this replaces the literal as dictated by the chosen correlation
            self.problem.position_translater.remove(exp_value_coeff[1])
            self.problem.remain_var_list.remove(exp_value_coeff[1])
            clause_index = 0
            for clause in copy.deepcopy(self.problem.cnf):
                # differentiate between the case where the length of the clause is one
                # .. and perform the replacement
                # TODO: Is done a bit awkwardly for now; to be beautified in the future.
                if len(clause) == 1:
                    if exp_value_coeff[1] in clause:
                        self.problem.cnf[clause_index].remove(exp_value_coeff[1])
                        self.problem.cnf[clause_index].append(
                            exp_value_sign * exp_value_coeff[0]
                        )
                    elif (-exp_value_coeff[1]) in clause:
                        self.problem.cnf[clause_index].remove((-exp_value_coeff[1]))
                        self.problem.cnf[clause_index].append(
                            (-exp_value_sign * exp_value_coeff[0])
                        )
                # cases where clause is of length 2:
                # perform the correct replacements:
                # TODO: Is done a bit awkwardly for now; to be beautified in the future.

                elif (
                    (np.abs(clause[0]) == exp_value_coeff[0])
                    and (np.abs(clause[1]) == exp_value_coeff[1])
                ) or (
                    (np.abs(clause[0]) == exp_value_coeff[1])
                    and (np.abs(clause[1]) == exp_value_coeff[0])
                ):
                    if np.sign(clause[0]) * np.sign(clause[1]) != exp_value_sign:
                        self.problem.cnf.remove(clause)
                        clause_index -= 1
                    else:
                        if exp_value_coeff[1] in copy.deepcopy(
                            self.problem.cnf[clause_index]
                        ):
                            self.problem.cnf[clause_index].remove(exp_value_coeff[1])
                        else:
                            self.problem.cnf[clause_index].remove((-exp_value_coeff[1]))
                elif exp_value_coeff[1] in clause:
                    self.problem.cnf[clause_index].remove(exp_value_coeff[1])
                    self.problem.cnf[clause_index].append(
                        exp_value_sign * exp_value_coeff[0]
                    )
                elif (-exp_value_coeff[1]) in clause:
                    self.problem.cnf[clause_index].remove((-exp_value_coeff[1]))
                    self.problem.cnf[clause_index].append(
                        (-exp_value_sign * exp_value_coeff[0])
                    )
                clause_index += 1

    def search_variables(self, formula, variable):
        """
        Searches for instances of a given variable in a propositional formula represented as a CNF expression.
        
        Args:
        formula (list of lists of ints): The propositional formula in CNF, where each clause is a list of integers.
                                        Positive integers represent the corresponding variables, and negative integers
                                        represent their negations.
        variable (int): The target variable to search for in the formula. Should be a positive integer.
        
        Returns:
        tuple: A tuple containing four lists:
            - pos_unit_clause_index: Indices of unit clauses where the variable appears positively.
            - neg_unit_clause_index: Indices of unit clauses where the variable appears negatively.
            - pos_duo_clause_index: Indices of clauses with more than one literal where the variable appears positively.
            - neg_duo_clause_index: Indices of clauses with more than one literal where the variable appears negatively.
        """
        
        # Initialize lists to store the indices of different types of clauses
        pos_unit_clause_index = []  # Unit clauses with positive instance of the variable
        neg_unit_clause_index = []  # Unit clauses with negative instance of the variable
        pos_duo_clause_index = []   # Clauses with more than one literal and positive instance of the variable
        neg_duo_clause_index = []   # Clauses with more than one literal and negative instance of the variable
        
        clause_index = 0  # Initialize the clause index
        
        # Iterate over all clauses in the formula
        for clause in formula:
            # Check if the positive instance of the variable is in the clause
            if variable in clause:
                if len(clause) == 1:
                    # If it is a unit clause, add its index to pos_unit_clause_index
                    pos_unit_clause_index.append(clause_index)
                else:
                    # If it is a larger clause, add its index to pos_duo_clause_index
                    pos_duo_clause_index.append(clause_index)
            
            # Check if the negative instance of the variable is in the clause
            if (-variable) in clause:
                if len(clause) == 1:
                    # If it is a unit clause, add its index to neg_unit_clause_index
                    neg_unit_clause_index.append(clause_index)
                else:
                    # If it is a larger clause, add its index to neg_duo_clause_index
                    neg_duo_clause_index.append(clause_index)
            
            # Move to the next clause
            clause_index += 1
        
        # Return the collected indices
        return (
            pos_unit_clause_index,
            neg_unit_clause_index,
            pos_duo_clause_index,
            neg_duo_clause_index,
        )

    def dominating_unit_clause_help(
        self,
        variable,
        dominating_unit_clause_index,
        dominating_duo_clause_index,
        submissive_unit_clause_index,
        submissive_duo_clause_index,
    ):
        """
        A helper function for the dominating unit clause inference rule.
        
        Args:
        variable (int): The target variable for which the inference rule is applied.
        dominating_unit_clause_index (list of ints): Indices of the unit clauses where the variable appears and dominates.
        dominating_duo_clause_index (list of ints): Indices of the clauses with more than one literal where the variable appears and dominates.
        submissive_unit_clause_index (list of ints): Indices of the unit clauses where the variable appears in its negated form.
        submissive_duo_clause_index (list of ints): Indices of the clauses with more than one literal where the variable appears in its negated form.
        
        This function applies the dominating unit clause inference rule, removing certain clauses and literals based on the presence of the dominating variable, and updating the remaining formula and variable list.
        """
        self.assignment.append(variable)  # Add the variable to the assignment list
        self.problem.remain_var_list.remove(np.abs(variable))  # Remove the variable from the remaining variable list

        # Remove the negation of the variable from all submissive duo clauses
        for clause_index in submissive_duo_clause_index:
            self.problem.cnf[clause_index].remove((-variable))

        del_duo_list = []  # List to store the duo clauses that need to be deleted
        # Iterate through the dominating duo clauses
        for index_clause in dominating_duo_clause_index:
            del_duo_list.append(self.problem.cnf[index_clause])  # Add the clause to the deletion list
        for del_clause in del_duo_list:
            del_clause_copy = copy.deepcopy(del_clause)  # Make a copy of the clause
            del_clause_copy.remove(variable)  # Remove the dominating variable from the copy
            removed_variable = del_clause_copy[0]  # Get the other variable in the clause
            self.problem.cnf.remove(del_clause)  # Remove the original clause from the formula
            # Search for the occurrences of the other variable in the formula
            (
                pos_unit_clause_index,
                neg_unit_clause_index,
                pos_duo_clause_index,
                neg_duo_clause_index,
            ) = self.search_variables(self.problem.cnf, np.abs(removed_variable))
            # If the other variable doesn't appear elsewhere in the formula, remove it from the remaining variable list and add it to the assignment list
            if (
                (len(pos_unit_clause_index) == 0)
                and (len(neg_unit_clause_index) == 0)
                and (len(pos_duo_clause_index) == 0)
                and (len(neg_duo_clause_index) == 0)
            ):
                self.problem.remain_var_list.remove(np.abs(removed_variable))
                self.assignment.append(removed_variable)

        # Remove the dominating unit clauses from the formula
        for _ in range(len(dominating_unit_clause_index)):
            self.problem.cnf.remove([variable])
        # Remove the submissive unit clauses from the formula
        for _ in range(len(submissive_unit_clause_index)):
            self.problem.cnf.remove([-variable])
        self.num_inference_appl += 1  # Increment the number of applied inferences

    def dominating_unit_clause(self):
        """
        The dominating unit clause inference rule function.
        
        This function applies the dominating unit clause inference rule to the whole formula, updating it based on the presence of dominating variables.
        """
        # Iterate through the remaining variables in the formula
        for variable in self.problem.remain_var_list:
            # Search for the occurrences of the variable in the formula
            (
                pos_unit_clause_index,
                neg_unit_clause_index,
                pos_duo_clause_index,
                neg_duo_clause_index,
            ) = self.search_variables(self.problem.cnf, variable)
            # Apply the dominating unit clause rule if the variable appears positively in more unit clauses than it and its negation appear in other clauses
            if len(pos_unit_clause_index) >= (
                len(neg_unit_clause_index) + len(neg_duo_clause_index)
            ):
                self.dominating_unit_clause_help(
                    variable,
                    pos_unit_clause_index,
                    pos_duo_clause_index,
                    neg_unit_clause_index,
                    neg_duo_clause_index,
                )
            # Apply the dominating unit clause rule if the negation of the variable appears in more unit clauses than it and its negation appear in other clauses
            elif len(neg_unit_clause_index) >= (
                len(pos_unit_clause_index) + len(pos_duo_clause_index)
            ):
                self.dominating_unit_clause_help(
                    -variable,
                    neg_unit_clause_index,
                    neg_duo_clause_index,
                    pos_unit_clause_index,
                    pos_duo_clause_index,
                )

    def almost_common_clause(self):
        """
        This function applies the almost common clause inference rule on the CNF formula. 
        It searches for pairs of clauses that can be simplified by this rule and updates the formula accordingly.
        """
        # Iterate through the remaining variables in the formula
        for variable in self.problem.remain_var_list:
            # Search for occurrences of the variable and its negation in the formula
            (
                pos_unit_clause_index,
                neg_unit_clause_index,
                pos_duo_clause_index,
                neg_duo_clause_index,
            ) = self.search_variables(self.problem.cnf, variable)

            delete_clause_list = []  # List to store clauses that need to be deleted
            added_clause = []  # List to store the new clauses that need to be added
            
            # Iterate through the clauses where the variable appears
            for clause_pos in pos_duo_clause_index:
                index_negative_deleting_clauses = []  # To keep track of clauses that have already been considered for deletion
                
                # Iterate through the clauses where the negation of the variable appears
                for clause_neg in neg_duo_clause_index:
                    # Copy the clauses and remove the current variable or its negation
                    pos_copy_clause = copy.deepcopy(self.problem.cnf[clause_pos])
                    pos_copy_clause.remove(variable)
                    neg_copy_clause = copy.deepcopy(self.problem.cnf[clause_neg])
                    neg_copy_clause.remove(-variable)
                    
                    # Check if the rest of the clauses are identical (indicating that we can apply the almost common clause rule)
                    if (
                        pos_copy_clause == neg_copy_clause
                        and clause_neg not in index_negative_deleting_clauses
                    ):
                        index_negative_deleting_clauses.append(clause_neg)
                        delete_clause_list.append(self.problem.cnf[clause_pos])
                        delete_clause_list.append(self.problem.cnf[clause_neg])
                        added_clause.append(pos_copy_clause)  # Add the simplified clause to the list
                        pos_duo_clause_index.remove(clause_pos)
                        neg_duo_clause_index.remove(clause_neg)
                        self.num_inference_appl += 1  # Increment the count of applied inferences
                        break
            
            # Remove the clauses that are to be deleted
            for removing_index in range(len(delete_clause_list)):
                self.problem.cnf.remove(delete_clause_list[removing_index])
            
            # Add the new clauses that have been simplified
            for removing_index in range(len(added_clause)):
                self.problem.cnf.append(added_clause[removing_index])
            
            # Re-search for occurrences of the variable to update the indices after modifying the formula
            (
                pos_unit_clause_index,
                neg_unit_clause_index,
                pos_duo_clause_index,
                neg_duo_clause_index,
            ) = self.search_variables(self.problem.cnf, variable)
            
            # If the variable no longer appears in the formula, remove it from the remaining variables list and add it to the assignment list
            if (
                (len(pos_unit_clause_index) == 0)
                and (len(neg_unit_clause_index) == 0)
                and (len(pos_duo_clause_index) == 0)
                and (len(neg_duo_clause_index) == 0)
            ):
                self.problem.remain_var_list.remove(variable)
                self.assignment.append(variable)


    def pure_literal(self):
        """
        This function applies the pure literal elimination inference rule on the CNF formula. 
        If a variable appears only in one polarity (either positive or negative) throughout the formula,
        all clauses containing that literal can be removed, and the variable is assigned the value that satisfies those clauses.
        """
        # Iterate through the remaining variables in the formula
        for variable in self.problem.remain_var_list:
            # Search for occurrences of the variable and its negation in the formula
            (
                pos_unit_clause_index,
                neg_unit_clause_index,
                pos_duo_clause_index,
                neg_duo_clause_index,
            ) = self.search_variables(self.problem.cnf, variable)

            # Check if the variable appears only in negative clauses
            if (
                len(pos_unit_clause_index + pos_duo_clause_index) == 0
                and len(neg_unit_clause_index + neg_duo_clause_index) > 0
            ):
                # If so, remove all clauses containing the negative literal
                del_duo_list = [self.problem.cnf[index_clause] for index_clause in neg_duo_clause_index]
                for del_clause in del_duo_list:
                    self.problem.cnf.remove(del_clause)
                
                for _ in range(len(neg_unit_clause_index)):
                    self.problem.cnf.remove([-variable])
                
                # Assign the variable a value that satisfies the negative clauses
                self.assignment.append(-variable)
                self.problem.remain_var_list.remove(variable)
                self.num_inference_appl += 1  # Increment the count of applied inferences
            
            # Check if the variable appears only in positive clauses
            elif (
                len(neg_unit_clause_index + neg_duo_clause_index) == 0
                and len(pos_unit_clause_index + pos_duo_clause_index) > 0
            ):
                # If so, remove all clauses containing the positive literal
                del_duo_list = [self.problem.cnf[index_clause] for index_clause in pos_duo_clause_index]
                for del_clause in del_duo_list:
                    self.problem.cnf.remove(del_clause)
                
                for _ in range(len(pos_unit_clause_index)):
                    self.problem.cnf.remove([variable])
                
                # Assign the variable a value that satisfies the positive clauses
                self.assignment.append(variable)
                self.problem.remain_var_list.remove(variable)
                self.num_inference_appl += 1  # Increment the count of applied inferences


    def help_for_complementary(
        self, pos_unit_clause_index, neg_unit_clause_index, variable
    ):
        """
        This function helps resolve cases where there are complementary unit clauses 
        (i.e., clauses with a single literal where the variable appears in both its 
        positive and negative forms). It simplifies the CNF formula by removing 
        complementary unit clauses until one form of the variable is exhausted.
        """
        # Determine the minimum number of complementary unit clauses
        min_clauses = np.min((len(pos_unit_clause_index), len(neg_unit_clause_index)))
        
        # Iterate through the complementary unit clauses
        for _ in range(min_clauses):
            # Remove a positive unit clause
            self.problem.cnf.remove([variable])
            # Remove a negative unit clause
            self.problem.cnf.remove([-variable])
            # Increment the count of applied inferences
            self.num_inference_appl += 1

    def complementary_unit_clause(self):
        """
        This function handles complementary unit clauses within the CNF formula.
        A complementary unit clause is one where a variable appears in both its positive and negative forms.
        """
        # Iterate through the remaining variables in the CNF formula
        for variable in self.problem.remain_var_list:
            # Search for occurrences of the current variable and its negation in the CNF formula
            (
                pos_unit_clause_index,
                neg_unit_clause_index,
                pos_duo_clause_index,
                neg_duo_clause_index,
            ) = self.search_variables(self.problem.cnf, variable)
            
            # Handle complementary unit clauses
            self.help_for_complementary(
                pos_unit_clause_index, neg_unit_clause_index, variable
            )
            
            # Search again for occurrences of the current variable and its negation in the CNF formula
            # This is necessary because the CNF formula might have changed after handling complementary unit clauses
            (
                pos_unit_clause_index,
                neg_unit_clause_index,
                pos_duo_clause_index,
                neg_duo_clause_index,
            ) = self.search_variables(self.problem.cnf, variable)
            
            # Check if the variable can be determined to be True or False based on the presence of unit clauses
            if (
                len(pos_duo_clause_index) == 0
                and len(neg_duo_clause_index) == 0
                and len(pos_unit_clause_index) == len(neg_unit_clause_index)
            ):
                # If the variable only appears in unit clauses, and the number of positive and negative appearances are equal,
                # it is safe to assign a value to the variable and remove it from the list of remaining variables
                self.problem.remain_var_list.remove(variable)
                self.assignment.append(variable)

    def calc_solution(self, x_start_sol):
        """
        This function calculates the solution based on the initial solution and fixed correlations.
        :param x_start_sol: A dictionary representing the initial solution.
        """
        # Adding initial solution assignments to the final assignment list
        for key in x_start_sol:
            self.assignment.append(key * x_start_sol[key])
            
        # Iterating through the fixed correlations to determine the rest of the variable assignments
        for correl in reversed(self.fixed_correlations):
            # Handling cases where the correlation coefficient is 0, setting it to 1
            if correl[1] == 0:
                correl[1] = 1

            # If the correlation only involves one variable, directly assign its value
            if len(correl[0]) == 1:
                self.assignment.append(correl[0][0] * correl[1])
            else:
                # If the correlation involves two variables, calculate the second variable's value based on the first
                test_count = 0
                sign = 1
                for variable in self.assignment:
                    if int(np.abs(variable)) == int(correl[0][0]):
                        sign = np.sign(variable)
                        test_count += 1

                # If the variable is not yet assigned, it doesn't matter since it appears in clauses that are fulfilled anyway
                # and is assigned a default positive sign
                if test_count == 0:
                    sign = 1
                
                # Assign the value to the second variable based on the first variable and correlation coefficient
                self.assignment.append(correl[0][1] * sign * correl[1])
                
        # Filling the solution list based on the final assignments
        for variable in self.problem.var_list:
            for assigned_variable in self.assignment:
                if variable == np.abs(assigned_variable):
                    self.solution.append(assigned_variable)
                    break
            else:
                # If a variable is not assigned, it is added to the solution list as is (positive)
                if (variable not in self.assignment) and ((-variable) not in self.assignment):
                    self.solution.append(variable)



def QIRO_BT_execute(qiro, backtracking=True):
    """Helper function that takes care of the backtracking and execution, that is then
    called from the class instance."""

    def calc_E(_qiro):
        _qiro.problem.SAT_to_Hamiltonian()
        brute_forced_sol = _qiro.brute_force()

        _qiro.calc_solution(brute_forced_sol)

        E = _qiro.problem.calc_violated_clauses(np.array(_qiro.solution).astype(int))
        return E, _qiro.solution

    def eliminations(_qiro, first_run=False):
        backtrack_data_new = []

        count = 0
        while len(_qiro.problem.remain_var_list) > _qiro.nc:
            if not (first_run) or count != 0:
                _qiro.inference_rules()
            else:
                _qiro.problem.SAT_to_Hamiltonian()

            if len(_qiro.problem.remain_var_list) <= _qiro.nc:
                return _qiro, backtrack_data_new

            (
                exp_value_coeff,
                exp_value_sign,
                max_exp_value,
            ) = _qiro.expectation_values.optimize()
            _qiro.fixed_correlations.append(
                [exp_value_coeff, int(exp_value_sign), max_exp_value]
            )
            backtrack_data_new.append(
                [max_exp_value, exp_value_coeff, exp_value_sign, copy.deepcopy(_qiro)]
            )
            _qiro.update_formula(exp_value_coeff, exp_value_sign)
            count += 1
        return _qiro, backtrack_data_new

    # BnB = QIRO(problem_input=generated_problem, nc=nc)
    qiro.fixed_correlations = []
    qiro_best, backtrack_data = eliminations(_qiro=qiro, first_run=True)
    E_first, first_solution = calc_E(qiro_best)

    improved_backtrack_solution = copy.deepcopy(first_solution)
    E_best_backtrack = copy.deepcopy(E_first)

    if backtracking:
        for data in backtrack_data:
            if len(data) == 0:
                break
            current_qiro = data[3]

            current_qiro.fixed_correlations[-1][1] = current_qiro.fixed_correlations[
                -1
            ][1] * (-1)
            current_qiro.update_formula(data[1], -data[2])
            current_qiro.inference_rules()

            BnB_current, backtrack_data_to_add = eliminations(_qiro=current_qiro)
            E, solution_try = calc_E(copy.deepcopy(BnB_current))

            if E < E_best_backtrack:
                E_best_backtrack = E
                improved_backtrack_solution = solution_try
    return (
        E_best_backtrack,
        E_first,
        np.array(first_solution).astype(int),
        np.array(improved_backtrack_solution).astype(int),
    )
