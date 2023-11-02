import numpy as np

# Function to create a random assignment for the variables
def create_rnd_assignment(variable_list, rg):
    """
    Creates a random assignment for the given variable list.
    The zero'th index is introduced due to the form of the correlation matrix and doesn't correspond to a
    physical variable.
    """
    assignment = np.array([0])  # Start with an array containing only the zero'th index
    for num in variable_list:
        sign = rg.choice((1, -1))  # Randomly choose a sign for each variable
        assignment = np.append(assignment, np.array([sign * num]), axis=0)  # Append the signed variable to the assignment
    return assignment

# Function to calculate the energy of a given assignment
def calc_energy(assignment, single_energy_vector, correl_energy_matrix):
    """
    Calculate the energy of a given assignment.
    The energy is calculated using the assignment's sign vector, single energy vector and correlation energy matrix.
    """
    # Calculate energy using matrix multiplication and return the result
    E_calc = np.sign(assignment).T @ correl_energy_matrix @ np.sign(assignment) + single_energy_vector.T @ np.sign(assignment)
    return E_calc

# Function to flip a single spin in the assignment
def flip_single_spin(assignment, rg):
    """
    Randomly choose a variable and flip its sign.
    """
    flip_var = rg.choice(range(1, len(assignment)))  # Choose a variable to flip, excluding the zero'th index
    assignment[flip_var] = assignment[flip_var] * (-1)  # Flip the sign of the chosen variable
    return assignment