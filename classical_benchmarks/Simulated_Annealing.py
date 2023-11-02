import numpy as np
import copy
from classical_benchmarks.monte_carlo_helpers import create_rnd_assignment, calc_energy, flip_single_spin


def Execute(problem_matrix, variables, num_iterations, temp, random_seed):
    global rg
    # Initialize a random number generator with the provided seed.
    rg = np.random.default_rng(random_seed)
    
    # Create an initial random assignment for the variables.
    assignment = np.sign(create_rnd_assignment(variable_list=variables, rg=rg))

    # Transform the correlation matrix into a form such that the overall energy can be calculated with matrix products via the assignment-sign vector.
    # This involves splitting the correlation matrix into the diagonal (single-point correlations) and off-diagonal terms (set the diagonal to zero).
    single_energy_vector = copy.deepcopy(problem_matrix.diagonal())
    correl_energy_matrix = copy.deepcopy(problem_matrix)
    np.fill_diagonal(correl_energy_matrix, 0)

    # Initialize the best found state and energy.
    best_state = assignment
    best_E = calc_energy(best_state, single_energy_vector=single_energy_vector, correl_energy_matrix=correl_energy_matrix)

    # Initialize the current state and energy.
    curr_state = copy.deepcopy(best_state)
    curr_E = copy.deepcopy(best_E)

    # Main loop for the simulated annealing algorithm.
    for i in range(num_iterations):
        # Propose a new state by flipping a single spin in the current state.
        cand_state = flip_single_spin(assignment=copy.deepcopy(curr_state), rg=rg)
        # Calculate the energy of the proposed state.
        cand_E = calc_energy(cand_state, single_energy_vector=single_energy_vector, correl_energy_matrix=correl_energy_matrix)

        # Update the best found state and energy if the proposed state is better.
        if cand_E < best_E:
            best_state = cand_state
            best_E = cand_E

        # Calculate the energy difference between the proposed and current state.
        diff = cand_E - curr_E
        # Calculate the current temperature based on the cooling schedule.
        curr_temp = temp / (i + 1)
        # Calculate the Metropolis criterion.
        crit = np.exp(-diff / curr_temp)
        # Generate a random number for the acceptance test.
        p = rg.uniform(low=0.0, high=1.0)
        # Check if the proposed state should be accepted.
        if diff < 0 or p < crit:
            curr_state = cand_state
            curr_E = cand_E

    # Transform the best found solution back into the original problem's variable space.
    solution = best_state * np.append(np.array([0]), variables, axis=0)
    # Return the best found energy and the corresponding solution.
    return best_E, solution
