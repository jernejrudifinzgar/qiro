import numpy as np
import copy
from classical_benchmarks.monte_carlo_helpers import create_rnd_assignment, calc_energy, flip_single_spin

def Execute(problem_matrix, variables, num_cycles, num_per_sweep, num_swaps, num_replicas, T_min, T_max, random_seed):
    global rg
    # Initialize a random number generator with the provided seed.
    rg = np.random.default_rng(random_seed)
    
    # Create list with random assignments for each replica.
    assignment_list = []
    for num in range(num_replicas):
        assignment_list.append(np.sign(create_rnd_assignment(variable_list=variables, rg=rg)))

    # Transform the correlation matrix into a form such that the overall energy can be calculated with matrix products via the assignment-sign vector.
    # This involves splitting the correlation matrix into the diagonal (single-point correlations) and off-diagonal terms (set the diagonal to zero).
    single_energy_vector = copy.deepcopy(problem_matrix.diagonal())
    correl_energy_matrix = copy.deepcopy(problem_matrix)
    np.fill_diagonal(correl_energy_matrix, 0)

    # Initialize lists to store the current state, energy, and temperature for each replica.
    curr_state_list = []
    curr_E_list = []
    T_list = []
    # Initialize the acceptance vector to keep track of the number of successful replica exchanges at each temperature.
    acceptance_vector = np.zeros(0, dtype=int)

    # Calculate the temperature difference between replicas.
    T_diff = (T_max - T_min) / (num_replicas - 1)
    # Initialize the replicas.
    for num in range(num_replicas):
        T_list.append(T_min + num*T_diff)
        curr_state_list.append(assignment_list[num])
        curr_E_list.append(calc_energy(curr_state_list[num], single_energy_vector=single_energy_vector, correl_energy_matrix=correl_energy_matrix))
        if num != num_replicas - 1:
            acceptance_vector = np.append(acceptance_vector, 0)
        # Initialize best energy and state.
        if num == 0 or curr_E_list[num] < best_E:
            best_E = curr_E_list[num]
            best_state = curr_state_list[num]

    # Main loop for the simulated annealing algorithm.
    for cycle in range(num_cycles):
        # Loop over all replicas.
        for replica_num in range(num_replicas):
            curr_state = curr_state_list[replica_num]
            curr_E = curr_E_list[replica_num]
            curr_temp = T_list[replica_num]
            # Perform Monte Carlo sweeps.
            for i in range(num_per_sweep):
                cand_state = copy.deepcopy(curr_state)
                cand_state = flip_single_spin(assignment=cand_state, rg=rg)
                cand_E = calc_energy(cand_state, single_energy_vector=single_energy_vector, correl_energy_matrix=correl_energy_matrix)
                # Update best energy and state if we find a lower energy configuration.
                if cand_E < best_E:
                    best_E = copy.deepcopy(cand_E)
                    best_state = copy.deepcopy(cand_state)
                # Metropolis criterion.
                diff = cand_E - curr_E
                crit = np.exp(-diff / curr_temp)
                p = rg.uniform(low=0.0, high=1.0)
                if diff < 0 or p < crit:
                    curr_state = copy.deepcopy(cand_state)
                    curr_E = cand_E
            # Update the current state and energy of the replica.
            curr_state_list[replica_num] = curr_state
            curr_E_list[replica_num] = curr_E

        # Attempt to swap configurations between replicas to ensure proper sampling.
        for swap_index in range(num_swaps):
            diff = (curr_E_list[swap_index] - curr_E_list[swap_index + 1]) * (1/T_list[swap_index] - 1/T_list[swap_index + 1])
            crit = np.exp(diff)
            p = rg.uniform(low=0.0, high=1.0)
            if p < crit:
                # Swap the configurations and energies of the adjacent replicas.
                curr_state_list[swap_index], curr_state_list[swap_index + 1] = curr_state_list[swap_index + 1], curr_state_list[swap_index]
                curr_E_list[swap_index], curr_E_list[swap_index + 1] = curr_E_list[swap_index + 1], curr_E_list[swap_index]
                # Increment the acceptance counter for this temperature.
                acceptance_vector[swap_index] += 1

    # Transform the best solution found into the original problem's variable space.
    best_solution = best_state * np.append(np.array([0]), variables, axis=0)
    # Return the best energy, best solution, and acceptance rates.
    return best_E, best_solution, acceptance_vector / num_cycles
