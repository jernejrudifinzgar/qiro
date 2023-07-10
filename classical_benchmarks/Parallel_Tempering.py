import numpy as np
import copy

def create_rnd_assignment(variable_list):
    """The zero'th index is introduced due to the form of the correlation matrix and doesn't correspond to a
    physical variable."""
    global rg
    assignment = np.array([0])
    for num in variable_list:
        sign = rg.choice((1, -1))
        assignment = np.append(assignment, np.array([sign * num]), axis=0)
    return assignment

def calc_energy(assignment, single_energy_vector, correl_energy_matrix):
    E_calc = np.sign(assignment).T @ correl_energy_matrix @ np.sign(assignment) + single_energy_vector.T @ np.sign(assignment)
    return E_calc

def flip_single_spin(assignment):
    global rg
    flip_var = rg.choice(range(1, len(assignment)))
    assignment[flip_var] = assignment[flip_var] * (-1)
    return assignment

def Execute(problem_matrix, variables, num_cycles, num_per_sweep, num_swaps, num_replicas, T_min, T_max, random_seed):
    global rg
    rg = np.random.default_rng(random_seed)
    # Create list with random assigments
    assignment_list = []
    for num in range(num_replicas):
        assignment_list.append(np.sign(create_rnd_assignment(variable_list=variables)))

    # Transform the correlation matrix into a form, such that the overall energy can be calculated with matrix products via the assignment-sign vector
    # This means the correlation matrix gets splitted into the diagonal (single-point correl.) and off diagonal terms (set the diagonal to zero)
    single_energy_vector = copy.deepcopy(problem_matrix.diagonal())
    correl_energy_matrix = copy.deepcopy(problem_matrix)
    np.fill_diagonal(correl_energy_matrix, 0)

    curr_state_list = []
    curr_E_list = []
    T_list = []
    # The acceptance vector keeps track of the number of successful replica exchanges at each temperature
    acceptance_vector = np.zeros(0, dtype=int)

    T_diff = (T_max - T_min) / (num_replicas - 1)
    for num in range(num_replicas):
        T_list.append(T_min + num*T_diff)
        curr_state_list.append(assignment_list[num])
        curr_E_list.append(calc_energy(curr_state_list[num], single_energy_vector=single_energy_vector, correl_energy_matrix=correl_energy_matrix))
        if num != num_replicas - 1:
            acceptance_vector = np.append(acceptance_vector, 0)
        if num == 0:
            best_E = curr_E_list[num]
            best_state = curr_state_list[num]
        elif curr_E_list[num] < best_E:
            best_E = curr_E_list[num]
            best_state = curr_state_list[num]

    for cycle in range(num_cycles):
        for replica_num in range(num_replicas):
            curr_state = curr_state_list[replica_num]
            curr_E = curr_E_list[replica_num]
            curr_temp = T_list[replica_num]
            for i in range(num_per_sweep):
                cand_state = copy.deepcopy(curr_state)
                cand_state = flip_single_spin(assignment=cand_state)
                cand_E = calc_energy(cand_state, single_energy_vector=single_energy_vector, correl_energy_matrix=correl_energy_matrix)
                # update our best_energy in case we reduced it
                if cand_E < best_E:
                    best_E = copy.deepcopy(cand_E)
                    best_state = copy.deepcopy(cand_state)

                diff = cand_E - curr_E
                crit = np.exp(-diff / curr_temp)
                p = np.random.uniform(low=0.0, high=1.0)
                # check if we should update state & energy
                if diff < 0 or p < crit:
                    curr_state = copy.deepcopy(cand_state)
                    curr_E = calc_energy(curr_state, single_energy_vector=single_energy_vector, correl_energy_matrix=correl_energy_matrix)

            curr_state_list[replica_num] = curr_state
            curr_E_list[replica_num] = curr_E

        for swap_index in range(num_swaps):
            diff = (curr_E_list[swap_index] - curr_E_list[swap_index + 1]) * (1/T_list[swap_index] - 1/T_list[swap_index + 1])
            crit = np.exp(diff)
            p = np.random.uniform(low=0.0, high=1.0)
            # check if we should keep the new state & energy and update them
            if p < crit:
                curr_state_list[swap_index], curr_state_list[swap_index + 1] = curr_state_list[swap_index + 1], curr_state_list[swap_index]
                curr_E_list[swap_index], curr_E_list[swap_index + 1] = curr_E_list[swap_index + 1], curr_E_list[swap_index]
                acceptance_vector[swap_index] += 1

    # Transform solution
    best_solution = best_state * np.append(np.array([0]), variables, axis=0)
    return best_E, best_solution, acceptance_vector/num_cycles
