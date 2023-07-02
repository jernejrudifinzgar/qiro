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

def Execute(problem_matrix, variables, num_iterations, temp, random_seed):
    global rg
    rg = np.random.default_rng(random_seed)
    # Create random assignment
    assignment = np.sign(create_rnd_assignment(variable_list=variables))

    # Transform the correlation matrix into a form, so that the overall energy can be calculated with matrix products via the assignment-sign vector
    # This means the correlation matrix gets splitted into the diagonal (single-point correl.) and off diagonal terms (set the diagonal to zero)
    single_energy_vector = copy.deepcopy(problem_matrix.diagonal())
    correl_energy_matrix = copy.deepcopy(problem_matrix)
    np.fill_diagonal(correl_energy_matrix, 0)

    best_state = assignment
    best_E = calc_energy(best_state, single_energy_vector=single_energy_vector, correl_energy_matrix=correl_energy_matrix)

    curr_state = copy.deepcopy(best_state)
    curr_E = copy.deepcopy(best_E)

    for i in range(num_iterations):
        cand_state = copy.deepcopy(curr_state)
        cand_state = flip_single_spin(assignment=cand_state)
        cand_E = calc_energy(cand_state, single_energy_vector=single_energy_vector, correl_energy_matrix=correl_energy_matrix)

        # update our best_energy in case we reduced it
        if cand_E < best_E:
            best_state = cand_state
            best_E = cand_E

        # calculate the energy difference:
        diff = cand_E - curr_E

        curr_temp = temp / (i + 1)

        # calculate energy exp term
        crit = np.exp(-diff / curr_temp)
        p = rg.uniform(low=0.0, high=1.0)
        # check if we should keep the new state & energy and update them
        if diff < 0 or p < crit:
            curr_state = copy.deepcopy(cand_state)
            curr_E = calc_energy(curr_state, single_energy_vector=single_energy_vector, correl_energy_matrix=correl_energy_matrix)

    # Transform solution
    solution = best_state * np.append(np.array([0]), variables, axis=0)
    return best_E, solution