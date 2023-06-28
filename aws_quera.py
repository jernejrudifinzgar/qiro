import os
import numpy as np

from braket.ahs.atom_arrangement import AtomArrangement
from braket.ahs.hamiltonian import Hamiltonian
from braket.ahs.analog_hamiltonian_simulation import AnalogHamiltonianSimulation
from braket.devices import LocalSimulator
from braket.aws import AwsDevice
from braket.timings.time_series import TimeSeries
from braket.ahs.driving_field import DrivingField
from braket.jobs.metrics import log_metric
from braket.jobs import save_job_checkpoint

from itertools import combinations, chain
import sys

from scipy.spatial import distance_matrix
from scipy.signal import butter, lfilter
from scipy.interpolate import interp1d

import networkx as nx
import json
from itertools import product
from copy import copy


# noinspection PyTypeChecker
def run_job(hyperparams=None):
    is_local = hyperparams is not None
    print("Job started!!!!!")

    # load hyperparameters
    if not is_local:
        hp_file = os.environ["AMZN_BRAKET_HP_FILE"]
        input_dir = os.environ["AMZN_BRAKET_INPUT_DIR"]
        with open(hp_file, "r") as f:
            hyperparams = json.load(f)
    else:
        input_dir = ""
    n_shots = int(hyperparams["n_shots"])
    graph_parameter = hyperparams["graph_parameter"]
    percent = float(hyperparams["percentile"])
    lattice_scale = float(hyperparams["lattice_scale"])
    graph_kind, graph_idx = graph_parameter[0], int(graph_parameter[1:])
    alpha = float(hyperparams["alpha"]) if "alpha" in hyperparams else 5
    postselection = bool(int(hyperparams["postselection"])) if "postselection" in hyperparams.keys() else False
    t_total = np.round(float(hyperparams['t_total']) if "t_total" in hyperparams.keys() else 4e-6, 9)
    nodes_to_remove = int(hyperparams["nodes_to_remove"]) if "nodes_to_remove" in hyperparams.keys() else 0
    remaining_nodes = int(hyperparams["remaining_nodes"]) if "remaining_nodes" in hyperparams.keys() else 19
    # vanilla = bool(int(hyperparams["vanilla"])) if "vanilla" in hyperparams.keys() else False

    parameters = {"omega_max": 1.58e7, "tau": 0.1, "delta_initial": 3e7, "delta_final": 6e7}

    # Use the device declared in the job script
    if is_local:
        device = None
    else:
        device = AwsDevice(os.environ["AMZN_BRAKET_DEVICE_ARN"])
        # device = LocalSimulator('braket_ahs')
    if graph_kind == 't':
        register, graph = prepare_test_register(n_cells=graph_idx, parallel=not is_local, a=4.5e-6 * lattice_scale)
    elif graph_kind == 'h':
        register, graph = prepare_register_hard_graph(
            os.path.join(input_dir, "hard-graphs-13-14/", f"hg{graph_idx}.txt"), scale=lattice_scale)
    elif graph_kind == 'g':
        register, graph = prepare_grid_register(which=graph_idx, parallel=False, a=4.5e-6 * lattice_scale)
    elif graph_kind == "e":
        register, graph = prepare_register_hard_graph(
            os.path.join(input_dir, "easy-graphs/", f"eg{graph_idx}.txt"), scale=lattice_scale)
    elif graph_kind == "u":
        register, graph = prepare_register_hard_graph(
            os.path.join(input_dir, "uniform-graphs/", f"ug{graph_idx}.txt"), scale=lattice_scale)
    else:
        raise Exception(f"Graph kind {graph_kind} not recognized.")
    print("Register generated!")
    if is_local:
        dir_path = ""
    else:
        dir_path = os.environ['AMZN_BRAKET_JOB_RESULTS_DIR']

    vertex_coordinates = np.transpose([register.coordinate_list(i) for i in range(2)])[nodes_to_remove:]

    # check for checkpoints:
    checkpoint_data = None
    if not is_local:
        checkpoints_path = os.environ["AMZN_BRAKET_CHECKPOINT_DIR"]
        checkpoint_file = ''
        # print(checkpoints_path)
        # print(os.listdir(checkpoints_path))
        for item in os.listdir(checkpoints_path):
            if item.endswith('checkpoint.json'):
                print('Checkpoint file found')
                checkpoint_file = os.path.join(checkpoints_path, item)
                break
        is_checkpoint = checkpoint_file != ''
        if is_checkpoint:
            with open(os.path.join(checkpoints_path, checkpoint_file), 'r') as f:
                checkpoint_data = json.load(f)

            print(checkpoint_data)
            print("THIS WORKED!!!!")


    if checkpoint_data is not None:
        print("Loading checkpoint data:")
        checkpoint_data_dict = checkpoint_data['dataDictionary']
        print(f"Found {len(checkpoint_data_dict)} removed variables in the checkpoint data. This is the checkpoint data:")
        print(checkpoint_data_dict)
        # sort the nodes such that the highest index is removed first, and the deletion of nodes does not affect the indices of the nodes to be removed
        nodes_to_be_removed = sorted([int(node) for node in checkpoint_data_dict.keys()])[::-1]
        for node in nodes_to_be_removed:
            graph.remove_node(node)
            vertex_coordinates = np.delete(vertex_coordinates, node, axis=0)
    
        print("checkpoint data loaded.. this worked!!!")
    else: 
        checkpoint_data_dict = {}

    for i in range(nodes_to_remove):
        graph.remove_node(i)

    # save graph and vertex positions
    if not is_local:
        nx.write_adjlist(graph, os.path.join(dir_path, "graph.adjlist"))
        np.savetxt(fname=os.path.join(dir_path, "atom_positions.txt"),
                   X=vertex_coordinates)
        
    def _get_dummy_bitstrings(g, percentile=1.):
        """generates and returns random bitstrings."""
        # prepares the Hamiltonian, given the parameters.
        measured_bitstrings = (np.random.randint(2, size=(n_shots, g.number_of_nodes())) - 0.5) * 2

        # returns the MIS cost and the bitstrings. We make sure that the number of shots considered for the optimizer is
        # the same.
        print(measured_bitstrings.shape)
        print(g.number_of_nodes())
        energies = [get_mis_cost(g, bs, alpha=alpha) for bs in measured_bitstrings]
        included = np.argsort(energies)[:int(percentile * len(energies))]
        # return bitstrings transformed into ising values
        return ((1 - measured_bitstrings[included]) - 0.5) * 2 


    def _get_bitstrings(graph, vertices, percentile=1):
        """returns the measured bitsrings."""
        # prepares the Hamiltonian, given the parameters.
        H_obj = prepare_drive(t_total, parameters)
        register = get_register_from_vertices(vertices)
        # defines the ahs_program object which is sent o the device.
        ahs_program = AnalogHamiltonianSimulation(hamiltonian=H_obj, register=register)
        ns = int(n_shots)
        result = device.run(ahs_program, shots=ns).result()
        # extracts bitstrings (as a list of lists)
        if postselection:
            measured_bitstrings = np.array([np.array(s.post_sequence) for s in result.measurements
                                           if np.all(s.pre_sequence)])
        else:
            measured_bitstrings = np.array([np.array(s.post_sequence) for s in result.measurements])
        # returns the MIS cost and the bitstrings. We make sure that the number of shots considered for the optimizer is
        # the same.
        measured_bitstrings = ((1 - measured_bitstrings) - 0.5) * 2 
        energies = [get_mis_cost(graph, bs, alpha=alpha) for bs in measured_bitstrings]
        included = np.argsort(energies)[:int(percentile * len(energies))]
        # return bitstrings transformed into ising values
        return measured_bitstrings[included]

    def _apply_single_reduction(g, node_index, applied_value):
        assig = {}
        if applied_value == 1:
            ns = copy(g.neighbors(node_index))
            for node in ns:
                g.remove_node(node)
                assig.update({node: -1})

        g.remove_node(node_index)
        assig.update({node_index: applied_value})

        return g, assig

    def _apply_corr_reduction(g, node_indices, correlation):
        assig = {}
        # check the if the nodes are connected by an edge:
        if not g.has_edge(*node_indices):
            return g, assig
        else:
            if correlation == 1:
                for node in node_indices:
                    # TODO: replace this to check that nodes are neighbors.
                    # TODO: Same for RQAOA.
                    g.remove_node(node)
                    assig.update({node: -1})
            else:
                mutual_neighbors = set(g.neighbors(node_indices[0])) & set(g.neighbors(node_indices[1]))
                for mn in mutual_neighbors:
                    g.remove_node(mn)
                    assig.update({mn: -1})
            return g, assig

    def _transform_index(idx, g):
        return sorted(g.nodes)[idx]
    
    def _prune_graph(g):
        """Prune the graphs -- if there's standalone nodes we remove them. Check for connected components and brute force them."""
        # we brute-force all subgraphs which are smaller than the brute-force threshold:
        prune_assig = {}

        for subnodes in nx.connected_components(g):
            if len(subnodes) < remaining_nodes:
                subgraph = g.subgraph(subnodes)
                _, miss = find_mis(subgraph)
                mis = miss[0]
                prune_assig.update({n: 1 if n in mis else -1 for n in subgraph.nodes})
    
        for node in prune_assig.keys():
            g.remove_node(node)
        return g, prune_assig
    
    assigned_values = {}
    # if we have checkpoint data, we need to update the assigned values.
    assigned_values.update({int(vert): int(assig) for vert, assig in checkpoint_data_dict.items()})
    
    iteration_counter = 0
    while graph.number_of_nodes() >= remaining_nodes:
        new_assignments = {}
        repetition_counter = 0
        if not is_local:
            bitstrings = _get_bitstrings(graph, vertex_coordinates, percentile=percent)
        else:
            bitstrings = _get_dummy_bitstrings(graph, percentile=percent)

        single_corrs = np.mean(bitstrings, axis=0)
        corrs = np.tril(np.corrcoef(bitstrings.T), k=-1)
        while len(new_assignments) == 0:
            if repetition_counter > 0:
                print(f"Repeating iteration, as no new assignments were found. Repetition number: {repetition_counter}")

            # take the single_corrs and double corrs and create a list which is sorted by the absolute value of the
            # correlations. We take the repetition_counter element of the list and apply the corresponding reduction.
            
            largest_single_idx = np.argmax(np.abs(single_corrs))
            largest_double_idx = tuple(list(np.unravel_index(np.argmax(np.abs(corrs)), corrs.shape)))
            nodes_in = sorted(graph.nodes)
            if np.abs(single_corrs[largest_single_idx]) >= np.abs(corrs[largest_double_idx]):
                sign = np.sign(single_corrs[largest_single_idx]).astype(int)
                node = _transform_index(largest_single_idx, graph)
                graph, new_assignments = _apply_single_reduction(graph, node, sign)
                # we set the magnitude of the correlation to 0 if we apply the reduction.
                # this is to avoid applying the same reduction twice, if we failed to remove a node in the previous iteration.
                single_corrs[largest_single_idx] = 0.

            else: 
                sign = np.sign(corrs[tuple(largest_double_idx)])
                nodes = [_transform_index(idx, graph) for idx in largest_double_idx]
                graph, new_assignments = _apply_corr_reduction(graph, nodes, sign)
                # we set the magnitude of the correlation to 0 if we apply the reduction.
                corrs[largest_double_idx] = 0. 
            print(new_assignments)
            

            # prune the graph to brute-force all connected components which are smaller than the brute-force threshold
            graph, prune_assig = _prune_graph(graph)

            # update the assignments
            new_assignments.update(prune_assig)
            assigned_values.update(new_assignments)
            print(f"Pruned: {prune_assig}")

            indices_to_remove = [nodes_in.index(node) for node in new_assignments.keys()]
            # remove the coordinates of the nodes which were removed
            vertex_coordinates = np.delete(vertex_coordinates, indices_to_remove, 0)
            
            repetition_counter += 1
        # log the number of variables removed
        log_metric(metric_name="Variables removed", value=len(assigned_values), iteration_number=iteration_counter + 1)
        # save the checkpoint data; ensure that the types are correct such that json does not complain
        save_job_checkpoint(checkpoint_data={int(k): int(v) for k, v in assigned_values.items()}, checkpoint_file_suffix="checkpoint")
        iteration_counter += 1
    print("Job completed!")

    if not is_local:
        with open(os.path.join(dir_path, "hyperparameters.json"), 'w') as fhyper:
            fhyper.write(json.dumps(hyperparams))
        with open(os.path.join(dir_path, "assignments.json"), 'w') as fassig:
            output_assignments = {int(k): int(v) for k, v in assigned_values.items()}
            fassig.write(json.dumps(output_assignments))
        
        nx.write_adjlist(graph, os.path.join(dir_path, "graph_out.adjlist"))
    else: 
        return assigned_values, graph
    
  

def get_register_from_vertices(vertices):
    vertices = vertices.round(7)
    reg = AtomArrangement()
    for v in vertices:
        reg.add(v)
    return reg

def keep_pbound_keys(parameters, pbounds):
    """Keeps only the keys of parameters that are also in pbounds."""
    return {k: v for k, v in parameters.items() if k in pbounds}


def prepare_register_hard_graph(fpath, scale=1):
    """Takes a filepath to a txt file containing the positions of vertices and returns the register and the graph
    object. scale=1 corresponds to a lattice constant of 4.5."""
    vs = np.loadtxt(fpath) * scale
    vs = vs.round(7)
    reg = AtomArrangement()
    for v in vs:
        reg.add(v)
    return reg, vertices_to_graph(vs, 1.7 * scale * 4.5e-6)


def add_save_optimizer_data(res_dict, params, target_value, filepath, is_local):
    """Adds optimizer data to the dictionary saving the results. Also writes it to filepath."""
    pos = len(res_dict)
    new_entry = {'params': params, 'target': target_value}
    res_dict[pos] = new_entry
    # print out for progress tracking!
    print(f"Iteration {pos + 1} completed!")
    if not is_local:
        with open(filepath, 'w') as f_resdict:
            f_resdict.write(json.dumps(res_dict))
        # for logging
        log_metric(metric_name="Target", value=target_value, iteration_number=pos + 1)
        # for checkpointing
        save_job_checkpoint(checkpoint_data=res_dict,
                            checkpoint_file_suffix="checkpoint")

    return res_dict


def log_iteration(iteration_number):
    print(f"Iteration {iteration_number} completed.")
    iteration_number += 1
    return iteration_number


def suggest_random_params(pbounds):
    """Suggests random params."""
    return {k: np.random.uniform(*v) for k, v in pbounds.items()}


def get_mis_cost(graph, solution, alpha=5):
    """Calculate the mis cost for a give solution for a given graph."""
    solution = np.array(solution)
    violations = 0
    excited_pos = np.where(solution == 1)[0]
    remaining_nodes = sorted(graph.nodes)
    for v1, v2 in graph.edges:
        i1, i2 = remaining_nodes.index(v1), remaining_nodes.index(v2)
        if solution[i1] == 1 and solution[i2] == 1:
            violations += 1
    return -len(excited_pos) + alpha * violations


def get_parameter_bounds(ndp, span=1, kind='real', scale=1., optimize_rabi=True):
    """Get the bounds for the BO."""
    omega_bounds = tuple(sorted(compute_rabi_bounds(4.5e-6 * scale)))
    pb = {
        "omega_max": omega_bounds,
        "delta_initial": (1e7, 1.25e8),
        "delta_final": (1e7, 1.25e8),
        "tau": (0.05, 0.45)
    }
    if not optimize_rabi:
        del pb["omega_max"]
        del pb['tau']

    if kind != 'fourier':
        pb.update({f"dp{nr + 1}": (
            max((nr + 1) / (ndp + 1) - span / (ndp + 1), 0),
            min((nr + 1) / (ndp + 1) + span / (ndp + 1), 1)
        ) for nr in range(ndp)})
    else:
        pb.update({f"dp{nr + 1}": (-1 / (nr + 1), 1 / (nr + 1)) for nr in range(ndp)})

    return pb


def generate_initial_params(omega_max, delta_initial, delta_final, tau, ndp, kind='real'):
    params = {
        'omega_max': omega_max,
        'delta_initial': delta_initial,
        'delta_final': delta_final,
        'tau': tau
    }
    if kind != 'fourier':
        params.update({f'dp{i}': i / (ndp + 1) for i in range(1, ndp + 1)})
    else:
        params.update({f'dp{i}': 0 for i in range(1, ndp + 1)})

    return params


def prepare_test_register(n_cells=1, parallel=False, a=4.5e-6, add_middle=True):
    """In the real case this will be replaced by a function preparing the graph arrangement."""
    # a = 4.5e-6
    register = AtomArrangement()
    if not parallel:
        n_reps = 1
    else:
        n_reps = int(np.floor((7.5e-5 / (2 * a) + 1) / (n_cells + 1)))

    for offset in product(range(n_reps), range(n_reps)):
        x_o, y_o = [o * 2 * a * (n_cells + 1) for o in offset]
        # grid
        for i in range(0, 2 * n_cells + 1, 2):
            for j in range(0, 2 * n_cells + 1, 2):
                register.add(np.array((i * a + x_o, j * a + y_o)).round(7))
        # add middle
        if add_middle:
            for i in range(1, 2 * n_cells + 1, 2):
                for j in range(1, 2 * n_cells + 1, 2):
                    register.add(np.array((i * a + x_o, j * a + y_o)).round(7))

    return register, vertices_to_graph(np.transpose([register.coordinate_list(0), register.coordinate_list(1)]),
                                       a * 1.5)


def prepare_grid_register(which=0, parallel=False, a=4.5e-6):
    # number of grid points in the x, y dimension
    mx = 4
    my = 3

    def _get_reps_and_shift(m, d=7.5e-5):
        reps = int((d + 2 * a) / (a * (m + 2)))
        # print(reps)
        shift = (d - reps * a * m) / ((reps - 1) * a)
        return reps, shift

    if not parallel:
        x_reps = y_reps = 1
        x_shift = y_shift = 0
    else:
        x_reps, x_shift = _get_reps_and_shift(mx - 1, d=7.5e-5)
        y_reps, y_shift = _get_reps_and_shift(my - 1, d=7.6e-5)

    _, poss = get_grid_graph_combs(9, 10, (mx, my), scale=a)
    if which >= len(poss):
        print("Warning; which is longer than the number of graphs generated.")
        which %= len(poss)
        print(f"Graph {which} generated.")
    pos = poss[which]

    pos_container = []

    for xrep in range(x_reps):
        for yrep in range(y_reps):
            pos_container.append(pos + np.array([xrep * (mx + x_shift - 1) * a, yrep * (my + y_shift - 1) * a]))

    total_pos = np.vstack(pos_container)

    register = AtomArrangement()
    for coord in total_pos:
        register.add(coord.round(7))

    return register, vertices_to_graph(total_pos, 1.5 * a)


def prepare_drive(total_time, params, kind='real'):
    H = Hamiltonian()

    # this function does the heavy lifting:
    if kind == 'real':
        drive = get_drive_real(total_time, params, lowpass=False)
    elif kind == "lowpass":
        drive = get_drive_real(total_time, params, lowpass=True)
    elif kind == "fourier":
        drive = get_drive_real(total_time, params)
    else:
        raise NotImplementedError("Kind of drive parametrization not implemented.")
    H += drive

    return H


def cutoff_ramp(t, total_time, tau):
    if t <= tau:
        return 0
    elif t >= total_time - tau:
        return 1
    else:
        return (t - tau) / (total_time - 2 * tau)


def get_drive_fourier(total_time, params):
    """Fourier drive."""
    Omega, phi = get_omega_phi_ts(total_time, params)
    delta_initial = round(params['delta_initial'])
    delta_final = round(params['delta_final'])
    coefs = {k: float(v) for (k, v) in params.items() if k.startswith('dp')}
    tau = params["tau"] * total_time
    interval = total_time - 2 * tau
    nts = int(total_time / 5e-8)
    times = [0] + list(np.linspace(tau, total_time - tau, nts)) + [total_time]

    def _sinusoid(t, freq_idx, magnitude):
        if t <= tau or t >= total_time - tau:
            return 0
        else:
            freq = freq_idx * np.pi / interval
            return np.sin(freq * (t - tau)) * magnitude

    # linear ramp
    vals = np.array([cutoff_ramp(t, total_time, tau) for t in times])
    for dpfi, coef in coefs.items():
        fi = int(dpfi[2:])
        vals += np.array([_sinusoid(t, fi, coef) for t in times])
    deltas = np.clip(-delta_initial + (delta_final + delta_initial) * vals, -1.25e8, 1.25e8)

    Delta_global = TimeSeries()
    for t, delta in zip(times, deltas):
        Delta_global.put(round(t, 9), round(delta))
    return DrivingField(amplitude=Omega, phase=phi, detuning=Delta_global)


def get_real_detuning_ts(params, total_time):
    """Computes the detuning schedule for a real-point schedule. No Low-Pass filtering."""
    delta_initial = round(params['delta_initial'])
    delta_final = round(params['delta_final'])
    delta_points = {k: float(v) for (k, v) in params.items() if k.startswith('dp')}
    ndp = len(delta_points)
    tau = round(params["tau"] * total_time, 9)
    # start with negative detuning
    Delta_global = TimeSeries().put(0.0, -delta_initial)
    Delta_global.put(tau, -delta_initial)
    # add intermediate points
    for i in range(1, ndp + 1):
        # correctly stretch the points in between, as they are dependent on delta_initial and delta_final
        d_val = round(-delta_initial + (delta_final - (-delta_initial)) * delta_points[f'dp{i}'])
        Delta_global.put(round(tau + (total_time - 2 * tau) * i / (ndp + 1), 9), d_val)
    Delta_global.put(round(total_time - tau, 9), delta_final)
    # add a positive delta_final
    Delta_global.put(round(total_time, 9), delta_final)  # (time [s], value [rad/s])

    return Delta_global


def get_lowpass_detuning_ts(params, total_time):
    def _butter_lowpass(cutOff, fs, order=5):
        nyq = 0.5 * fs
        normalCutoff = cutOff / nyq
        # noinspection PyTupleAssignmentBalance
        b, a = butter(order, normalCutoff, btype='low', analog=False, output='ba')
        return b, a

    def _butter_lowpass_filter(data, cutOff, fs, order=4):
        b, a = _butter_lowpass(cutOff, fs, order=order)
        ys = lfilter(b, a, data)
        return ys

    delta_initial = round(params['delta_initial'])
    delta_final = round(params['delta_final'])
    delta_points = {k: float(v) for (k, v) in params.items() if k.startswith('dp')}
    tau = round(params["tau"] * total_time, 9)
    points = [0, 0] + [delta_points[k] for k in sorted(delta_points.keys())] + [1, 1]
    interp_times = [0, tau] + list(np.linspace(tau, total_time - tau, len(delta_points) + 2)[1:-1]) + \
                   [total_time - tau, total_time]
    interp = interp1d(interp_times, points)
    nts = int(total_time / 5e-8)
    times = list(np.linspace(0, total_time, nts))

    ramp = np.array([cutoff_ramp(t, total_time, tau) for t in times])
    series_notrend = interp(times) - ramp
    y = _butter_lowpass_filter(series_notrend, nts // (len(points) + 1), nts, 3) + ramp

    Delta_global = TimeSeries()
    for t, pt in zip(times, y):
        delta = int(np.round(np.clip(-delta_initial + (delta_final - (-delta_initial)) * pt, -1.25e8, 1.25e8)))
        Delta_global.put(round(t, 9), delta)

    return Delta_global


def get_omega_phi_ts(total_time, params):
    Omega_max = (params['omega_max'] // 400) * 400  # rad / seconds. We're fixing the resolution to 400.
    tau = params['tau']  # fraction of total time

    # Rabi frequency path
    Omega = TimeSeries()
    # first linear ramp up
    Omega.put(0.0, 0.)
    Omega.put(round(tau * total_time, 9), Omega_max)
    # constant for (1 - 2 * tau) * total_time
    Omega.put(round(total_time * (1 - tau), 9), Omega_max)
    # ramp down
    Omega.put(round(total_time, 9), 0.)
    # no phase (i.e. 0 the whole time)
    phi = TimeSeries().put(0.0, 0.0).put(total_time, 0.0)  # (time [s], value [rad])
    return Omega, phi


def get_drive_real(total_time, params, lowpass=False):
    Omega, phi = get_omega_phi_ts(total_time, params)
    if not lowpass:
        Delta_global = get_real_detuning_ts(params, total_time)
    else:
        Delta_global = get_lowpass_detuning_ts(params, total_time)

    drive = DrivingField(
        amplitude=Omega,
        phase=phi,
        detuning=Delta_global
    )
    return drive


def get_grid_graph_combs(natoms_min, natoms_max, gridshape, scale=5.2e-6, return_combs=False):
    """Returns all different graphs of sizes natoms_min to natoms_max which have a unique MIS. The atoms are positioned
    in a grid of shape gridshape. The distance between atoms is scale."""
    poss = []
    graphs = []
    good_combs = []

    for natoms in range(natoms_min, natoms_max + 1):
        for comb in combinations(range(np.multiply(*gridshape)), natoms):
            indices = np.unravel_index(comb, gridshape)
            pos = np.transpose(indices) * scale
            g = vertices_to_graph(pos)
            _, miss = find_mis(g)
            if len(miss) == 1 and not np.any([nx.is_isomorphic(g, graph) for graph in graphs]):
                graphs.append(g)
                poss.append(pos)
                good_combs.append(comb)

    if return_combs:
        return good_combs
    else:
        return graphs, poss


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

def count_violations(graph, solution):
    nvio = 0
    for edge in product(solution, repeat=2):
        if graph.has_edge(*edge):
            nvio += 1
    return nvio


def find_mis(graph, maximum=True):
    """Finds a maximal independent set of a graph, and returns its bitstrings."""
    if maximum is False:
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


def vertices_to_graph(vertices, radius=7.5e-6):
    """Converts the positions of vertices into a UDG"""
    dmat = distance_matrix(vertices, vertices)
    adj = (dmat < radius).astype(int) - np.eye(len(vertices))
    # zr = np.where(~np.any(adj, axis=0))
    # adj = np.delete(np.delete(adj, zr, 1), zr, 0)
    return nx.from_numpy_array(adj)


def compute_rabi_bounds(lattice_spacing):
    """Computes the (approximate) bounds on the allowed Rabi frequencies."""
    c6 = 5.42e-24
    return min(c6 / (lattice_spacing * np.sqrt(2)) ** 6, 1.58e7), min(c6 / (lattice_spacing * 2) ** 6, 1.58e7)


def get_mis_size(graph, solution):
    print(f"Set is independent? {is_independent(graph, solution)}")
    if isinstance(solution, dict):
        return np.count_nonzero([int(i > 0) for i in solution.values()])
    else:
        return len(solution)





