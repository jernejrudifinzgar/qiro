import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

from problem_generation.Generate_MIS import *
from expval_calculation.StateVecQAOA import StateVecQAOAExpectationValues
from expval_calculation.SingleLayerQAOA import SingleLayerQAOAExpectationValues
from QIRO_MIS import QIRO_MIS
from copy import deepcopy


def mappable(idx, partition):
    reps = 1
    ps = ["single", 1, 2]
    n = 10
    densities = [k / (n - 1) for k in [3, 6]]
    data = []
    for p in ps:
        p_data = []
        for density in densities:
            density_data = []
            graph = nx.erdos_renyi_graph(n, density, seed=idx)
            for repseed in range(reps):
                mis_problem = MIS(deepcopy(graph), alpha=1.1, seed=repseed + reps * partition)
                mis_problem.graph_to_matrix()
                if p == "single":
                    mis_expval = SingleLayerQAOAExpectationValues(mis_problem)
                else:
                    mis_expval = StateVecQAOAExpectationValues(mis_problem, p, num_opts=10, num_opt_steps=200)
                
                qmis = QIRO_MIS(expectation_values_input=mis_expval, nc=1)
                qmis.execute()
                qmis.problem.graph = deepcopy(graph)
                mis_size = qmis.problem.evaluate_solution(qmis.solution)[1]
                density_data.append(mis_size)
            p_data.append(density_data)
        data.append(p_data)
    return data

if __name__ == "__main__":
    import multiprocessing as mp
    from itertools import product

    num_partitions = 3
    num_parallel = 50

    map_params = list(product(range(num_parallel), range(num_partitions)))

    with mp.Pool(len(map_params)) as pool:
        results = pool.starmap(mappable, map_params)

    np.save("experiments/morelayers.npy", np.array(results).astype(int))
                

    



