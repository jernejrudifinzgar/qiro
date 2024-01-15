import os
import time 

def get_change_time(ns, ps, runs, version):
    my_path = os.path.dirname(__file__)
    my_path = os.path.dirname(my_path)
    for n in ns:
        for run in runs:
            for p in ps:
                file_path = my_path + f"/data/results_run_{run}_n_{n}_p_{p}_wo_recalc_version_{version}.pkl"
                if os.path.exists(file_path):
                    mod_time = os.path.getmtime(file_path)
                    readable_time = time.ctime(mod_time)

                    print(f"Run {run}, p {p}: {readable_time}")



if __name__ == '__main__':
    ns = [50]
    ps = [1, 2, 3]
    runs = list(range(10))
    version = 1

    get_change_time(ns, ps, runs, version)
