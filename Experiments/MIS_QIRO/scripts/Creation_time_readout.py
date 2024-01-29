import os
import time 

def get_change_time(ns, ps, runs, initialization, version):
    my_path = os.path.dirname(__file__)
    my_path = os.path.dirname(my_path)
    for n in ns:
        for run in runs:
            for p in ps:
                file_path = my_path + f"/data/results_run_{run}_n_{n}_p_{p}_initialization_{initialization}_version_{version}.pkl"
                if os.path.exists(file_path):
                    mod_time = os.path.getmtime(file_path)
                    readable_time = time.ctime(mod_time)

                    if p==1 and run==0:
                        base_time = mod_time

                    required_time = mod_time-base_time
                    required_time_hours = required_time/3600
                    required_time_days = required_time_hours/24

                    print(f"Run {run}, p {p}: {readable_time}, Required time: {required_time_hours} h = {required_time_days} d")
                else:
                    print(f"Run {run}, p {p} not available")




if __name__ == '__main__':
    ns = [80]
    ps = [1, 2, 3]
    runs = list(range(20))
    version = 1
    for init in ['interpolation', 'transition_states']:
        print(init)
        get_change_time(ns, ps, runs, init, version)
