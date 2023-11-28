import networkx as nx
import pickle
import os
import matplotlib.pyplot as plt
import matplotlib
from IPython.display import set_matplotlib_formats
__plot_height = 9.119
matplotlib.rcParams['figure.figsize'] = (1.718*__plot_height, __plot_height)
set_matplotlib_formats('svg')
import numpy as np

ns = [60, 80]#, 100, 120, 140, 160, 180, 200]
ps= [1, 2]#, 3]
num_runs = 1

colors=['tab:blue', 'tab:orange', 'tab:green']#, 'tab:red', 'tab:purple', 'tab:pink', 'tab:brown', 'tab:grey', 'tab:olive']

p_times = [[], [], []]
cuts_qtensor_list = []
for n in ns:
    for p in ps:
        for run in range(num_runs):
            file = open (f"results_test_run_{run}_n_{n}_p_{p}.txt")
            print(run, p, n)
            try:
                file = open (f"results_test_run_{run}_n_{n}_p_{p}.txt")
                lines = file.readlines()

                if p==1:
                    for line in lines:
                        if line.find("Calculated number of cuts with analytic method:") != -1:
                            line_analytic = line
                            split_word = 'method:: '
                            res = line_analytic.split(split_word, 1)
                            cuts_analytic = res[1]

                time_in_hours = lines[3]
                split_word = 'Required time in hours for RQAOA: '
                res = time_in_hours.split(split_word, 1)
                time_in_hours = res[1]
                print(time_in_hours)
                cuts_qtensor = lines[4]
                split_word = "networks: "
                res = cuts_qtensor.split(split_word, 1)
                cuts_qtensor = res[1]
                cuts_qtensor_list.append(cuts_qtensor)
                p_times[p-1].append(time_in_hours)

                file.close()

            except:
                print('file not available')
print(p_times)
width = 0.25
multiplier = 0
x = np.arange(len(ns))

ps_plot = (1, 2, 3)
p_times_dict = {'p=1': p_times[0], 
                'p=2': p_times[1]
                }
    
fig = plt.figure()
for p, time in p_times_dict.items():
    offset = width * multiplier
    rects = plt.bar(x + offset, time, width, label=p)
    plt.bar_label(rects, padding=3)
    multiplier += 1

plt.ylim(bottom=0)

plt.show()


