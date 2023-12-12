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

ns = [60, 80, 100, 120, 140, 160, 180, 200]
ps= [1, 2, 3]
runs = [1]

colors=['tab:blue', 'tab:orange', 'tab:green']#, 'tab:red', 'tab:purple', 'tab:pink', 'tab:brown', 'tab:grey', 'tab:olive']

p_times = [[], [], []]
cuts_qtensor_list = []

my_path = os.path.dirname(__file__)
my_path = os.path.dirname(my_path)
for n in ns:
    for p in ps:
        for run in runs:
            try:
                file = open (my_path + f"/data/results_test_run_{run}_n_{n}_p_{p}.txt")
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
                cuts_qtensor = lines[4]
                split_word = "networks: "
                res = cuts_qtensor.split(split_word, 1)
                cuts_qtensor = res[1]
                cuts_qtensor_list.append(cuts_qtensor)
                p_times[p-1].append(time_in_hours)

                file.close()

            except:
                print('file not available')
                p_times[p-1].append('0')
                cuts_qtensor_list.append('0')

print(p_times)
width = 0.25
multiplier = 0
x = np.arange(len(ns))

p_times_dict = {'p=1': p_times[0], 
                'p=2': p_times[1],
                'p=3': p_times[2]}
    
#fig, ax = plt.subplots(layout='constrained')

for p, time in p_times_dict.items():
    offset = width * multiplier
    rects = plt.bar(x + offset, time, width, label=p)
    plt.bar_label(rects, padding=3)
    multiplier += 1

#plt.xticks([r + ])

#ax.set_ylim(bottom=0)
plt.show()


