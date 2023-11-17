import pickle
import networkx as nx

import matplotlib.pyplot as plt
import matplotlib
from IPython.display import set_matplotlib_formats
__plot_height = 8.919
matplotlib.rcParams['figure.figsize'] = (1.718*__plot_height, __plot_height)
set_matplotlib_formats('svg')


"""Structure of correlation file:
[{'graph': networkx graph, 'analytic_single_p': {'energy': energy, 'correlations': correlations_dictionary}, 'random_init': {'p=1': {'energy': energy, 'correlations': correlations_dictionary}, 'p=2':...}}, {'graph':...}...]"""

if __name__ == '__main__':

    version = 3
    reg = 3
    n = 100
    p = 2
    instance = 0

    
    

    #Get number of different instances:
    num_instances = len(data_list)
    print('Number of instances:', num_instances)

    #Get infos of an example instance:
    #Graph:
    example_graph = data_list[instance]['graph']
    #Energy for analytic p=1 grid search: 
    E_single = data_list[instance]['analytic_single_p']['energy']
    #Correlations dictionary for analytic p=1 grid search: 
    Correlations_single = data_list[instance]['analytic_single_p']['correlations']
    #Energy for Qtensor optimization with random initialization for p=p:
    E_qtensor_p = data_list[instance]['random_init'][f'p={p}']['energy']
    #Correlations dictionary for Qtensor optimization with random initialization for p=p:
    Correlations_qtensor_p = data_list[instance]['random_init'][f'p={p}']['correlations']
    
    """
    print(example_graph)
    print(E_single)
    print(Correlations_single)
    print(E_qtensor_p)
    print(Correlations_qtensor_p)
    """
