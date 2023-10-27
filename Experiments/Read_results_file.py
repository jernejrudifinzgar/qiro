import networkx as nx
import pickle 
import matplotlib.pyplot as plt

if __name__ == '__main__':

    imported_dict = pickle.load(open('results_run_1_reg_3_n_20_numCases_8.pkl', 'rb'))

    print(imported_dict)

    fig = plt.figure(figsize=(15,8))
    #ax=fig.add_subplot(111)

    numCases = 8
    p_single=[1]
    p_qtensor=[1,2,3]
    cmap = plt.cm.get_cmap('hsv', numCases)

    for i in range(numCases):
        ax=fig.add_subplot(2,4, i+1)
        size_indep_set_min_greedy=[imported_dict['reg_3_n_20_numCases_8'][i]['size_indep_set_min_greedy']]
        size_indep_set_single=[imported_dict['reg_3_n_20_numCases_8'][i]['output_single_p'][0]]
        size_indep_set_qtensor = []
        for p in p_qtensor:
            size_indep_set_qtensor.append(imported_dict['reg_3_n_20_numCases_8'][i]['output_qtensor'][f'p={p}'][0])

        ax.plot(p_single, size_indep_set_min_greedy, 'o', c=cmap(i), label=f'Min Greedy')
        ax.plot(p_single, size_indep_set_single, 'x', c=cmap(i), label=f'Single', markersize= 15)
        ax.plot(p_qtensor, size_indep_set_qtensor, '-.', c=cmap(i), label='Qtensor', markersize=30)
        plt.legend()
    
    fig.suptitle(f'Independent set sizes for n = {20}')
    plt.show()



