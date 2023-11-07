import networkx as nx
import pickle 
import matplotlib.pyplot as plt

if __name__ == '__main__':

    n=60
    imported_dict = pickle.load(open(f'results_run_1_reg_3_n_{n}_p_2_numCases_8.pkl', 'rb'))

    print(imported_dict)

    for i in range(8):
        graph=imported_dict[f'reg_3_n_{n}_p_2_numCases_8'][i]['graph']

        print(graph.edges())

"""     
    
    fig = plt.figure(figsize=(15,8))
    #ax=fig.add_subplot(111)

    numCases = 8
    p_single=[1]
    p_qtensor=[1,2]#,3]
    cmap = plt.cm.get_cmap('hsv', numCases)

    for i in range(numCases):
        ax=fig.add_subplot(2,4, i+1)
        size_indep_set_min_greedy=[imported_dict[f'reg_3_n_{n}_p_2_numCases_8'][i]['size_indep_set_min_greedy']]
        size_indep_set_single=[imported_dict[f'reg_3_n_{n}_p_2_numCases_8'][i]['output_single_p'][0]]
        size_indep_set_qtensor = []
        for p in p_qtensor:

            imported_dict_p = pickle.load(open(f'results_run_1_reg_3_n_{60}_p_{p}_numCases_8.pkl', 'rb'))
            size_indep_set_qtensor.append(imported_dict_p[f'reg_3_n_{60}_p_{p}_numCases_8'][i]['output_qtensor'][f'p={p}'][0])

        ax.plot(p_single, size_indep_set_min_greedy, 'o', c=cmap(i), label=f'Min Greedy')
        ax.plot(p_single, size_indep_set_single, 'x', c=cmap(i), label=f'Single', markersize= 15)
        ax.plot(p_qtensor, size_indep_set_qtensor, '-.', c=cmap(i), label='Qtensor', markersize=30)
        plt.legend()
    
    fig.suptitle(f'Independent set sizes for n = {n}')
    fig.savefig(f'results_run_1_reg_3_n_{n}_numCases_8.png')
    plt.show() """



