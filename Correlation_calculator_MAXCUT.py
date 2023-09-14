import Generating_Problems as Generator
import networkx as nx
from Calculating_Expectation_Values import ExpectationValues
import numpy as np


if __name__ == '__main__':
    p = 1
    reg = 3
    n = 20
    seed=666
    G = nx.random_regular_graph(reg, n, seed = seed)

    problem = Generator.MAXCUT(G)

    exp_values = ExpectationValues(problem)

    exp_value_coeff, exp_value_sign, max_exp_value = exp_values.optimize()

    print(exp_values.gamma)
    print(exp_values.beta)

    print(exp_values.max_exp_dict)

