import gurobipy as gp
from gurobipy import GRB
import networkx as nx
import pickle
from itertools import product
import numpy as np
import time


def cb(model, where, threshold_time=1):
    if where == GRB.Callback.MIPNODE:
        # Get model objective
        obj = model.cbGet(GRB.Callback.MIPNODE_OBJBST)

        # Has objective changed?
        if abs(obj - model._cur_obj) > 1e-2:
            # If so, update incumbent and time
            model._cur_obj = obj
            model._time = time.time()

    # Terminate if objective has not improved in thershold_time hrs
    if time.time() - model._time > threshold_time * 3600:
        model.terminate()
    # elif model.status == GRB.OPTIMAL:
    #     model._cur_obj = model.opt


def solve_max_cut(G, id=None, timeout=1, nthr=1, verbose=True, return_optimality=False):
    """Function to solve Maximum Cut problem using Gurobi"""
    # Create a new Gurobi model
    m = gp.Model("max_cut")
    m.setParam(GRB.Param.Threads, nthr)
    # Add binary variables for each node in the graph
    x = m.addVars(sorted(G.nodes()), vtype=GRB.BINARY, name="x")

    # Objective function: maximize the sum of edge weights crossing the cut
    obj = gp.quicksum(G[u][v].get('weight', 1) * (x[u] - x[v]) * (x[u] - x[v]) for u, v in G.edges())
    m.setObjective(obj, GRB.MAXIMIZE)
    m.setParam("OutputFlag", int(verbose))
    m._cur_obj = float('inf')
    m._time = time.time()   
    # Optimize the model
    callback = lambda model, where: cb(model, where, threshold_time=timeout)
    m.optimize(callback=callback)

    if m.status == GRB.OPTIMAL:
        print("Found optimal cut.")
        obj = 0
        for e1, e2 in G.edges:
            if m.x[e1] != m.x[e2]:
                obj += G[e1][e2].get("weight", 1)
        # return m
        return obj if not return_optimality else [obj, 1]
    else:
        return int(m._cur_obj) if not return_optimality else [obj, 0]
    
if __name__ == '__main__':
    list_solutions = []
    with open('100_regular_graphs_nodes_30_reg_3.pkl', 'rb') as f:
        data = pickle.load(f)
    counter = 0
    for graph in data:
        counter += 1
        solution = solve_max_cut(graph)
        print(f"Solution for graph {counter} found")
        list_solutions.append(solution)

    pickle.dump(list_solutions, open("100_regular_graphs_nodes_30_reg_3_solutions.pkl", 'wb'))
