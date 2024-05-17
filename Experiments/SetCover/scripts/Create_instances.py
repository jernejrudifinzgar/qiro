import numpy as np
import random
import copy

def find_start(x, distance):
    start = x-distance if x-distance >= 0 else 0
    return start

def find_end(x, distance, shape):
    end = x+distance if x+distance <= shape else shape
    return end

def find_neighbors(matrix, i, j, distance):
    neighbors = []
    row_start, row_end = find_start(i, distance), find_end(i, distance, matrix.shape[0])
    col_start, col_end = find_start(j, distance), find_end(j, distance, matrix.shape[1])

    for y in range(matrix.shape[0]):
        for z in range(matrix.shape[1]):
            if y >= row_start and y <= row_end:
                if z >= col_start and z <= col_end:
                    if matrix[y][z] >= 0:
                        neighbors.append(matrix[y][z]) 
    return neighbors

def create_set_cover(x=8, y=5, node_probability = 0.6, neighbor_probability = 0.3, sensor_distances = [0, 1, 2, 3], sensor_probabilities = [0.7, 0.1, 0.1, 0.1]):
    set = []
    subsets = []

    nodes = -np.ones((x, y))
    num_nodes = x * y
    counter = 0
    for i in range(x):
        for j in range(y):
            node_placement = bool(np.random.choice([0,1], p=[1-node_probability, node_probability]))
            if node_placement:
                nodes[i, j] = int(counter)
                set.append(int(counter))
                counter += 1

    set_copy = copy.deepcopy(set)

    for distance, sensor_probability in zip(sensor_distances, sensor_probabilities):
        for i in range(x):
            for j in range(y):
                sensor_placement = bool(np.random.choice([0, 1], p=[1-sensor_probability, sensor_probability]))
                if sensor_placement:
                    subset = []
                    neighbors = find_neighbors(nodes, i, j, distance)
                    for neighbor in neighbors:
                        if neighbor >= 0:
                            neighbor_placement = bool(np.random.choice([0, 1], p=[1-neighbor_probability, neighbor_probability]))
                            if neighbor_placement:
                                subset.append(int(neighbor))
                                try:
                                    set_copy.remove(neighbor)
                                except:
                                    pass
                    if len(subset)>0:
                        subsets.append(subset)

    for node in set_copy:
        subsets.append([node])
    
    return set, subsets