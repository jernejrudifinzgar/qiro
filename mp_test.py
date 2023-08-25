from itertools import product
import multiprocessing as mp
import numpy as np

def fn(x, y, z):
    totals = []
    for _ in range(2):
        total = 0
        for i, j, k in product(range(x), range(y), range(z)):
            total += i * i + j * j + k * k
        totals.append(total)
        print('hallo')
    return totals

if __name__ == "__main__":
    
    # my parameter that I want to check (domain of the experiment)
    xs = range(400, 401)
    ys = range(1000, 1002)
    zs = range(20, 21)

    domain = list(product(xs, ys, zs))

    print(domain)
    testpar = {"x": 1, "z": 1, "y": 2}
    print(fn(**testpar))

    with mp.Pool(len(domain)) as p:
        result = p.starmap(fn, domain)

    print(f"Final result is {result}.")

    np.save("test_result.npy", np.array(result))
    # save as a dictionary, including the parameters
    # output_dict = {par: res for par, res in zip(domain, result)}
    # and then store this with e.g., json

    loaded = np.load("test_result.npy")
    print(loaded)
    # other options include json, pickle (pkl), ...