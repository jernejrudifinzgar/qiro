import cvxpy as cp
import networkx as nx
import numpy as np

def sdp_vectors(G):
    """
    G: A networkx.Graph instance where edges have a 'weight' attribute.
    Returns the vectors corresponding to each node after solving the SDP.
    """
    # Step 1: Formulate the SDP
    n = len(G.nodes)
    X = cp.Variable((n, n), symmetric=True)

    constraints = [X >> 0]  # X is positive semidefinite
    for i in range(n):
        constraints.append(X[i, i] == 1)

    # Objective function
    objective = cp.Constant(0)
    for u, v, data in G.edges(data=True):
        weight = data.get("weight", 1)
        objective += weight * (1 - X[u, v]) / 4

    problem = cp.Problem(cp.Maximize(objective), constraints)
    problem.solve(solver=cp.SCS)  # Use SCS for semidefinite programming

    # Step 2: Ensure the matrix is positive definite
    X_val = X.value
    eigenvalues, eigenvectors = np.linalg.eigh(X_val)
    # Replace negative eigenvalues with a small positive value
    eigenvalues[eigenvalues < 1e-6] = 1e-6
    
    X_positive_definite = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    
    # Perform Cholesky decomposition to get the vectors
    L = np.linalg.cholesky(X_positive_definite)
    
    return L


def random_hyperplane_rounding(L):
    """
    L: The vectors corresponding to each node (output from sdp_vectors).
    Returns the cut (S, T).
    """
    n = L.shape[0]
    r = np.random.randn(n)
    S = [i for i in range(n) if np.dot(r, L[i]) >= 0]
    # T = [i for i in range(n) if i not in S]
    
    return [0.] + [1. if i in S else -1. for i in range(n)]


def get_sdp_correlations(graph):
    """Computes the sdp correlation matrix as required for the cluster algorithm."""

    vectors = sdp_vectors(graph)

    matrix = np.zeros((len(graph.nodes), len(graph.nodes)))

    for u, v in graph.edges:
        matrix[u, v] = np.dot(vectors[u], vectors[v])
    
    return np.pad(matrix, ((1, 0), (1, 0)), mode='constant', constant_values=0.)