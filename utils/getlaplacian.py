import numpy as np

def getlaplacian(n, type=0):
    if type == 0:
        adj = np.zeros((n, n))
        for i in range(1, n-1):
            adj[i, i-1] = 1
            adj[i, i+1] = 1
        adj[0, 1] = 1
        adj[0, n-1] = 1
        adj[n-1, n-2] = 1
        adj[n-1, 0] = 1
    elif type == 1:
        adj = np.zeros((n, n))
        for i in range(1, n-1):
            adj[i, i-1] = 1
            adj[i, i+1] = 1
        adj[0, 1] = 1
        adj[n-1, n-2] = 1
    elif type == 2:
        adj = np.array[[0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                       [1, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                       [0, 1, 0, 1, 0, 0, 1, 0, 0, 0],
                       [0, 1, 1, 0, 1, 1, 0, 0, 0, 0],
                       [0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
                       [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                       [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
                       [0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
                       [0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
                       [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]]
    elif type == 3:
        adj = np.array([[0, 1, 1, 1, 1],
                        [1, 0, 1, 1, 1],
                        [1, 1, 0, 1, 1],
                        [1, 1, 1, 0, 1],
                        [1, 1, 1, 1, 0]])

    deg = np.diag(np.sum(adj, axis=1))
    lap = deg - adj
    return adj, lap
