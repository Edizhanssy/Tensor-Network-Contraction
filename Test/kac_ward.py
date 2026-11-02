import numpy as np


def kacWardLogZ(rows, cols, K):
    """Exact ln Z of the h=0 Ising model on an open rows x cols square lattice.

    Kac-Ward formula for a planar graph with straight edges:
        Z = 2^N * prod_e cosh(K) * sqrt(det(I - T))
    T acts on directed edges: T[(u->v),(v->w)] = tanh(K) * exp(i*theta/2),
    where theta is the turning angle at v and w != u (no backtracking).
    """
    N = rows * cols
    pos = {i: np.array(divmod(i, cols), dtype=float) for i in range(N)}
    nbrs = [[] for _ in range(N)]
    for i in range(N):
        x, y = divmod(i, cols)
        if x > 0: nbrs[i].append(i - cols)
        if y > 0: nbrs[i].append(i - 1)
        if y < cols - 1: nbrs[i].append(i + 1)
        if x < rows - 1: nbrs[i].append(i + cols)

    darts = [(u, v) for u in range(N) for v in nbrs[u]]   # directed edges
    idx = {d: n for n, d in enumerate(darts)}
    E = len(darts) // 2
    t = np.tanh(K)

    T = np.zeros((2 * E, 2 * E), dtype=complex)
    for (u, v), a in idx.items():
        d1 = pos[v] - pos[u]
        for w in nbrs[v]:
            if w == u:
                continue
            d2 = pos[w] - pos[v]
            theta = np.arctan2(d1[0]*d2[1] - d1[1]*d2[0], d1 @ d2)   # signed turning angle
            T[a, idx[(v, w)]] = t * np.exp(0.5j * theta)

    sign, logdet = np.linalg.slogdet(np.eye(2 * E) - T)
    # det(I - T) is real and positive; logdet is its log
    return N * np.log(2) + E * np.log(np.cosh(K)) + 0.5 * logdet
