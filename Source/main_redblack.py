import numpy as np
from Test.test_mps import testMPS
from Test.test_mps import checkLeftCanonical
from Test.test_mps import mpsToTensor
import scipy.linalg
import heapq


def safeSVD(mat):
    # fast divide-and-conquer (gesdd) first; on the rare LAPACK non-convergence
    # retry this one matrix with the slower but more robust QR-iteration driver (gesvd)
    try:
        return np.linalg.svd(mat, full_matrices=False)
    except np.linalg.LinAlgError:
        print(f"[safeSVD] gesdd failed on shape {mat.shape}, retrying with gesvd")
        return scipy.linalg.svd(mat, full_matrices=False, lapack_driver='gesvd')

# define the tensor network vector!
def defineTensors(B, legs):

    A = [[] for _ in range(len(legs))]
    # the tensors are being made via summation of two outer product u and v!
    u = B[0]
    v = B[1]

    for k in range(len(legs)):
        d = len(legs[k])
        Tu = u
        Tv = v
        for _ in range(d-1):
            Tu = np.multiply.outer(Tu, u)
            Tv = np.multiply.outer(Tv, v)
        A[k] = Tu+Tv

    return A

# Lexicographic mapping function!!
def lexicographicmapping(rows, column):
    totaldim = rows * column
    x = np.zeros(totaldim, dtype=int)
    y = np.zeros(totaldim, dtype=int)
    for i in range(0, totaldim):
        x[i], y[i] = divmod(i, column)
    return x, y

# the IDs are created properly over the whole grid!!
def createIDs(x, y, rows, column):

    adjacency = [[] for _ in range(len(x))]

    for i in range(len(x)):
        if x[i] > 0:
            adjacency[i].append(i - column)
        if y[i] > 0:
            adjacency[i].append(i - 1)
        if y[i] < column-1:
            adjacency[i].append(i+1)
        if x[i] < rows - 1:
            adjacency[i].append(i + column)

    return adjacency


# Converting tensors to MPS form!
# the aim Matrix Product State (MPS) form
#  M1 ──α₁── M2 ──α₂── M3
#  │         │         │
#  a         b         c
# the main idea, unfolding and rank!!
# 1 MPS bound is meaning that dividing tensors into two pieces
# every division corresponds to a matrix
# row: left indexes on the division
# column: right indexes on the division
# This matrix's rank gives the bond dimension (chi) since with a matrix with rank r,
# r times outer products summation: $M_{xy} = \sum_{\alpha=1}^{r} U_{x\alpha} V_{\alpha y}$
# the middle alpha is the bond index!

# every step will have this following pattern:

# 1- Expanding the following block to matrix
# 2- SVD => M = U*S*V.T
# 3- saving U as local tensor, saving S*V.T for the next step!
# the bond index alpha is the output of the SVD!

# for the unfolding a-[bc]

# the kronecker delta vector is constructed for unfolding

# a: row, [bc]: column (can take 4 different values)

# (b, c)  column
# (0, 0)    0
# (0, 1)    1
# (1, 0)    2
# (1, 1)    3

# unfolded matrix!
#            (b,c)=00  01  10  11
#  a = 0  [    1,      0,  0,  0  ]
#  a = 1  [    0,      0,  0,  1  ]

# but on numpy this is just a one command :)
# A1 = delta.reshape(2, 4)


def modifyTensortoMPS(A):

    A_m = [[] for _ in range(len(A))]
    for k in range(len(A)):

        A_m[k] = modifyLegtoMPS(A[k])
    #print([M.shape for M in A_m[17]])
    return A_m

def modifyLegtoMPS(A_l):

    M = []
    carry = A_l.copy()
    # initial value for M1
    chi_L = 1
    tol = 1e-12
    for k in range(A_l.ndim-1):
        physical = A_l.shape[k]
        # Unfolding!
        mat = carry.reshape(physical*chi_L, -1)

        # SVD! M = U * S * V.T
        U, S, Vt = safeSVD(mat)

        Bools = S > tol*S[0]

        chi_R = np.sum(Bools)

        U = U[:, :chi_R]
        S = S[:chi_R]
        Vt = Vt[:chi_R, :]

        # saving U as local tensor!
        M.append(U.reshape(chi_L, physical, chi_R))

        # SV.T goes to the next step!
        carry = np.diag(S) @ Vt

        # update!
        chi_L = chi_R

    M.append(carry.reshape(chi_L, A_l.shape[-1], 1))

    return M

def logSize(mps):

    logsize = 0
    for k in range(len(mps)):
        logsize += np.log(mps[k].shape[1])

    return logsize


def edgeCost(A_m, legs, ls, mu, p):
    # cost of contracting edge (mu, legs[mu][p])!
    nu = legs[mu][p]
    return ls[mu] + ls[nu] - 2 * np.log(A_m[mu][p].shape[1])


def triangles(legs, mu, nu):
    # number of common neighbors of mu and nu [= triangles on edge mu-nu]
    return len(set(legs[mu]) & set(legs[nu]))


def edgeKey(A_m, legs, ls, mu, nu):
    # heap key (mu < nu): cost rounded to a log2 bucket, inside a bucket the edge with the
    # most triangles first (min-heap -> -tri), then the smaller IDs
    p = legs[mu].index(nu)
    cost = edgeCost(A_m, legs, ls, mu, p)
    bucket = int(round(cost / np.log(2)))
    tri = triangles(legs, mu, nu)
    return (bucket, -tri, mu, nu)


def pushEdge(heap, A_m, legs, ls, a, b):
    # entries are always stored from the smaller tensor ID's side: (bucket, -tri, mu, nu) with mu < nu
    mu, nu = (a, b) if a < b else (b, a)
    heapq.heappush(heap, edgeKey(A_m, legs, ls, mu, nu))


def isValid(entry, A_m, legs, ls):
    _, _, mu, nu = entry                               # (bucket, -tri, mu, nu)
    if len(A_m[mu]) == 0 or len(A_m[nu]) == 0:         # an end was contracted away
        return False
    if nu not in legs[mu]:                             # no longer neighbors
        return False
    return edgeKey(A_m, legs, ls, mu, nu) == entry     # bucket or triangles changed since the push?


def popBestEdge(heap, A_m, legs, ls):
    # lazy invalidation: stale entries are not removed when keys change, they are skipped here
    while heap:
        entry = heapq.heappop(heap)
        if isValid(entry, A_m, legs, ls):
            return entry[-2], entry[-1]
    raise RuntimeError("heap is empty but edges are left: an update was not pushed")

# Kernel for red-black methodology:
def touched(legs, i, j):
    # tensors whose data (A_m, legs, ls, cano) the contraction i <- j reads or writes.
    # i survives, j disappears: every neighbour k of j gets legs[k] j -> i.
    # legs[j] already contains i
    return {j} | set(legs[j])


def selectBatch(heap, A_m, legs, ls, K=None):
    # one round = edges of the cheapest bucket whose touched sets are pairwise disjoint.
    # Conflicting and out-of-bucket entries go back to the heap: they are still live edges.
    batch, used, deferred = [], set(), []
    b0 = None
    while heap:
        entry = heapq.heappop(heap)
        if not isValid(entry, A_m, legs, ls):
            continue                                   # stale: drop (a fresh entry was pushed)
        i, j = entry[-2], entry[-1]
        if b0 is None:
            b0 = entry[0]                              # the bucket of this round: set once
        if entry[0] != b0 or (K is not None and len(batch) >= K):
            deferred.append(entry)                     # keep it: back to the heap
            break                                      # heap is ordered -> the rest is >= b0
        t = touched(legs, i, j)
        if t & used:
            deferred.append(entry)                     # conflict: retry next round
        else:
            batch.append((i, j))
            used |= t
    for e in deferred:
        heapq.heappush(heap, e)
    if not batch:
        raise RuntimeError("heap is empty but edges are left: an update was not pushed")
    return batch


def ContractionOrder(A_m, legs, ls, rng=None, tie_tol=1e-9):
    # rng=None: deterministic greedy, the FIRST cheapest edge wins
    # rng given: collect ALL edges whose cost equals the minimum
    best_cost = np.inf
    best_edges = []
    for mu in range(len(legs)):
        for p, nu in enumerate(legs[mu]):
            if mu < nu:
                D_mu_nu = A_m[mu][p].shape[1]
                cost = ls[mu] + ls[nu] - 2 * np.log(D_mu_nu)
                if rng is None:
                    if cost < best_cost:
                        best_cost = cost
                        best_edges = [(mu, nu)]
                elif cost < best_cost - tie_tol:          # strictly cheaper compared to baseline: new candidate list!
                    best_cost = cost
                    best_edges = [(mu, nu)]
                elif abs(cost - best_cost) <= tie_tol and (mu, nu) not in best_edges:   # a tie
                    best_edges.append((mu, nu))

    if rng is None or len(best_edges) == 1:
        return best_edges[0]
    return best_edges[rng.integers(len(best_edges))]

def moveCenter(A_m, cano, i, target):
    # move the orthogonality center of MPS i from cano[i] to site `target` with QR steps
    # (left of the center: left-canonical, right of it: right-canonical)!!
    c = cano[i]
    while c < target:                                   # one step right: Q stays, R goes right
        l, ph, r = A_m[i][c].shape
        Q, R = np.linalg.qr(A_m[i][c].reshape(l * ph, r))
        A_m[i][c] = Q.reshape(l, ph, -1)
        A_m[i][c + 1] = np.einsum('kr, rph -> kph', R, A_m[i][c + 1])
        c += 1
    while c > target:                                   # one step left: Q^T stays, R^T goes left
        l, ph, r = A_m[i][c].shape
        Q, R = np.linalg.qr(A_m[i][c].reshape(l, ph * r).T)
        A_m[i][c] = Q.T.reshape(-1, ph, r)
        A_m[i][c - 1] = np.einsum('lpr, rk -> lpk', A_m[i][c - 1], R.T)
        c -= 1
    cano[i] = c


def tensorSwap(A_m, legs, cano, i, p, right=True, Xhead=None, err=None, tol=1e-12):
    # swap sites p and p+1 of MPS i!!
    # right=True : center p -> p+1 (used when a tensor travels to the right)
    # right=False: center p+1 -> p (used when a tensor travels to the left)
    moveCenter(A_m, cano, i, p if right else p + 1)

    M1 = A_m[i][p]
    M2 = A_m[i][p+1]
    Theta = np.einsum('lap, pbr->lbar', M1, M2)
    l, p2, p0, r = Theta.shape
    theta_mat = Theta.reshape(l*p2, p0*r)
    # B4: the isfinite debug check was removed (the SVD failures were gesdd, handled by safeSVD)

    U, S, Vt = safeSVD(theta_mat)

    rank = int((S > tol * S[0]).sum())                  # numerical rank computed ONCE
    chi_R = rank if Xhead is None else min(Xhead, rank)

    if err is not None:
        if chi_R < len(S):                              # nothing discarded -> delta = 0, skip the sums
            S2 = S * S
            err[0] += S2[chi_R:].sum() / S2.sum()       # err[0]: chi_hat (virtual bonds, swaps)
        if Xhead is not None and rank > Xhead:
            err[2] += 1                                 # err[2]: how often chi_hat was hit
        err[4] = max(err[4], chi_R)                     # err[4]: largest virtual bond kept
    U = U[:, :chi_R]
    S = S[:chi_R]
    Vt = Vt[:chi_R, :]
    legs[i][p], legs[i][p + 1] = legs[i][p + 1], legs[i][p]

    if right:                                           # S goes right: p is a left isometry
        A_m[i][p] = U.reshape(l, p2, chi_R)
        A_m[i][p+1] = (S[:, None] * Vt).reshape(chi_R, p0, r)
        cano[i] = p + 1
    else:                                               # S goes left: p+1 is a right isometry
        A_m[i][p] = (U * S).reshape(l, p2, chi_R)
        A_m[i][p+1] = Vt.reshape(chi_R, p0, r)
        cano[i] = p


def moveToTail(A_m, legs, cano, i, j, Xhead, err):

    p = legs[i].index(j)
    d = len(legs[i])
    for q in range(p, d-1):                             # travels right, center travels with it
        tensorSwap(A_m, legs, cano, i, q, True, Xhead, err)
    moveCenter(A_m, cano, i, d - 1)                     # merge needs the center at the tail

def moveToHead(A_m, legs, cano, i, j, Xhead, err):

    p = legs[j].index(i)
    for q in range(p-1, -1, -1):                        # travels left, center travels with it
        tensorSwap(A_m, legs, cano, j, q, False, Xhead, err)
    moveCenter(A_m, cano, j, 0)                   # merge needs the center at the head


def mergeMPSTensors(A_m, legs, cano, i, j):
    # precondition: center of i at its tail, center of j at its head
    n_i = len(A_m[i])

    M2 = A_m[i][-1]
    N1 = A_m[j][0]
    X = np.einsum('asx, xsb->ab', M2, N1)

    # log-scale normalization, once for all branches
    # (c == 0: amplitude is exactly zero, e.g. destructive interference -> keep X, log(c) = 0)
    c = np.linalg.norm(X)
    if c == 0:
        c = 1.0
    X = X / c

    if len(A_m[j][:]) >= 2:
        N2 = A_m[j][1]
        # not isometric anymore!!, we want to keep the isometry of M1!
        N2p = np.einsum('ab, bpr->apr', X, N2)
        # merging
        A_m[i] = A_m[i][:-1] + [N2p] + A_m[j][2:]
        cano[i] = n_i - 1                               # X absorbed into N2: that site is the center
        # E <- E \ {(i,j)}: already done inside mergeMPSTensors
        # (legs[i] drops j via [:-1], legs[j] is cleared)
        nbrs_j = legs[j][1:]
        legs[i] = legs[i][:-1] + nbrs_j
        A_m[j] = []
        legs[j] = []

    elif len(A_m[j][:]) == 1 and len(A_m[i][:]) >= 2:
        M1 = A_m[i][-2]
        M1p = np.einsum('lpa, ab->lpb', M1, X)
        # merging
        A_m[i] = A_m[i][:-2] + [M1p] + A_m[j][2:]
        cano[i] = n_i - 2                               # X absorbed into M1: the new tail is the center
        # E <- E \ {(i,j)}: already done inside mergeMPSTensors
        # (legs[i] drops j via [:-1], legs[j] is cleared)
        nbrs_j = legs[j][1:]
        legs[i] = legs[i][:-1] + nbrs_j
        A_m[j] = []
        legs[j] = []

    else:
        nbrs_j = legs[j][1:]

    Z = X

    return nbrs_j, Z, np.log(c)

def moveAdjacent(A_m, legs, cano, i, k, Xhead=None, err=None):

    p1 = legs[i].index(k)
    p2 = legs[i].index(k, p1+1)
    for q in range(p1, p2-1):
        tensorSwap(A_m, legs, cano, i, q, True, Xhead, err)

    return legs[i].index(k)

def mergeAdjacent(A_m, cano, site, p):
    # two left (or two right) isometries multiply to a left (right) isometry, so the
    # canonical form survives; only the center index shifts when it sits at or after p+1
    if cano[site] > p:
        cano[site] -= 1

    T1 = A_m[site][p]
    T2 = A_m[site][p+1]

    X = np.einsum('lap, pbr->labr', T1, T2)
    l, a, b, r = X.shape
    X = X.reshape(l, a*b, r)
    A_m[site][p] = X
    del A_m[site][p+1]

def canonicalize(A_m, site, p):

    # from left to right sweep!
    for q in range(0, p):

        l, ph, r = A_m[site][q].shape

        # Unfolding!
        mat_cano = A_m[site][q].reshape(l * ph, -1)

        Q, R = np.linalg.qr(mat_cano)

        A_m[site][q] = Q.reshape(l, ph, -1)

        A_m[site][q+1] = np.einsum('kr, rph-> kph', R, A_m[site][q+1])

    n = len(A_m[site])

    # from right to left sweep!
    for j in range(n-1, p, -1):
        l, ph, r = A_m[site][j].shape

        # Unfolding!
        mat_cano = A_m[site][j].reshape(l, ph * r)

        Q, R = np.linalg.qr(mat_cano.T)

        A_m[site][j] = Q.T.reshape(-1, ph, r)

        A_m[site][j - 1] = np.einsum('lpr, rk -> lpk', A_m[site][j - 1], R.T)


def compressMPS(A_m, cano, i, Xhead=None, tol=1e-12, err=None):
    # two-site sweep over the whole MPS (as CATN's compress_opt)!!
    n = len(A_m[i])
    if n < 2:
        return
    moveCenter(A_m, cano, i, n - 1)
    for c in range(n - 1, 0, -1):
        l, ph, r = A_m[i][c].shape
        U, S, Vt = safeSVD(A_m[i][c].reshape(l, ph * r))            # center site, split off its left bond
        chi = int(np.sum(S > tol * S[0]))
        if Xhead is not None:
            chi = min(chi, Xhead)
        if err is not None:
            err[0] += np.sum(S[chi:] ** 2) / np.sum(S ** 2)
            if Xhead is not None and np.sum(S > tol * S[0]) > Xhead:
                err[2] += 1
        A_m[i][c] = Vt[:chi, :].reshape(chi, ph, r)                  # right isometry
        A_m[i][c - 1] = np.einsum('lpa, ab -> lpb', A_m[i][c - 1], U[:, :chi] * S[:chi])
    cano[i] = 0


def cutBond(A_m, cano, i, p_i, k, p_k, dhead, err, tol=1e-12):
    # Physical bond between site p_i of MPS i and site p_k of MPS k (both carry the same bond D)!!
    # 1. centers onto the two sites (canonical gauge -> the singular values below are the true
    #    Schmidt values across this bond, so cutting the smallest is the optimal truncation)
    # 2. QR trick: never build the (l_i r_i) x (l_k r_k) matrix; its singular values are those of
    #    the small D x D matrix R_i R_k^T
    # 3. keep min(D_hat, numerical rank): above D_hat this truncates, below it only drops
    #    singular values ~0, i.e. it is lossless and just removes redundant dimension
    moveCenter(A_m, cano, i, p_i)
    moveCenter(A_m, cano, k, p_k)
    l_i, D, r_i = A_m[i][p_i].shape
    l_k, _, r_k = A_m[k][p_k].shape
    M_i = A_m[i][p_i].transpose(0, 2, 1).reshape(l_i * r_i, D)  # (outer legs, bond)
    M_k = A_m[k][p_k].transpose(0, 2, 1).reshape(l_k * r_k, D)
    Q_i, R_i = np.linalg.qr(M_i) if M_i.shape[0] > D else (None, M_i)
    Q_k, R_k = np.linalg.qr(M_k) if M_k.shape[0] > D else (None, M_k)
    U, S, Vt = safeSVD(R_i @ R_k.T)                       # = SVD of M_i M_k^T
    rank = int(np.sum(S > tol * S[0]))
    chi = min(dhead, rank)
    err[1] += np.sum(S[chi:] ** 2) / np.sum(S ** 2)       # err[1]: D_hat (physical bonds)
    if rank > dhead:
        err[3] += 1                                       # err[3]: how often D_hat was hit
    err[5] = max(err[5], chi)                             # err[5]: largest physical bond kept
    sq = np.sqrt(S[:chi])
    new_i = U[:, :chi] * sq                               # bond split as sqrt(S) | sqrt(S)
    new_k = Vt[:chi, :].T * sq
    if Q_i is not None:
        new_i = Q_i @ new_i
    if Q_k is not None:
        new_k = Q_k @ new_k
    A_m[i][p_i] = new_i.reshape(l_i, r_i, chi).transpose(0, 2, 1)
    A_m[k][p_k] = new_k.reshape(l_k, r_k, chi).transpose(0, 2, 1)
    # centers stay at p_i and p_k: left of them left-canonical, right of them right-canonical

# main MPS Calculus kernel!
# Z will be returning in logarithmic scale!!
def MPSCalculus(A, legs, dhead, Xhead=None, tol=1e-12, rng=None, compress=False, stats=None, K=1):

    # MPS Form! (its already in Canonical form and checked!!)
    A_m = modifyTensortoMPS(A)
    cano = [len(m) - 1 for m in A_m]                    # TT-SVD is left-canonical: center at the last site
    ls = [logSize(m) for m in A_m]                      # B1: logSize cache
    heap = []                                           # B2: all edges once, then only updates
    if rng is None:
        for mu in range(len(legs)):
            for p, nu in enumerate(legs[mu]):
                if mu < nu:
                    heap.append(edgeKey(A_m, legs, ls, mu, nu))   # B3 key
        heapq.heapify(heap)
    #testMPS(A, A_m)
    #checkLeftCanonical(A_m[17])
    #print(f"||A[17]|| = {np.linalg.norm(A[17]):.6f}")

    # to keep the alive sites!!
    alive = set(range(len(A_m)))
    log_scale = 0.0
    Z_total = 1.0                                       # product of the scalars of finished components
    err = [0.0, 0.0, 0, 0, 0, 0]   # eps_chi, eps_D, hits_chi, hits_D, max_chi, max_D
    # while size of the vertex list is greater than one!
    rounds = 0                                          # C1: number of rounds (= steps when K=1)
    while len(alive) > 1:
        # Choosing edges: one ROUND = a batch of edges with pairwise disjoint touched sets
        if rng is None:
            batch = selectBatch(heap, A_m, legs, ls, K)  # C1 (K=1: exactly the B4 greedy pick)
        else:
            batch = [ContractionOrder(A_m, legs, ls, rng)]  # random tie-breaking: old scan (experiments only)
        rounds += 1
        # one STEP per edge; the steps of a batch touch disjoint tensors (still run serially here)
        for i, j in batch:
            # move the corresponding tensor on (i,j) on A(i) to the TAIL &
            moveToTail(A_m, legs, cano, i, j, Xhead, err)

            # move the corresponding tensor on (i,j) on A(j) to the HEAD
            moveToHead(A_m, legs, cano, i, j, Xhead, err)

            # merge two MPS tensors Ai and Aj by contracting edge (i,j) that connects them
            D_ij = A_m[i][-1].shape[1]                      # read before the merge eats that site
            last = len(A_m[i]) == 1 and len(A_m[j]) == 1    # i and j only share this edge: last step of a component
            nbrs_j, Z, log_c = mergeMPSTensors(A_m, legs, cano, i, j)
            ls[i] += ls[j] - 2 * np.log(D_ij)
            log_scale = log_c + log_scale
            if last:
                # this connected component is now one scalar Z. A network can have several components
                # (e.g. a short circuit whose CZs split the qubits into groups): multiply their scalars
                Z_total = Z_total * Z.item()
                A_m[i], A_m[j], legs[i], legs[j] = [], [], [], []
                alive.discard(i)
                alive.discard(j)
                continue

            # for loop for updating the contracting j for legs
            for k in nbrs_j:
                j_first = (i in legs[k]) and (legs[k].index(j) < legs[k].index(i))
                # E ← E \ {(j,k)}
                legs[k][legs[k].index(j)] = i

                if legs[i].count(k) > 1:
                    p_i = moveAdjacent(A_m, legs, cano, i, k, Xhead, err)
                    p_k = moveAdjacent(A_m, legs, cano, k, i, Xhead, err)
                    if j_first:
                        tensorSwap(A_m, legs, cano, k, p_k, True, Xhead, err)
                    mergeAdjacent(A_m, cano, i, p_i)
                    mergeAdjacent(A_m, cano, k, p_k)
                    del legs[i][p_i+1]
                    del legs[k][p_k+1]

                    D_ik = A_m[i][p_i].shape[1]
                    cutBond(A_m, cano, i, p_i, k, p_k, dhead, err, tol)   # A2: always, lossless below D_hat
                    chi = A_m[i][p_i].shape[1]
                    ls[i] += np.log(chi) - np.log(D_ik)
                    ls[k] += np.log(chi) - np.log(D_ik)

            if compress:
                compressMPS(A_m, cano, i, Xhead, tol, err)
            alive.remove(j)

            # push the new key of every edge whose key may have changed. The key depends on ls (cost)
            # and on legs (triangles). legs changed for i (inherits j's neighbours) and for every former
            # neighbour k of j (j -> i); ls changed for i and the cut k's, which are also in nbrs_j.
            if rng is None and len(alive) > 1:
                pushed = set()
                for t in [i] + nbrs_j:
                    for nu in legs[t]:
                        key = (min(t, nu), max(t, nu))
                        if key not in pushed:
                            pushed.add(key)
                            pushEdge(heap, A_m, legs, ls, t, nu)


    # exactly zero amplitude -> ln Z = -inf (exp(-inf) = 0), without a log(0) warning
    if stats is not None:
        stats.update(eps_chi=err[0], eps_D=err[1], hits_chi=err[2], hits_D=err[3],
                     max_chi=err[4], max_D=err[5], rounds=rounds)
    eps = err[0] + err[1]                               # same total as before
    if Z_total == 0:
        return -np.inf, eps
    return log_scale + np.log(Z_total), eps



if __name__ == '__main__':

    # tensors
    numberofColumns = 4
    numberofRows = 4
    totaldim = numberofColumns * numberofRows

    # Spin Directions
    spin = np.array(([1.0], [-1.0]))

    # onside terms are neglected
    h = 0.0
    beta = 0.1
    J = 1.0
    K = beta*J
    a = np.sqrt(np.cosh(K)/2.0)
    b = np.sqrt(np.sinh(K)/2.0)
    B = np.array(([a+b, a-b], [a-b, a+b]))
    # Defining parameters
    dhead = 50
    Xhead = 500

    # defining the x and y coordinates
    x, y = lexicographicmapping(numberofRows, numberofColumns)

    # defining leg vector!
    legs = createIDs(x, y, numberofRows, numberofColumns)

    # Defining tensors!
    A = defineTensors(B, legs)

    # the main kernel for MPS Calculus
    lnZ, accumerror = MPSCalculus(A, legs, dhead, Xhead)

    Z_bf_4x4 = 73943.41234729308
    Z_bf = 66.2889504116254
    print(np.exp(lnZ), abs(np.exp(lnZ) - Z_bf_4x4) / Z_bf_4x4)
    print("the accumulated error is: ", accumerror)


