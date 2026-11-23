import numpy as np
#from Test.test_mps import testMPS
#from Test.test_mps import checkLeftCanonical
#from Test.test_mps import mpsToTensor
import scipy.linalg

# This specific function is added via AI guidance!
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


def ContractionOrder(A_m, legs, rng=None, tie_tol=1e-9):
    # greedy: cheapest edge; ties (same log2 bucket) -> most triangles, then smaller IDs
    best_key, best_edge = None, None
    for mu in range(len(legs)):
        for p, nu in enumerate(legs[mu]):
            if mu < nu:
                D_mu_nu = A_m[mu][p].shape[1]
                cost = logSize(A_m[mu]) + logSize(A_m[nu]) - 2 * np.log(D_mu_nu)
                tri = len(set(legs[mu]) & set(legs[nu]))
                key = (int(round(cost / np.log(2))), -tri, mu, nu)
                if best_key is None or key < best_key:
                    best_key, best_edge = key, (mu, nu)
    return best_edge

def tensorSwap(A_m, legs, i, p, Xhead = None, err=None, tol=1e-12):

    M1 = A_m[i][p]
    M2 = A_m[i][p+1]
    Theta = np.einsum('lap, pbr->lbar', M1, M2)
    l, p2, p0, r = Theta.shape
    theta_mat = Theta.reshape(l*p2, p0*r)
    isFinite = np.isfinite(theta_mat)

    if not isFinite.all():
        good = np.abs(theta_mat[isFinite])
        raise FloatingPointError(
            f"tensorSwap: theta_mat not finite | shape={theta_mat.shape} | "
            f"n_nonfinite={np.sum(~isFinite)} (nan={np.sum(np.isnan(theta_mat))}, "
            f"inf={np.sum(np.isinf(theta_mat))}) | "
            f"max|finite|={good.max() if good.size else 'none'} | "
            f"|M1|max={np.abs(M1).max()}, |M2|max={np.abs(M2).max()}")

    U, S, Vt = safeSVD(theta_mat)

    if Xhead is None:
        chi_R = np.sum(S > tol*S[0])
    else:
        chi_R = min(Xhead, np.sum(S > tol*S[0]))

    delta = np.sum(S[chi_R:]**2)/np.sum(S**2)
    if err is not None:
        err[0] += delta
    U = U[:, :chi_R]
    S = S[:chi_R]
    Vt = Vt[:chi_R, :]
    legs[i][p], legs[i][p + 1] = legs[i][p + 1], legs[i][p]

    A_m[i][p] = U.reshape(l, p2, chi_R)
    A_m[i][p+1] = (np.diag(S) @ Vt).reshape(chi_R, p0, r)


def moveToTail(A_m, legs, i, j, Xhead, err):

    p = legs[i].index(j)
    d = len(legs[i])
    for q in range(p, d-1):
        tensorSwap(A_m, legs, i, q, Xhead, err)

def moveToHead(A_m, legs, i, j, Xhead, err):

    p = legs[j].index(i)
    for q in range(p-1, -1, -1):
        tensorSwap(A_m, legs, j, q, Xhead, err)


def mergeMPSTensors(A_m, legs, i, j):

    M2 = A_m[i][-1]
    N1 = A_m[j][0]
    X = np.einsum('asx, xsb->ab', M2, N1)

    # log-scale normalization, once for all branches
    # (c == 0: amplitude is exactly zero)
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

def moveAdjacent(A_m, legs, i, k, Xhead=None, err=None):

    p1 = legs[i].index(k)
    p2 = legs[i].index(k, p1+1)
    for q in range(p1, p2-1):
        tensorSwap(A_m, legs, i, q, Xhead, err)

    return legs[i].index(k)

def mergeAdjacent(A_m, site, p):

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


# main MPS Calculus kernel!
# Z will be returning in logarithmic scale!!
def MPSCalculus(A, legs, dhead, Xhead=None, tol=1e-12, rng=None):
    # rng: np.random.Generator -> random tie-breaking in ContractionOrder (None = baseline order)

    # MPS Form! (its already in Canonical form and checked!!)
    A_m = modifyTensortoMPS(A)
    #testMPS(A, A_m)
    #checkLeftCanonical(A_m[17])
    #print(f"||A[17]|| = {np.linalg.norm(A[17]):.6f}")

    # to keep the alive sites!!
    alive = set(range(len(A_m)))
    log_scale = 0.0
    err = [0.0]
    # while size of the vertex list is greater than one!
    while len(alive) > 1:
        # Choosing edges
        i, j = ContractionOrder(A_m, legs, rng)
        #print(i, j ,legs)

        # move the corresponding tensor on (i,j) on A(i) to the TAIL &
        moveToTail(A_m, legs, i, j, Xhead, err)

        # move the corresponding tensor on (i,j) on A(j) to the HEAD
        moveToHead(A_m, legs, i, j, Xhead, err)

        # merge two MPS tensors Ai and Aj by contracting edge (i,j) that connects them
        nbrs_j, Z, log_c = mergeMPSTensors(A_m, legs, i, j)
        log_scale = log_c + log_scale

        # for loop for updating the contracting j for legs
        for k in nbrs_j:
            j_first = (i in legs[k]) and (legs[k].index(j) < legs[k].index(i))
            # E ← E \ {(j,k)}
            legs[k][legs[k].index(j)] = i

            if legs[i].count(k) > 1:
                p_i = moveAdjacent(A_m, legs, i, k, Xhead, err)
                p_k = moveAdjacent(A_m, legs, k, i, Xhead, err)
                if j_first:
                    tensorSwap(A_m, legs, k, p_k, Xhead, err)
                mergeAdjacent(A_m, i, p_i)
                mergeAdjacent(A_m, k, p_k)
                del legs[i][p_i+1]
                del legs[k][p_k+1]

                if A_m[k][p_k].shape[1] > dhead:
                    canonicalize(A_m, i, p_i)
                    canonicalize(A_m, k, p_k)

                    T_i = A_m[i][p_i]
                    T_k = A_m[k][p_k]

                    l_i, D, r_i = T_i.shape
                    l_k, D, r_k = T_k.shape

                    Tp = np.einsum('ldr, mdn->lrmn', T_i, T_k)
                    Tp_mat = Tp.reshape(l_i*r_i, l_k*r_k)
                    U, S, Vt = safeSVD(Tp_mat)
                    chi_R = min(dhead, np.sum(S > tol*S[0]))
                    delta = np.sum(S[chi_R:] ** 2) / np.sum(S ** 2)
                    #if err is not None:
                    err[0] += delta
                    U = U[:, :chi_R]
                    Vt = Vt[:chi_R, :]
                    S = S[:chi_R]
                    T_i_new = U.reshape(l_i, r_i, chi_R).transpose(0, 2, 1)
                    T_k_new = Vt.reshape(chi_R, l_k, r_k).transpose(1, 0, 2)
                    sqrtS = np.sqrt(S)
                    A_m[i][p_i] = np.einsum('lar, a->lar', T_i_new, sqrtS)
                    A_m[k][p_k] = np.einsum('mak, a->mak', T_k_new, sqrtS)
                    assert A_m[i][p_i].shape[1] == A_m[k][p_k].shape[1]

        alive.remove(j)


    # exactly zero amplitude -> ln Z = -inf (exp(-inf) = 0), without a log(0) warning
    if Z.item() == 0:
        return -np.inf, err[0]
    return log_scale + np.log(Z.item()), err[0]


if __name__ == '__main__':
    from TensorPrimitive.circuits import randomCircuit
    from TensorPrimitive.network import circuitToEdges, edgesToLegs

    L, d = 10, 8                      # n = 100 qubits
    dhead, Xhead = 50, 500            # D_max, chi_max

    circuit = randomCircuit(L, d, seed=0)
    A, edges = circuitToEdges(circuit, [0] * (L * L))
    legs = edgesToLegs(edges)

    ln_amp, eps = MPSCalculus(A, legs, dhead, Xhead)
    print(f"ln|amp| = {ln_amp.real:.10f}   eps_SVD = {eps:.2e}")


'''
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

'''
