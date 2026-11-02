import numpy as np


def mpsToTensor(mps):
    """Contract an MPS (list of (l, p, r) tensors) back into a dense tensor."""
    T = mps[0]
    for M in mps[1:]:
        T = np.tensordot(T, M, axes=([-1], [0]))
    return T.reshape(T.shape[1:-1])


def testMPS(A, A_m, tol=1e-12):
    """Max reconstruction error of every site's MPS against its dense tensor."""
    worst = 0.0
    for k in range(len(A)):
        err = np.max(np.abs(mpsToTensor(A_m[k]) - A[k]))
        worst = max(worst, err)
    print(f"testMPS: max reconstruction error = {worst:.2e}")
    return worst < tol


def checkLeftCanonical(mps, verbose=True):
    """All tensors except the last must satisfy Q^H Q = I (left-canonical)."""
    ok = True
    for q, M in enumerate(mps[:-1]):
        l, p, r = M.shape
        Q = M.reshape(l * p, r)
        good = np.allclose(Q.conj().T @ Q, np.eye(r))
        ok = ok and good
        if verbose:
            print(f"  tensor {q}: shape {M.shape}  left-canonical = {good}")
    if verbose:
        print(f"  norm of last tensor = {np.linalg.norm(mps[-1]):.6f}")
    return ok


def testMergeAdjacent(mergeAdjacent, seed=0):
    """Merging two neighbours must not change the dense tensor (a != b on purpose)."""
    rng = np.random.default_rng(seed)
    mps = [rng.standard_normal(s) for s in [(1, 2, 3), (3, 2, 4), (4, 3, 2), (2, 2, 1)]]
    before = mpsToTensor(mps)
    A_m = [[m.copy() for m in mps]]
    mergeAdjacent(A_m, 0, 1)
    after = mpsToTensor(A_m[0])
    err = np.max(np.abs(after - before.reshape(2, 6, 2)))
    print(f"testMergeAdjacent: len={len(A_m[0])} (expect 3), err={err:.1e}")
    return err < 1e-12


def testCanonicalize(canonicalize, seed=1):
    """Mixed-canonical form at p: dense tensor unchanged, ||center|| = ||MPS||."""
    rng = np.random.default_rng(seed)
    mps = [rng.standard_normal(s) for s in [(1, 2, 3), (3, 2, 4), (4, 2, 3), (3, 2, 2), (2, 2, 1)]]
    before = mpsToTensor(mps)
    A_m = [[m.copy() for m in mps]]
    p = 2
    canonicalize(A_m, 0, p)
    err = np.max(np.abs(mpsToTensor(A_m[0]) - before))
    norm_ok = np.isclose(np.linalg.norm(A_m[0][p]), np.linalg.norm(before))
    print(f"testCanonicalize: dense err={err:.1e}, ||center|| == ||MPS||: {norm_ok}")
    return err < 1e-12 and norm_ok
