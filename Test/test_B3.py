# B3 test: CATN rule (log2 bucket of the cost, ties -> most triangles) on top of the B2 heap.
#   1. key check:    at every step the popped edge's key (recomputed from scratch with edgeKey) equals the
#                    minimum key over ALL live edges. The key is made of ints, so this must hold exactly:
#                    smallest bucket, inside it the most triangles, then the smallest (mu, nu).
#                    If an update were not pushed (a stale triangle count), this is where it shows.
#   2. cache check:  ls[t] == logSize(A_m[t]) for every live MPS (B1 still correct)
#   3. no self-loop: t not in legs[t] for every live MPS (triangles() relies on it)
#   4. accuracy:     eps and ln|amp| vs main_B2.py. Not a pass/fail item: a different tie-break gives a
#                    different order, which may hit D_hat or not. Shown so L6 d16 (B2: eps 3e-2) is visible.
#   python test_B3.py            # small circuits, a few seconds
#   python test_B3.py 20         # also L = 20
import sys
import numpy as np
import main_B2 as old
import main_redblack as new
from TensorPrimitive.circuits import randomCircuit
from TensorPrimitive.network import circuitToEdges, edgesToLegs

CASES = [(4, 8), (5, 12), (6, 14), (6, 16), (10, 8)] + [(int(L), 8) for L in sys.argv[1:]]


def run_new(A, legs):
    st = dict(key_err=0, drift=0.0, self_loops=0, steps=0, max_heap=0, n_edges=0, tri_wins=0)
    orig = new.popBestEdge

    def rec(heap, A_m, legs_, ls):
        st["max_heap"] = max(st["max_heap"], len(heap))
        live = [t for t in range(len(A_m)) if len(A_m[t]) > 0]
        for t in live:
            st["drift"] = max(st["drift"], abs(ls[t] - new.logSize(A_m[t])))
            st["self_loops"] += t in legs_[t]
        keys = [new.edgeKey(A_m, legs_, ls, mu, nu) for mu in live for nu in set(legs_[mu]) if mu < nu]
        st["n_edges"] = max(st["n_edges"], len(keys))
        best = min(keys)
        e = orig(heap, A_m, legs_, ls)
        if new.edgeKey(A_m, legs_, ls, *e) != best:
            st["key_err"] += 1
        # did the triangle count decide? (another edge in the same bucket with fewer triangles)
        if any(k[0] == best[0] and k[1] > best[1] for k in keys):
            st["tri_wins"] += 1
        st["steps"] += 1
        return e

    new.popBestEdge = rec
    try:
        s = {}
        lnZ, eps = new.MPSCalculus(A, legs, 50, 500, stats=s)
    finally:
        new.popBestEdge = orig
    return lnZ, eps, st, s


ok_all = True
for L, d in CASES:
    def net():
        A, el = circuitToEdges(randomCircuit(L, d, seed=0), [0] * (L * L))
        return A, edgesToLegs(el)
    z_old, e_old = old.MPSCalculus(*net(), 50, 500)
    z_new, eps, st, s = run_new(*net())
    dz = abs(float(np.real(z_old)) - float(np.real(z_new)))
    ok = st["key_err"] == 0 and st["drift"] < 1e-9 and st["self_loops"] == 0
    ok_all &= ok
    print(f"L{L} d{d}: steps {st['steps']}  key errors {st['key_err']}  cache drift {st['drift']:.1e}  "
          f"self-loops {st['self_loops']}  steps decided by triangles {st['tri_wins']}  "
          f"max heap {st['max_heap']} (edges {st['n_edges']})  -> {'OK' if ok else 'FAIL'}")
    print(f"        eps B2/B3 {e_old:.1e}/{eps:.1e}   D_hat hits B3 {s['hits_D']}   "
          f"|Δ ln|amp|| B3 vs B2 {dz:.1e}")
print("B3 test:", "PASSED" if ok_all else "FAILED")
