# B2 test: the heap must pick, at every step, an edge of minimum cost (as a full scan would).
#   1. greedy check: cost of the popped edge (recomputed from scratch) - true minimum over all edges < 1e-9
#   2. cache check:  ls[t] == logSize(A_m[t]) for every live MPS (B1 still correct)
#   3. same result:  same ln|amp| as main_B1.py, checked only when BOTH runs are exact (eps < 1e-10).
#      When ties are broken differently, one order may hit D_hat and the other not (see L6 d16);
#      that is a property of greedy tie-breaking (B3), not an error of the heap.
#   4. heap size:    how many stale entries pile up (memory sanity)
#   python test_B2.py            # small circuits, a few seconds
#   python test_B2.py 20         # also L = 20
import sys
import numpy as np
import main_B1 as old
import main_redblack as new
from TensorPrimitive.circuits import randomCircuit
from TensorPrimitive.network import circuitToEdges, edgesToLegs

CASES = [(4, 8), (5, 12), (6, 14), (6, 16), (10, 8)] + [(int(L), 8) for L in sys.argv[1:]]


def fresh_cost(A_m, legs, mu, nu):
    p = legs[mu].index(nu)
    return new.logSize(A_m[mu]) + new.logSize(A_m[nu]) - 2 * np.log(A_m[mu][p].shape[1])


def run_new(A, legs):
    st = dict(gap=0.0, drift=0.0, steps=0, max_heap=0, n_edges=0)
    orig = new.popBestEdge
    def rec(heap, A_m, legs_, ls):
        st["max_heap"] = max(st["max_heap"], len(heap))
        n_edges = sum(1 for mu in range(len(legs_)) for nu in legs_[mu] if mu < nu)
        st["n_edges"] = max(st["n_edges"], n_edges)
        for t in range(len(A_m)):
            if len(A_m[t]) > 0:
                st["drift"] = max(st["drift"], abs(ls[t] - new.logSize(A_m[t])))
        e = orig(heap, A_m, legs_, ls)
        best = min(fresh_cost(A_m, legs_, mu, nu) for mu in range(len(legs_)) for nu in legs_[mu] if mu < nu)
        st["gap"] = max(st["gap"], fresh_cost(A_m, legs_, *e) - best)
        st["steps"] += 1
        return e
    new.popBestEdge = rec
    try:
        lnZ, eps = new.MPSCalculus(A, legs, 50, 500)
    finally:
        new.popBestEdge = orig
    return lnZ, eps, st


ok_all = True
for L, d in CASES:
    def net():
        A, el = circuitToEdges(randomCircuit(L, d, seed=0), [0] * (L * L))
        return A, edgesToLegs(el)
    z_old, e_old = old.MPSCalculus(*net(), 50, 500)
    z_new, eps, st = run_new(*net())
    dz = abs(complex(z_old) - complex(z_new))
    both_exact = e_old < 1e-10 and eps < 1e-10
    ok = st["gap"] < 1e-9 and st["drift"] < 1e-9 and (dz < 1e-9 or not both_exact)
    ok_all &= ok
    print(f"L{L} d{d}: steps {st['steps']}  greedy gap {st['gap']:.1e}  cache drift {st['drift']:.1e}  "
          f"|Δ lnZ| vs B1 {dz:.1e}{'' if both_exact else ' (not both exact: not compared)'}  eps B1/B2 {e_old:.0e}/{eps:.0e}  max heap {st['max_heap']} (edges {st['n_edges']})  "
          f"-> {'OK' if ok else 'FAIL'}")
print("B2 test:", "PASSED" if ok_all else "FAILED")
