# B1 test: the logSize cache must not change WHAT is computed, only how fast.
#   1. cache check: at every ContractionOrder call, ls[t] == logSize(A_m[t]) (up to rounding, 1e-9)
#   2. greedy check: every picked edge has the minimum cost (recomputed from scratch) up to 1e-9.
#      This is the real invariant: the exact SEQUENCE may differ from main_A2.py, because when two
#      edges have mathematically equal cost, rounding at the 1e-14 level decides which one is "first",
#      and the cache rounds differently than a fresh sum. "same order" is therefore only reported.
#   3. same result: same ln|amp| as main_A2.py (both exact on these circuits)
#   python test_B1.py            # small circuits, a few seconds
#   python test_B1.py 20         # also L = 20 (about 1-2 min)
import sys
import numpy as np
import main_A2 as old
import main_redblack as new
from TensorPrimitive.circuits import randomCircuit
from TensorPrimitive.network import circuitToEdges, edgesToLegs

CASES = [(4, 8), (5, 12), (6, 14), (10, 8)] + [(int(L), 8) for L in sys.argv[1:]]


def run_old(A, legs):
    picks, orig = [], old.ContractionOrder
    def rec(A_m, legs_, rng=None, *a, **k):
        e = orig(A_m, legs_, rng, *a, **k); picks.append(e); return e
    old.ContractionOrder = rec
    try:
        lnZ, eps = old.MPSCalculus(A, legs, 50, 500)
    finally:
        old.ContractionOrder = orig
    return picks, lnZ, eps


def run_new(A, legs):
    picks, drift, orig = [], [0.0], new.ContractionOrder
    gap = [0.0]
    def fresh_cost(A_m, legs_, mu, nu):
        p = legs_[mu].index(nu)
        return new.logSize(A_m[mu]) + new.logSize(A_m[nu]) - 2 * np.log(A_m[mu][p].shape[1])
    def rec(A_m, legs_, ls, rng=None, *a, **k):
        for t in range(len(A_m)):                    # cache check against a fresh logSize
            if len(A_m[t]) > 0:
                drift[0] = max(drift[0], abs(ls[t] - new.logSize(A_m[t])))
        e = orig(A_m, legs_, ls, rng, *a, **k); picks.append(e)
        best = min(fresh_cost(A_m, legs_, mu, nu) for mu in range(len(legs_)) for nu in legs_[mu] if mu < nu)
        gap[0] = max(gap[0], fresh_cost(A_m, legs_, *e) - best)   # 0 if the pick is a true minimum
        return e
    new.ContractionOrder = rec
    try:
        lnZ, eps = new.MPSCalculus(A, legs, 50, 500)
    finally:
        new.ContractionOrder = orig
    return picks, lnZ, eps, drift[0], gap[0]


ok_all = True
for L, d in CASES:
    def net():
        A, el = circuitToEdges(randomCircuit(L, d, seed=0), [0] * (L * L))
        return A, edgesToLegs(el)
    p_old, z_old, e_old = run_old(*net())
    p_new, z_new, e_new, drift, gap = run_new(*net())
    first = next((n for n, (a, b) in enumerate(zip(p_old, p_new)) if a != b), None)
    same_order = first is None and len(p_old) == len(p_new)
    dz = abs(complex(z_old) - complex(z_new))
    ok = drift < 1e-9 and gap < 1e-9 and dz < 1e-9
    ok_all &= ok
    print(f"L{L} d{d}: steps {len(p_new)}  cache drift {drift:.1e}  greedy gap {gap:.1e}  |Δ lnZ| {dz:.1e}  "
          f"same sequence as A2: {same_order}" + ("" if same_order else f" (first tie at step {first})")
          + f"  -> {'OK' if ok else 'FAIL'}")
print("B1 test:", "PASSED" if ok_all else "FAILED")
