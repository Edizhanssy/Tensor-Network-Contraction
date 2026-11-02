# C1 test: batches of independent edges.
#   1. K=1 is bit-identical to main.py (the default must not change anything)
#   2. inside every batch the touched sets are pairwise disjoint
#   3. the C1-4 argument, checked: right before EVERY step of a batch, the edge is still valid
#      (both ends alive, still neighbours, key unchanged since selection) -> no re-check needed
#   4. accuracy for K = 4 and K = None vs the exact references (reported; order may change eps)
#   python test_C1.py
import numpy as np
import main_redblack as new
import main as old
from TensorPrimitive.circuits import randomCircuit
from TensorPrimitive.network import circuitToEdges, edgesToLegs

CASES = [(4, 8, None), (5, 12, -10.863674078189362), (6, 14, -13.533747486762),
         (6, 16, -12.552317591793326), (10, 8, -37.58825906084385)]


def net(L, d):
    A, el = circuitToEdges(randomCircuit(L, d, seed=0), [0] * (L * L))
    return A, edgesToLegs(el)


def run(L, d, K):
    st = dict(overlap=0, invalid=0, max_batch=0, steps=0)
    pending = []                                   # keys of the current batch, checked at each step
    orig_sel, orig_tail = new.selectBatch, new.moveToTail

    def sel(heap, A_m, legs, ls, K_):
        batch = orig_sel(heap, A_m, legs, ls, K_)
        sets = [new.touched(legs, i, j) for i, j in batch]
        for a in range(len(sets)):
            for b in range(a + 1, len(sets)):
                st["overlap"] += bool(sets[a] & sets[b])
        st["max_batch"] = max(st["max_batch"], len(batch))
        pending.clear()
        pending.extend(new.edgeKey(A_m, legs, ls, i, j) for i, j in batch)
        return batch

    def tail(A_m, legs, cano, i, j, *a):           # first call of every step: is (i, j) still valid?
        key = next(k for k in pending if k[-2:] == (i, j))
        st["invalid"] += not new.isValid(key, A_m, legs, LS[0])
        st["steps"] += 1
        return orig_tail(A_m, legs, cano, i, j, *a)

    # isValid needs ls: grab MPSCalculus' list through the selectBatch call
    LS = [None]
    def sel_ls(heap, A_m, legs, ls, K_):
        LS[0] = ls
        return sel(heap, A_m, legs, ls, K_)

    new.selectBatch, new.moveToTail = sel_ls, tail
    try:
        s = {}
        z, eps = new.MPSCalculus(*net(L, d), 50, 500, stats=s, K=K)
    finally:
        new.selectBatch, new.moveToTail = orig_sel, orig_tail
    return z, eps, s["rounds"], st


ok_all = True
for L, d, ref in CASES:
    z4, _ = old.MPSCalculus(*net(L, d), 50, 500)
    z1, _, r1, st1 = run(L, d, 1)
    same = complex(z1) == complex(z4)
    ok = same and st1["overlap"] == 0 and st1["invalid"] == 0
    line = f"L{L} d{d}: K=1 same as B4 {same}, rounds {r1}"
    for K in (4, None):
        z, eps, r, st = run(L, d, K)
        ok &= st["overlap"] == 0 and st["invalid"] == 0
        err = "" if ref is None else f" err {abs(float(np.real(z)) - ref):.1e}"
        line += (f" | K={K}: rounds {r} (steps/rounds {st['steps'] / r:.1f}, max batch {st['max_batch']})"
                 f" overlaps {st['overlap']} invalid {st['invalid']} eps {eps:.1e}{err}")
    ok_all &= ok
    print(line + f"  -> {'OK' if ok else 'FAIL'}")
print("C1 test:", "PASSED" if ok_all else "FAILED")
