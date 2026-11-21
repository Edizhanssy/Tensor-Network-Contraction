# Disconnected tensor networks: the amplitude is the product of the components' scalars.
# Compares main, main_parallel and main_redblack with a direct np.einsum contraction.
#   python test_components_mac.py
import copy, string
import numpy as np
import main, Source.main_parallel, Source.main_redblack


def rand_net(edges, n, seed):
    rng = np.random.default_rng(seed)
    legs = [[] for _ in range(n)]
    for a, b in edges:
        legs[a].append(b); legs[b].append(a)
    A = [rng.standard_normal([2] * len(l)) + 1j * rng.standard_normal([2] * len(l)) for l in legs]
    return A, legs


def exact(A, legs, edges):
    letters = iter(string.ascii_letters)
    lab = {(min(a, b), max(a, b)): next(letters) for a, b in edges}
    subs = ["".join(lab[(min(k, x), max(k, x))] for x in l) for k, l in enumerate(legs)]
    return np.einsum(",".join(subs) + "->", *A)


def grid(R, C, o=0):
    return ([(o + r * C + c, o + r * C + c + 1) for r in range(R) for c in range(C - 1)] +
            [(o + r * C + c, o + (r + 1) * C + c) for r in range(R - 1) for c in range(C)])


cases = {"connected 3x3": (grid(3, 3), 9),
         "two parts 2x2 + 2x3": (grid(2, 2) + grid(2, 3, 4), 10),
         "three parts": (grid(2, 2) + grid(1, 3, 4) + grid(2, 2, 7), 11)}
variants = [("main, id", main, {}), ("main, fifo", main, {"tiebreak": "fifo"}),
            ("main_parallel, K=1", main_parallel, {}), ("main_parallel, K=None", main_parallel, {"K": None}),
            ("main_redblack, K=1", main_redblack, {}), ("main_redblack, K=None", main_redblack, {"K": None})]
worst = 0.0
for name, (edges, n) in cases.items():
    for seed in range(3):
        A, legs = rand_net(edges, n, seed)
        ex = exact(A, legs, edges)
        for vname, mod, kw in variants:
            lnZ, _ = mod.MPSCalculus([a.copy() for a in A], copy.deepcopy(legs), 50, 500, **kw)
            worst = max(worst, abs(np.exp(lnZ) - ex) / abs(ex))
    print(f"{name:22s}: worst relative error so far {worst:.1e}")
print("OK" if worst < 1e-10 else "FAILED")
