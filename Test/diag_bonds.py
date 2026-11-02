# Which limit is hit, D_hat or chi_hat?  Runs the current kernel (main_redblack.py) with the truncation split.
#   python diag_bonds.py            # L = 20, 30, 40 at d = 8 (about 20 min in total with A2)
#   python diag_bonds.py 30         # only L = 30
import sys, time
import numpy as np
from TensorPrimitive.circuits import randomCircuit
from TensorPrimitive.network import circuitToEdges, edgesToLegs
from main_redblack import MPSCalculus

DHEAD, XHEAD = 50, 500
REF = {20: -146.4967154741864, 30: -348.830739446184, 40: -663.4292474226539}   # exact (CATN)
Ls = [int(a) for a in sys.argv[1:]] or [20, 30, 40]

print(f"D_hat = {DHEAD}, chi_hat = {XHEAD}")
print("| L | time [s] | ε from χ̂ (swaps) | χ̂ hit | max χ kept | ε from D̂ (bond cuts) | D̂ hit | max D kept | true err |")
print("|---|---|---|---|---|---|---|---|---|")
for L in Ls:
    A, el = circuitToEdges(randomCircuit(L, 8, seed=0), [0] * (L * L))
    st = {}
    t0 = time.time()
    lnZ, eps = MPSCalculus(A, edgesToLegs(el), DHEAD, XHEAD, stats=st)
    err = abs(float(np.real(lnZ)) - REF[L]) if L in REF else float("nan")
    print(f"| {L} | {time.time() - t0:.1f} | {st['eps_chi']:.1e} | {st['hits_chi']} | {st['max_chi']} | "
          f"{st['eps_D']:.1e} | {st['hits_D']} | {st['max_D']} | {err:.1e} |", flush=True)
