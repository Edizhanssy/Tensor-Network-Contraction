# AI usage is avaliable for this particular code!

# Is eps_SVD a trustworthy error estimate? Compare MPS calculus against an exact state vector
# for a few chosen (L, d) runs -- also beyond SV_MAX_QUBITS of the sweep (n = 25 is ~0.5 GB).
#
#   python validate_sv.py 5 12 5 14          # pairs: L d  L d ...
#   python validate_sv.py 5 12 --dhead 100   # optional D_hat / chi_hat override
#
# Cost of the reference: memory 16 * 2^n bytes (x ~3 while gates are applied), time ~ (#gates) * 2^n.
import argparse, time
import numpy as np
from TensorPrimitive.circuits import randomCircuit
from TensorPrimitive.network import circuitToEdges, edgesToLegs, stateVectorAmplitude
from Source.main_redblack import MPSCalculus
from sweep_circuits import SEED, DHEAD, XHEAD

ap = argparse.ArgumentParser()
ap.add_argument("Ld", type=int, nargs="+", help="pairs: L d L d ...")
ap.add_argument("--dhead", type=int, default=DHEAD)
ap.add_argument("--xhead", type=int, default=XHEAD)
args = ap.parse_args()
assert len(args.Ld) % 2 == 0, "give L d pairs"

for L, d in zip(args.Ld[::2], args.Ld[1::2]):
    n = L * L
    print(f"\n=== L={L} (n={n}) d={d}  D_hat={args.dhead} chi_hat={args.xhead} "
          f"| state vector needs ~{3 * 16 * 2**n / 1e9:.1f} GB")
    circuit = randomCircuit(L, d, seed=SEED)
    x = [0] * n
    A, el = circuitToEdges(circuit, x)
    legs = edgesToLegs(el)

    t0 = time.time()
    lnZ, eps = MPSCalculus(A, legs, args.dhead, args.xhead)
    t_mps = time.time() - t0

    t0 = time.time()
    ref = stateVectorAmplitude(circuit, x)
    t_sv = time.time() - t0

    amp = np.exp(lnZ)
    rel = abs(amp - ref) / max(abs(ref), 1e-300)
    dlog = abs(np.real(lnZ) - np.log(abs(ref)))          # |Delta ln|amp||, no branch issues
    print(f"MPS : amp = {amp:.10e}   eps_SVD = {eps:.2e}   ({t_mps:.1f} s)")
    print(f"SV  : amp = {ref:.10e}   P = |amp|^2 = {abs(ref)**2:.3e} (uniform 1/2^n = {2.0**-n:.3e})   ({t_sv:.1f} s)")
    print(f"error: relative |amp_MPS - amp_SV| / |amp_SV| = {rel:.2e}   |Delta ln|amp|| = {dlog:.2e}")
