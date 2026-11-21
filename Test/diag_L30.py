# AI usage is avaliable for this particular code!

# Is ln|<0..0|C|0..0>| at L = 30 (d = 8) numerically well defined?
# Our exact runs (eps ~ 1e-26) disagree with CATN by ~31 in ln|amp|, and with each other.
# Hypothesis: the amplitude is ~e^-37 below a typical one (~1e-16, i.e. machine precision), so it is
# the result of near-complete cancellation and rounding decides the digits.
# Test 1: the same circuit and bitstring with several contraction orders (rng tie-breaking).
#         Well-conditioned -> all exact runs agree to ~1e-12. Ill-conditioned -> they scatter.
# Test 2: the same, for a few RANDOM bitstrings (typical amplitudes, no suppression).
#   python diag_L30.py              # L = 30 (each run ~45 s with B1 -> ~10 min)
#   python diag_L30.py 20           # cheaper check on L = 20
import sys, time
import numpy as np
from TensorPrimitive.circuits import randomCircuit
from TensorPrimitive.network import circuitToEdges, edgesToLegs
from main_redblack import MPSCalculus

L = int(sys.argv[1]) if len(sys.argv) > 1 else 30
n = L * L
circuit = randomCircuit(L, 8, seed=0)
typical = -0.5 * n * np.log(2)
print(f"L = {L}, d = 8, typical ln|amp| = -n/2 ln2 = {typical:.2f}")


def run(x, order_seed):
    A, el = circuitToEdges(circuit, x)
    rng = None if order_seed is None else np.random.default_rng(order_seed)
    st = {}
    t0 = time.time()
    lnZ, eps = MPSCalculus(A, edgesToLegs(el), 50, 500, rng=rng, stats=st)
    return float(np.real(lnZ)), float(np.imag(lnZ)), eps, st["hits_D"], time.time() - t0


bitstrings = [("0...0", [0] * n)]
gen = np.random.default_rng(123)
bitstrings += [(f"random #{k}", list(gen.integers(0, 2, n))) for k in range(2)]

for name, x in bitstrings:
    print(f"\n--- bitstring {name} ---")
    vals = []
    for os_ in [None, 0, 1, 2]:
        re, im, eps, hits, sec = run(x, os_)
        vals.append(re)
        tag = "greedy" if os_ is None else f"order {os_}"
        print(f"  {tag:8s}: ln|amp| = {re:12.6f}  (vs typical {re - typical:+8.2f})  phase {im:+.4f}  "
              f"eps {eps:.1e}  D_hat hits {hits}  {sec:.0f} s", flush=True)
    print(f"  spread of ln|amp| over orders: {max(vals) - min(vals):.2e}")
