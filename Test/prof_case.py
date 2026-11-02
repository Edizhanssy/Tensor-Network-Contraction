# Profile the current kernel (main_redblack.py) on one circuit and write the top functions to a text file.
#   python prof_case.py 30 8        # L = 30, d = 8   -> profile_L30_d8.txt
#   python prof_case.py 8 14        # L = 8,  d = 14  -> profile_L8_d14.txt
import cProfile, io, pstats, sys, time
import numpy as np
from TensorPrimitive.circuits import randomCircuit
from TensorPrimitive.network import circuitToEdges, edgesToLegs
from main_redblack import MPSCalculus

L, d = int(sys.argv[1]), int(sys.argv[2])
A, el = circuitToEdges(randomCircuit(L, d, seed=0), [0] * (L * L))
legs = edgesToLegs(el)
st = {}
prof = cProfile.Profile()
t0 = time.time()
prof.enable()
lnZ, eps = MPSCalculus(A, legs, 50, 500, stats=st)
prof.disable()
wall = time.time() - t0

out = io.StringIO()
out.write(f"L={L} d={d}  tensors={len(A)}  wall (with profiler) = {wall:.1f} s  eps = {eps:.2e}  "
          f"ln|amp| = {float(np.real(lnZ)):.6f}\n")
out.write(f"bonds: max chi = {st['max_chi']}, max D = {st['max_D']}, chi_hat hits = {st['hits_chi']}, "
          f"D_hat hits = {st['hits_D']}\n\n")
for key, title in (("tottime", "own time"), ("cumulative", "cumulative time")):
    out.write(f"===== sorted by {title} =====\n")
    pstats.Stats(prof, stream=out).sort_stats(key).print_stats(15)
fname = f"profile_L{L}_d{d}.txt"
with open(fname, "w") as f:
    f.write(out.getvalue())
print(out.getvalue().splitlines()[0])
print("saved", fname)
