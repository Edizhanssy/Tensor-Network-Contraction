# The amplitude of a random L x L circuit of depth d!

#   python example_circuit.py            # L=10, d=8
#   python example_circuit.py 30 8       # n=900, truncates at D_max=50
import sys, time
from TensorPrimitive.circuits import randomCircuit
from TensorPrimitive.network import circuitToEdges, edgesToLegs
from main_heap import MPSCalculus

L = int(sys.argv[1]) if len(sys.argv) > 1 else 10
d = int(sys.argv[2]) if len(sys.argv) > 2 else 8
D_MAX, CHI_MAX = 50, 500

circuit = randomCircuit(L, d, seed=0)
A, edges = circuitToEdges(circuit, [0] * (L * L))
legs = edgesToLegs(edges)

t0 = time.time()
ln_amp, eps = MPSCalculus(A, legs, D_MAX, CHI_MAX)
print(f"n = {L*L}, depth = {d}, D_max = {D_MAX}, chi_max = {CHI_MAX}")
print(f"ln|amp| = {ln_amp.real:.10f}   eps_SVD = {eps:.2e}   ({time.time()-t0:.2f} s)")