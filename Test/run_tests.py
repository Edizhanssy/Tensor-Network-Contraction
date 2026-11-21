# Quick regression tests: python run_tests.py
import itertools
import numpy as np
from main import (lexicographicmapping, createIDs, defineTensors, modifyTensortoMPS,
                           mergeAdjacent, canonicalize, moveCenter, MPSCalculus)
from Test.test_mps import testMPS, checkLeftCanonical, testMergeAdjacent, testCanonicalize, mpsToTensor
from Test.kac_ward import kacWardLogZ
from TensorPrimitive.network import circuitToEdges, edgesToLegs, stateVectorAmplitude
from TensorPrimitive.circuits import H, T, Xh, Yh, CZ, randomCircuit


def build(rows, cols, K):
    a = np.sqrt(np.cosh(K) / 2.0)
    b = np.sqrt(np.sinh(K) / 2.0)
    B = np.array([[a + b, a - b], [a - b, a + b]])
    x, y = lexicographicmapping(rows, cols)
    legs = createIDs(x, y, rows, cols)
    return defineTensors(B, legs), legs


A, legs = build(4, 4, 0.1)
testMPS(A, modifyTensortoMPS(A))
checkLeftCanonical(modifyTensortoMPS(A)[5])
testMergeAdjacent(lambda A_m, site, p: mergeAdjacent(A_m, [0], site, p))
testCanonicalize(canonicalize)


def testMoveCenter(seed=2):
    """Center tracking: after moveCenter(.., p) the dense tensor is unchanged, every site left of p is a
    left isometry, every site right of p a right isometry, and ||A[p]|| = ||MPS||. Moves right AND left."""
    rng = np.random.default_rng(seed)
    mps = [rng.standard_normal(s) + 1j * rng.standard_normal(s)
           for s in [(1, 2, 3), (3, 2, 4), (4, 2, 3), (3, 2, 2), (2, 2, 1)]]
    before = mpsToTensor(mps)
    A_m, cano = [[m.copy() for m in mps]], [0]
    worst = 0.0
    for p in [4, 1, 3, 0, 2]:                       # first sweep to the end makes the gauge valid
        moveCenter(A_m, cano, 0, p)
        A = A_m[0]
        worst = max(worst, np.max(np.abs(mpsToTensor(A) - before)))
        for q in range(p):
            l, ph, r = A[q].shape
            M = A[q].reshape(l * ph, r)
            worst = max(worst, np.max(np.abs(M.conj().T @ M - np.eye(r))))
        for q in range(p + 1, len(A)):
            l, ph, r = A[q].shape
            M = A[q].reshape(l, ph * r)
            worst = max(worst, np.max(np.abs(M @ M.conj().T - np.eye(l))))
        worst = max(worst, abs(np.linalg.norm(A[p]) - np.linalg.norm(before)))
    print(f"testMoveCenter: max deviation over 5 moves = {worst:.1e}")
    return worst < 1e-12


testMoveCenter()

for (R, C), dhead in [((2, 3), 50), ((4, 4), 50), ((4, 4), 1), ((16, 16), 50), ((16, 16), 4)]:
    A, legs = build(R, C, 0.1)
    lnZ, err = MPSCalculus(A, legs, dhead, 500)
    ex = kacWardLogZ(R, C, 0.1)
    print(f"{R}x{C} dhead={dhead:>2}: rel.err(lnZ)={abs(lnZ - ex) / ex:.2e}  acc.err={err:.1e}")

# --- quantum circuits vs. state vector ---
circuit = [(H, [0]), (T, [1]), (CZ, [0, 1]), (Yh, [0]), (H, [1])]        # contains zero amplitudes
worst = 0
for x in itertools.product([0, 1], repeat=2):
    A, el = circuitToEdges(circuit, list(x))
    lnZ, _ = MPSCalculus(A, edgesToLegs(el), 50, 500)
    worst = max(worst, abs(np.exp(lnZ) - stateVectorAmplitude(circuit, list(x))))
print(f"2-qubit H/T/CZ/Yh circuit: max abs err = {worst:.1e}")

circuit = randomCircuit(L=3, d=6, seed=1)                                    # 9 qubits
worst = 0
for idx, x in enumerate(itertools.product([0, 1], repeat=9)):
    if idx % 37:                                                             # every 37th bitstring (14 of 512)
        continue
    A, el = circuitToEdges(circuit, list(x))
    lnZ, _ = MPSCalculus(A, edgesToLegs(el), 50, 500)
    worst = max(worst, abs(np.exp(lnZ) - stateVectorAmplitude(circuit, list(x))))
print(f"3x3 random circuit, d=6: max abs err = {worst:.1e}")
