# MPS calculus on random quantum circuits: sweep report

Generated 2026-10-22T10:57:07. Raw data: `circuits_sweep.csv`.

## 1. Setup

| Setting | Value |
|---|---|
| Max physical bond dimension D̂ | 50 |
| Max virtual bond dimension χ̂ | 500 |
| Time limit per run | 3600 s |
| Memory limit (stops a sweep) | 56 GB |
| Circuit | Pan et al. protocol: H layer, d CZ layers (Fig. 13 layouts, order 1,3,5,7,2,4,6,8), random T / X^½ / Y^½ on idle qubits, H layer; seed 0 |
| Quantity | single amplitude ⟨0…0│C│0…0⟩ |
| State-vector check | for n ≤ 16 qubits |

## 2. Machine

| Item | Value |
|---|---|
| os | Windows 10 |
| machine | AMD64 |
| cpu | Intel64 Family 6 Model 183 Stepping 1, GenuineIntel |
| cores_logical | 24 |
| python | 3.10.4 |
| numpy | 2.2.6 |
| cores_physical | 16 |
| ram_GB | 68.5 |
| blas | scipy-openblas 0.3.29 |
| thread_env | {'OMP_NUM_THREADS': '24'} |
| session 1 | 2026-10-22T07:32:18 → interrupted |
| session 2 | 2026-10-22T10:52:49 → 2026-10-22T10:53:48 |

## 3. Sweep A: growing lattice at depth 8

| L | qubits | gates | tensors | status | time [s] | peak mem [MB] | ε_SVD | quality | rel. err vs state vector |
|---|---|---|---|---|---|---|---|---|---|
| 4 | 16 | 124 | 156 | ok | 0.2 | 0.2 | 1.8e-30 | exact | 2.9e-15 |
| 5 | 25 | 190 | 240 | ok | 0.6 | 0.3 | 1.3e-29 | exact | – |
| 6 | 36 | 270 | 342 | ok | 1.2 | 0.5 | 6.7e-29 | exact | – |
| 8 | 64 | 472 | 600 | ok | 3.8 | 0.7 | 1.6e-28 | exact | – |
| 10 | 100 | 730 | 930 | ok | 9.5 | 5.2 | 4.5e-27 | exact | – |
| 12 | 144 | 1044 | 1332 | ok | 19.7 | 5.6 | 6.7e-28 | exact | – |
| 14 | 196 | 1414 | 1806 | ok | 40.6 | 9.3 | 7.8e-26 | exact | – |
| 16 | 256 | 1840 | 2352 | ok | 64.9 | 21.3 | 1.9e-27 | exact | – |
| 20 | 400 | 2860 | 3660 | ok | 164.8 | 160.8 | 1.8e-02 | UNRELIABLE | – |
| 24 | 576 | 4104 | 5256 | ok | 358.7 | 467.5 | 4.7e-01 | UNRELIABLE | – |
| 28 | 784 | 5572 | 7140 | ok | 742.0 | 277.8 | 1.2e-02 | UNRELIABLE | – |
| 32 | 1024 | 7264 | 9312 | ok | 1087.9 | 977.2 | 1.6e-02 | UNRELIABLE | – |
| 36 | 1296 | – | – | crash | – | – | – | – | – |

- Largest circuit completed: **1024 qubits** (L = 32) in 1087.9 s, peak 977.2 MB.
- Empirical time scaling over this range: time ∝ n^2.04.
- Stop point: L = 36: crash (numpy.linalg.LinAlgError: SVD did not converge).

## 4. Sweep B: growing depth at fixed lattice

### L = 5 (25 qubits)

| depth | gates | status | time [s] | peak mem [MB] | ε_SVD | quality | rel. err vs state vector |
|---|---|---|---|---|---|---|---|
| 8 | 190 | ok | 0.6 | 0.3 | 1.3e-29 | exact | – |
| 10 | 220 | ok | 0.8 | 0.6 | 9.1e-29 | exact | – |
| 12 | 250 | ok | 2.5 | 41.3 | 6.4e-02 | UNRELIABLE | – |
| 14 | – | crash | – | – | – | – | – |

- Deepest circuit completed: d = 12.
- First depth with ε_SVD ≥ 1e-3 (truncation no longer negligible): d = 12.
- Stop point: d = 14: crash (numpy.linalg.LinAlgError: SVD did not converge).

### L = 6 (36 qubits)

| depth | gates | status | time [s] | peak mem [MB] | ε_SVD | quality | rel. err vs state vector |
|---|---|---|---|---|---|---|---|
| 8 | 270 | ok | 1.2 | 0.5 | 6.7e-29 | exact | – |
| 10 | 312 | ok | 1.9 | 5.6 | 3.2e-28 | exact | – |
| 12 | 354 | ok | 3.4 | 9.7 | 7.8e-28 | exact | – |
| 14 | 411 | ok | 13.8 | 179.2 | 2.8e-02 | UNRELIABLE | – |
| 16 | 468 | ok | 30.5 | 935.3 | 3.5e-01 | UNRELIABLE | – |
| 18 | – | timeout | 3600.0 | – | – | – | – |

- Deepest circuit completed: d = 16.
- First depth with ε_SVD ≥ 1e-3 (truncation no longer negligible): d = 14.
- Stop point: d = 18: timeout (exceeded 3600 s).

### L = 8 (64 qubits)

| depth | gates | status | time [s] | peak mem [MB] | ε_SVD | quality | rel. err vs state vector |
|---|---|---|---|---|---|---|---|
| 8 | 472 | ok | 3.7 | 0.7 | 1.6e-28 | exact | – |
| 10 | 544 | ok | 6.7 | 12.2 | 6.0e-03 | UNRELIABLE | – |
| 12 | 616 | ok | 766.9 | 5775.6 | 4.4e+00 | UNRELIABLE | – |
| 14 | 716 | ok | 195.5 | 4511.3 | 8.8e-01 | UNRELIABLE | – |
| 16 | 816 | ok | 1533.0 | 11866.8 | 2.4e+00 | UNRELIABLE | – |

- Deepest circuit completed: d = 16.
- First depth with ε_SVD ≥ 1e-3 (truncation no longer negligible): d = 10.
- Stop point: d = 18: not run: sweep was interrupted before this point.

## 5. Validation against the state vector

1 runs with n ≤ 16 qubits were checked against an independent state-vector simulation. Largest relative error: **2.9e-15**.

## 6. Figure

![sweep](circuits_sweep.png)

## 7. How to read this

- **ε_SVD** is the accumulated discarded weight Σδ of all truncations (swaps and D̂ cuts). It is an a-priori estimate, not a bound; on the 16×16 Ising benchmark it tracked the true error within roughly one order of magnitude. 'exact' < 1e-10, 'reliable' < 1e-3. Because it is a sum over many truncations it can exceed 1; values that large only say that truncation dominated the run, not by how much the amplitude is off.
- **Peak memory** is measured with `tracemalloc` (all Python and NumPy allocations of the run); the interpreter's own baseline is not included.
- **Time** is wall-clock for `MPSCalculus` only (building the network is excluded). For crashed runs it is the wall time until the crash, including start-up; their peak memory is unknown.
- Each point is a single random circuit (one seed) and a single amplitude.
- The CZ layout order is not specified in the paper; the order used here alternates horizontal and vertical layouts so that consecutive layers never act on the same pair.
