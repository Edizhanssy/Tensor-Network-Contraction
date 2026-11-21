# C3: real threads vs. the ideal (C2)

Minimum wall time over 3 runs. inter = steps of a batch on P Python threads (BLAS 1 thread). intra = serial steps, BLAS on B threads. speedup vs 'serial K=None' for inter, vs 'serial K=1' for intra. same = identical ln|amp| as the serial run with the same K.

| circuit | config | time [s] | speedup | same result |
|---|---|---|---|---|
| ours L40 | serial K=1 | 2.89 | 1.00× | – |
| ours L40 | serial K=None | 3.01 | 1.00× | – |
| ours L40 | inter P=2 | 3.88 | 0.78× | yes |
| ours L40 | inter P=4 | 3.82 | 0.79× | yes |
| ours L40 | inter P=8 | 3.99 | 0.75× | yes |
| ours L40 | inter P=24 | 4.01 | 0.75× | yes |
| ours L40 | intra BLAS=4 | 2.95 | 0.98× | Δ 1.1e+00 |
| ours L40 | intra BLAS=24 | 2.99 | 0.97× | Δ 1.1e+00 |
| authors n=900 | serial K=1 | 1.10 | 1.00× | – |
| authors n=900 | serial K=None | 1.12 | 1.00× | – |
| authors n=900 | inter P=2 | 1.34 | 0.83× | yes |
| authors n=900 | inter P=4 | 1.40 | 0.80× | yes |
| authors n=900 | inter P=8 | 1.49 | 0.75× | yes |
| authors n=900 | inter P=24 | 1.50 | 0.74× | yes |
| authors n=900 | intra BLAS=4 | 1.19 | 0.92× | Δ 5.7e-14 |
| authors n=900 | intra BLAS=24 | 1.30 | 0.85× | yes |
| authors n=1600 | serial K=1 | 4.01 | 1.00× | – |
| authors n=1600 | serial K=None | 5.38 | 1.00× | – |
| authors n=1600 | inter P=2 | 5.82 | 0.92× | yes |
| authors n=1600 | inter P=4 | 5.99 | 0.90× | yes |
| authors n=1600 | inter P=8 | 5.99 | 0.90× | yes |
| authors n=1600 | inter P=24 | 6.02 | 0.89× | yes |
| authors n=1600 | intra BLAS=4 | 4.64 | 0.87× | Δ 1.1e-13 |
| authors n=1600 | intra BLAS=24 | 7.10 | 0.57× | Δ 1.1e-13 |
