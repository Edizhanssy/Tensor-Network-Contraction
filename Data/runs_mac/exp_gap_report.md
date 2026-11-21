# Accuracy gap on the authors' circuits: which hypothesis?

D_max = 50, chi_max = 500. err: ours = discarded weight, CATN = sum of discarded singular values. Δ = |ln|amp| − ln|amp| of CATN as published| (same n).

| n | config | time [s] | err | ln\|amp\| | Δ vs CATN ref |
|---|---|---|---|---|---|
| 100 | ours tol=1e-12 | 0.6 | 2.1e-30 | -34.5414627053 | 4.3e-14 |
| 100 | ours tol=1e-14 | 0.1 | 2.1e-30 | -34.5414627053 | 4.3e-14 |
| 100 | catn reverse=T cut=1e-14 | 1.6 | 7.7e-15 | -34.5414627053 | 0.0e+00 |
| 100 | catn reverse=F cut=1e-14 | 1.5 | 1.7e-14 | -34.5414627053 | 8.5e-14 |
| 100 | catn reverse=T cut=1e-12 | 1.5 | 7.7e-15 | -34.5414627053 | 0.0e+00 |
| 900 | ours tol=1e-12 | 0.9 | 7.7e-10 | -311.5506360098 | 4.1e-07 |
| 900 | ours tol=1e-14 | 0.8 | 7.7e-10 | -311.5506360098 | 4.1e-07 |
| 900 | ours tol=1e-12 compress | 1.1 | 7.7e-10 | -311.5506360098 | 4.1e-07 |
| 900 | catn reverse=T cut=1e-14 | 123.4 | 1.8e-13 | -311.5506356005 | 0.0e+00 |
| 900 | catn reverse=F cut=1e-14 | 115.6 | 4.2e-13 | -311.5506356005 | 3.4e-13 |
| 900 | catn reverse=T cut=1e-12 | 120.0 | 1.8e-13 | -311.5506356005 | 0.0e+00 |
| 1600 | ours tol=1e-12 | 3.6 | 7.6e-03 | -553.5571285231 | 1.9e-02 |
| 1600 | ours tol=1e-14 | 3.9 | 7.6e-03 | -553.5571285231 | 1.9e-02 |
| 1600 | ours tol=1e-12 compress | 4.7 | 7.6e-03 | -553.5571285231 | 1.9e-02 |
| 1600 | catn reverse=T cut=1e-14 | 471.8 | 2.3e-02 | -553.5383730790 | 0.0e+00 |
| 1600 | catn reverse=F cut=1e-14 | 538.2 | 6.6e-02 | -553.4780398407 | 6.0e-02 |
| 1600 | catn reverse=T cut=1e-12 | 487.9 | 2.3e-02 | -553.5383730790 | 0.0e+00 |
