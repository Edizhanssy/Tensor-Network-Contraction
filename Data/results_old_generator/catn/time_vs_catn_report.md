# Wall-clock time: our kernel (B4) vs. CATN

D̂ = 50, χ̂ = 500. Minimum over 3 runs, each in a fresh process, no profiler/tracemalloc. Circuit loading not timed; CATN's network setup is included.

| circuits | qubits | ours [s] | CATN [s] | CATN / ours | Δ ln\|amp\| |
|---|---|---|---|---|---|
| ours (d8) | 100 | 0.33 | 2.19 | 6.6× | 8.5e-14 |
| ours (d8) | 400 | 0.80 | 21.97 | 27.5× | 1.4e-13 |
| ours (d8) | 900 | 1.73 | 112.11 | 64.9× | 3.1e+01 |
| ours (d8) | 1600 | 2.96 | 423.78 | 143.0× | 2.7e+00 |
| authors (d10) | 100 | 0.35 | 2.76 | 7.8× | 5.0e-14 |
| authors (d10) | 400 | 0.65 | 34.48 | 52.9× | 2.3e-13 |
| authors (d10) | 900 | 1.61 | 194.47 | 120.8× | 4.1e-07 |
| authors (d10) | 1600 | 8.12 | – | – | – |

Δ ln|amp| is only meaningful for n ≤ 400 on our circuits (L30/L40: double-precision cancellation).
