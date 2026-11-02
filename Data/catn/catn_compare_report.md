# Our kernel vs. CATN on identical circuits

err: ours = Σ discarded weight Σσ²/Σσ² (ε_SVD); CATN = Σ discarded singular values (absolute). Not the same measure; both are 0 when nothing is truncated.

## D̂ = 50, χ̂ = 500

| circuits | qubits | code | status | time [s] | peak [MB] | err | ln\|amp\| | Δ ln\|amp\| ours vs CATN |
|---|---|---|---|---|---|---|---|---|
| authors (d10 = 1+8+1) | 100 | ours | ok | 20.1 | 14.6 | 1.2e-29 | -34.541463 | 1.4e-14 |
| authors (d10 = 1+8+1) | 100 | catn | ok | 21.7 | 27.9 | 8.9e-15 | -34.541463 |  |
| authors (d10 = 1+8+1) | 400 | ours | ok | 347.6 | 18.9 | 2.0e-07 | -140.137640 | 9.8e-04 |
| authors (d10 = 1+8+1) | 400 | catn | ok | 312.7 | 37.8 | 7.8e-14 | -140.138617 |  |
| authors (d10 = 1+8+1) | 900 | ours | ok | 1677.2 | 97.2 | 8.2e-04 | -311.538290 | 1.2e-02 |
| authors (d10 = 1+8+1) | 900 | catn | ok | 1634.0 | 54.7 | 1.7e-13 | -311.550636 |  |
| authors (d10 = 1+8+1) | 1600 | ours | timeout | 3600.0 | – | – | – | – |
| authors (d10 = 1+8+1) | 1600 | catn | timeout | 3600.0 | – | – | – |  |
| ours (d = 8) | 100 | ours | ok | 20.8 | 18.2 | 4.5e-27 | -37.588259 | 2.8e-13 |
| ours (d = 8) | 100 | catn | ok | 15.7 | 27.1 | 1.2e-13 | -37.588259 |  |
| ours (d = 8) | 400 | ours | ok | 341.7 | 175.7 | 1.8e-02 | -146.577226 | 8.1e-02 |
| ours (d = 8) | 400 | catn | ok | 139.2 | 36.6 | 5.2e-13 | -146.496715 |  |
| ours (d = 8) | 900 | ours | ok | 1760.6 | 583.8 | 5.9e-01 | -383.802824 | 3.5e+01 |
| ours (d = 8) | 900 | catn | ok | 501.1 | 52.8 | 1.6e-12 | -348.830739 |  |
| ours (d = 8) | 1600 | ours | timeout | 3600.0 | – | – | – | – |
| ours (d = 8) | 1600 | catn | ok | 1768.3 | 75.2 | 3.0e-12 | -663.429247 |  |

