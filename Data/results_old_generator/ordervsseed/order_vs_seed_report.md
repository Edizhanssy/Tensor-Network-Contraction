# Seed vs. order: what makes a circuit hard for MPS calculus?

D̂ = 50, χ̂ = 500. Experiment 1 'seed': circuit seeds 0..9, deterministic greedy order. Experiment 2 'order': circuit seed 0, random tie-breaking with order seeds 0..9. 'exact' = ε_SVD < 1e-10.

| L | d | experiment | runs | exact | ε of truncated runs: min / median / max | time [s] exact runs | time [s] truncated runs |
|---|---|---|---|---|---|---|---|
| 5 | 12 | seed | 10 | 7 | 3.1e-02 / 4.2e-02 / 6.4e-02 | 0.9–1.7 | 1.5–1.9 |
| 5 | 12 | order | 10 | 9 | 4.7e-02 / 4.7e-02 / 4.7e-02 | 0.5–2.1 | 2.5–2.5 |
| 5 | 14 | seed | 10 | 10 | – | 0.9–1.4 | – |
| 5 | 14 | order | 10 | 10 | – | 0.7–2.2 | – |
| 6 | 14 | seed | 10 | 0 | 8.7e-03 / 2.9e-02 / 3.6e-01 | – | 10.2–17.6 |
| 6 | 14 | order | 10 | 5 | 1.1e-02 / 4.3e-02 / 2.2e-01 | 1.8–7.3 | 4.5–18.3 |
| 6 | 16 | seed | 10 | 2 | 2.4e-02 / 2.1e-01 / 5.4e-01 | 4.7–6.5 | 9.7–63.9 |
| 6 | 16 | order | 10 | 2 | 1.4e-02 / 1.0e-01 / 9.8e-01 | 3.9–5.4 | 5.9–63.3 |

## Is ε_SVD a good error estimate?

Every circuit that some run contracted exactly gives a reference value of ln|amp| (all exact runs of the same circuit agree to ~1e-13). Truncated runs of such circuits:

| L | d | experiment | circuit | order | ε_SVD | true abs(Δ ln abs(amp)) | ratio true / ε |
|---|---|---|---|---|---|---|---|
| 5 | 12 | seed | 0 | greedy | 6.4e-02 | 6.9e-01 | 10.8 |
| 5 | 12 | order | 0 | 0 | 4.7e-02 | 1.8e-01 | 3.9 |
| 6 | 14 | seed | 0 | greedy | 2.8e-02 | 4.0e-01 | 14.3 |
| 6 | 14 | order | 0 | 0 | 1.1e-02 | 2.0e-01 | 18.5 |
| 6 | 14 | order | 0 | 2 | 4.3e-02 | 6.6e-02 | 1.5 |
| 6 | 14 | order | 0 | 3 | 1.1e-02 | 2.4e-02 | 2.2 |
| 6 | 14 | order | 0 | 5 | 2.2e-01 | 7.8e-01 | 3.6 |
| 6 | 14 | order | 0 | 9 | 7.7e-02 | 2.0e-01 | 2.6 |
| 6 | 16 | seed | 0 | greedy | 3.5e-01 | 3.9e-01 | 1.1 |
| 6 | 16 | order | 0 | 0 | 6.4e-02 | 2.3e-01 | 3.5 |
| 6 | 16 | order | 0 | 1 | 1.5e-01 | 7.5e-02 | 0.5 |
| 6 | 16 | order | 0 | 2 | 2.3e-01 | 3.5e-01 | 1.5 |
| 6 | 16 | order | 0 | 3 | 4.6e-02 | 3.0e-02 | 0.6 |
| 6 | 16 | order | 0 | 4 | 1.7e-01 | 8.8e-02 | 0.5 |
| 6 | 16 | order | 0 | 6 | 4.2e-02 | 2.4e-01 | 5.6 |
| 6 | 16 | order | 0 | 7 | 1.4e-02 | 3.5e-02 | 2.4 |
| 6 | 16 | order | 0 | 8 | 9.8e-01 | 1.4e+00 | 1.4 |

17 truncated runs with a reference: true error / ε between 0.5 and 18.5 (median 2.4).

![seed vs order](order_vs_seed.png)

