### Circuit statistics (L = 10, n = 100, d = 8, seed = 0)

| Layer | Pattern | Dir. | CZ | T | X<sup>1/2</sup> | Y<sup>1/2</sup> | CZ across │ cut (cum.) | CZ across ─ cut (cum.) | Light cone (centre) | Light cone (avg.) |
|:---:|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | ↔ | 25 | 50 | 0 | 0 | 5 | 0 | 2/100 | 2 % |
| 2 | 2 | ↔ | 25 | 50 | 0 | 0 | 10 | 0 | 2/100 | 2 % |
| 3 | 3 | ↕ | 20 | 0 | 10 | 20 | 10 | 0 | 3/100 | 3 % |
| 4 | 4 | ↕ | 20 | 0 | 18 | 22 | 10 | 0 | 4/100 | 4 % |
| 5 | 5 | ↔ | 20 | 6 | 12 | 6 | 10 | 0 | 6/100 | 5 % |
| 6 | 6 | ↔ | 20 | 16 | 9 | 15 | 10 | 0 | 8/100 | 6 % |
| 7 | 7 | ↕ | 25 | 9 | 4 | 7 | 10 | 5 | 12/100 | 10 % |
| 8 | 8 | ↕ | 25 | 15 | 21 | 14 | 10 | 10 | 16/100 | 13 % |
| **Σ** |  |  | **180** | **146** | **74** | **84** | **10** | **10** |  |  |

*Table: CZ layers and gate counts per layer.*

| Quantity | Count |
|:---|---:|
| H | 200 |
| T | 146 |
| X^1/2 | 74 |
| Y^1/2 | 84 |
| CZ | 180 |
| **Total gates** | **684** |
| Tensors (kets + gates + bras) | 884 |
| Bonds (edges, dim 2) | 964 |

*Table: Total gate and tensor counts.*

| Indicator | Value | Meaning |
|:---|---:|:---|
| CZ across vertical cut │ | 10 | S ≤ min(10, 50) = 10 ebits |
| CZ across horizontal cut ─ | 10 | S ≤ min(10, 50) = 10 ebits |
| Light cone of centre qubit | 16/100 | qubits influenced after d layers |
| Avg. light cone | 13 % | average over all qubits |
| Treewidth upper bound (min-degree) | 9 | exact contraction cost ~ 2<sup>O(9)</sup> [[8]](#ref8) |

*Table: Indicators for high entanglement at d = 8.*
