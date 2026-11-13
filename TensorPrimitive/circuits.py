import numpy as np

H  = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
T  = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)
Xh = 0.5 * np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]], dtype=complex)
Yh = 0.5 * np.array([[1 + 1j, -1 - 1j], [1 + 1j, 1 + 1j]], dtype=complex)
CZ = np.diag([1, 1, 1, -1]).astype(complex).reshape(2, 2, 2, 2)


def czPattern(k, L):
    """Qubit pairs that get a CZ in layout k = 1..8 (Fig. 13), qubit q = L*r + c.
    Every layout is sparse: over the 8 layouts each grid edge gets exactly one CZ."""
    pairs = []
    for r in range(L):
        for c in range(L):
            if k in (1, 2, 5, 6):                       # horizontal: (r,c)-(r,c+1)
                if c + 1 >= L:
                    continue
                shift = {2: 0, 6: 1, 1: 2, 5: 3}[k]     # pair start moves by 2 every row
                if c % 4 == (2 * r + shift) % 4:
                    pairs.append((L * r + c, L * r + c + 1))
            else:                                       # vertical: (r,c)-(r+1,c)
                if r + 1 >= L:
                    continue
                row_parity = 1 if k in (3, 4) else 0    # layouts 3, 4 start in odd rows, 7, 8 in even rows
                col_shift = 1 if k in (3, 8) else 0     # column pattern flips every second row pair
                if r % 2 == row_parity and c % 2 == (r // 2 + col_shift) % 2:
                    pairs.append((L * r + c, L * (r + 1) + c))
    return pairs


def randomCircuit(L, d, seed=0):
    """Random circuit as in the authors' circuit files (CATN repo, Fig. 13 layouts):
    1. H on every qubit.
    2. Layer t applies the CZs of layout 1, 2, ..., 8 (then repeats).
    3. A qubit without a CZ in this layer gets a single-qubit gate only if this is the
       first layer or it had a CZ in the previous layer. Its first gate is T; later
       gates are chosen at random from {T, X^1/2, Y^1/2}, never the same as its previous gate.
    4. H on every qubit."""
    rng = np.random.default_rng(seed)
    n = L * L
    circuit = [(H, [q]) for q in range(n)]              # H on every qubit
    gates = {"T": T, "X": Xh, "Y": Yh}
    last_gate = [None] * n                              # last single-qubit gate (after the first H)
    had_cz = [False] * n                                # CZ in the previous layer?
    for layer in range(d):
        k = layer % 8 + 1                               # layouts in the order 1, 2, ..., 8
        has_cz = [False] * n
        for a, b in czPattern(k, L):
            circuit.append((CZ, [a, b]))
            has_cz[a] = True
            has_cz[b] = True
        for q in range(n):
            if has_cz[q] or not (layer == 0 or had_cz[q]):
                continue
            if last_gate[q] is None:
                name = "T"
            else:
                options = [g for g in ("T", "X", "Y") if g != last_gate[q]]
                name = options[rng.integers(2)]
            circuit.append((gates[name], [q]))
            last_gate[q] = name
        had_cz = has_cz
    circuit += [(H, [q]) for q in range(n)]             # H on every qubit
    return circuit
