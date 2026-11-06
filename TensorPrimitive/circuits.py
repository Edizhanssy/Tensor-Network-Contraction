import numpy as np

H  = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
T  = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)
Xh = 0.5 * np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]], dtype=complex)
Yh = 0.5 * np.array([[1 + 1j, -1 - 1j], [1 + 1j, 1 + 1j]], dtype=complex)
CZ = np.diag([1, 1, 1, -1]).astype(complex).reshape(2, 2, 2, 2)


def czPattern(k, L):
    """Qubit pairs that get a CZ in layout k = 1..8 (Fig. 13), qubit q = L*r + c."""
    pairs = []
    for r in range(L):
        for c in range(L):
            if k in (1, 2, 5, 6):                       # horizontal: (r,c)-(r,c+1)
                if c + 1 >= L:
                    continue
                if k == 1: ok = c % 2 == 0              # dense, even offset
                if k == 5: ok = c % 2 == 1              # dense, odd offset
                if k == 2: ok = c % 4 == (2 * r) % 4    # sparse, shifts every row
                if k == 6: ok = c % 4 == (2 * r + 1) % 4
                if ok:
                    pairs.append((L * r + c, L * r + c + 1))
            else:                                       # vertical: (r,c)-(r+1,c)
                if r + 1 >= L:
                    continue
                if k == 3: ok = r % 2 == 1              # dense, odd rows
                if k == 7: ok = r % 2 == 0              # dense, even rows
                if k == 4: ok = r % 2 == 1 and c % 2 == (r // 2) % 2   # sparse
                if k == 8: ok = r % 2 == 0 and c % 2 == (r // 2) % 2
                if ok:
                    pairs.append((L * r + c, L * (r + 1) + c))
    return pairs


def randomCircuit(L, d, seed=0):
    """Random circuit following the paper's protocol (Supplemental Material)."""
    rng = np.random.default_rng(seed)
    n = L * L
    circuit = [(H, [q]) for q in range(n)]              # H on every qubit
    order = [1, 3, 5, 7, 2, 4, 6, 8]                    # alternate horizontal / vertical
    for layer in range(d):
        pairs = czPattern(order[layer % 8], L)
        has_cz = [False] * n
        for a, b in pairs:
            circuit.append((CZ, [a, b]))
            has_cz[a] = True
            has_cz[b] = True
        for q in range(n):
            if not has_cz[q]:
                g = [T, Xh, Yh][rng.integers(3)]
                circuit.append((g, [q]))
    circuit += [(H, [q]) for q in range(n)]             # H on every qubit
    return circuit
