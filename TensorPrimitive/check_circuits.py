# Compare circuits.randomCircuit with the authors' circuit files in the CATN repo.
# For each n it checks: CZ count, single-qubit gate count, and, qubit by qubit, that
# CZs (with the same partner) and single-qubit gates sit at the same positions.
# Gate TYPES (T / X1/2 / Y1/2 after the first T) are random, so they are not compared.
#   python check_circuits.py --catn /path/to/catn
import argparse, collections, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import circuits as C


def load_catn(prefix, n):
    """Per qubit: the wire's operations in order, ('CZ', (q1, q2)) or ('T'|'X'|'Y'|'H', None)."""
    labs = [list(map(int, l.split())) for l in open(prefix + ".labels.dat")]
    data = [np.array(list(map(float, l.split()))) for l in open(prefix + ".tensors.dat")]
    sizes = [list(map(int, l.split())) for l in open(prefix + ".sizes.dat")]
    where = collections.defaultdict(list)
    for t, l in enumerate(labs):
        for x in l:
            where[x].append(t)
    # a CZ is stored as two rank-3 tensors: a copy tensor and a copy tensor with a (-1) phase
    kind = {t: ("phase" if data[t][0::2][7] < 0 else "copy") for t, l in enumerate(labs) if len(l) == 3}
    link, cz = {}, {}
    for t, k in kind.items():
        if k == "phase":
            kl = labs[t][2]
            o = [u for u in where[kl] if u != t][0]
            link[t] = link[o] = kl
            cz[t] = cz[o] = (o, t)
    qubit_of, seq = {}, {}
    inputs = [t for t, l in enumerate(labs) if len(l) == 1][:n]
    for q, t0 in enumerate(inputs):                  # follow each wire from its input tensor
        prev, lab, s = t0, labs[t0][0], []
        while True:
            nxt = [u for u in where[lab] if u != prev][0]
            qubit_of[nxt] = q
            if len(labs[nxt]) == 1:
                break
            s.append(nxt)
            prev, lab = nxt, [x for x in labs[nxt] if x != lab and x != link.get(nxt)][0]
        seq[q] = s

    def gate(t):
        z = (data[t][0::2] + 1j * data[t][1::2]).reshape(sizes[t], order="F")
        m = z / z.flat[np.argmax(abs(z))]
        if abs(m[0, 1]) < 1e-9 and abs(m[1, 0]) < 1e-9:
            return "T"
        if abs(m[0, 1] + 1j) < 1e-3:
            return "X"
        if abs(m[1, 0] + 1) < 1e-3:
            return "Y"
        return "H"
    return {q: [("CZ", tuple(sorted(qubit_of[x] for x in cz[t]))) if t in cz else (gate(t), None) for t in s]
            for q, s in seq.items()}


def ours(L, seed):
    names = {id(C.T): "T", id(C.Xh): "X", id(C.Yh): "Y", id(C.H): "H"}
    w = collections.defaultdict(list)
    for g, q in C.randomCircuit(L, 8, seed):
        if len(q) == 2:
            e = tuple(sorted(q))
            w[q[0]].append(("CZ", e)); w[q[1]].append(("CZ", e))
        else:
            w[q[0]].append((names[id(g)], None))
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catn", required=True)
    a = ap.parse_args()
    shape = lambda s: [(g, e) if g in ("CZ", "H") else ("gate", None) for g, e in s]
    ok_all = True
    for n in (100, 400, 900, 1600):
        L = int(round(n ** 0.5))
        theirs = load_catn(os.path.join(a.catn, "graphs", f"supremacy2d{n}-d10-seed1-rank3-new"), n)
        mine = ours(L, seed=1)
        same = sum(shape(mine[q]) == shape(theirs[q]) for q in range(n))
        cz_m = sum(1 for s in mine.values() for g, _ in s if g == "CZ") // 2
        cz_t = sum(1 for s in theirs.values() for g, _ in s if g == "CZ") // 2
        g_m = sum(1 for s in mine.values() for g, _ in s if g in "TXY")
        g_t = sum(1 for s in theirs.values() for g, _ in s if g in "TXY")
        ok = same == n and cz_m == cz_t and g_m == g_t
        ok_all &= ok
        print(f"n = {n:4d}: CZ {cz_m} vs {cz_t}, single-qubit gates {g_m} vs {g_t}, "
              f"same positions on {same}/{n} qubits  {'OK' if ok else 'MISMATCH'}")
    print("all match" if ok_all else "MISMATCH")


if __name__ == "__main__":
    main()
