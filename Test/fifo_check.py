# Tie-break rule "id" (B3) vs "fifo" (CATN's rule) on many circuits, D_max = 50, chi_max = 500.
#   python fifo_check.py --catn /path/to/catn          # authors' n = 900 and 1600, seeds 1-10, + ours L6 d16
# One line per circuit: eps and number of D_max hits for both rules, and which one is smaller.
import argparse, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.dirname(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)
import numpy as np


def run(A, legs, tb):
    import main
    st = {}
    lnZ, eps = main.MPSCalculus(A, legs, 50, 500, stats=st, tiebreak=tb)
    return eps, st["hits_D"], float(np.real(lnZ))


def compare(label, build):
    t0 = time.perf_counter()
    e0, h0, _ = run(*build(), "id")                    # fresh network for every run: legs change in place
    e1, h1, _ = run(*build(), "fifo")
    win = "same" if max(e0, e1) < 1e-20 or abs(np.log10(max(e0, 1e-30) / max(e1, 1e-30))) < 0.3 else \
          ("fifo" if e1 < e0 else "id")
    print(f"{label:28s} id: eps {e0:9.2e} hits {h0:3d}   fifo: eps {e1:9.2e} hits {h1:3d}   better: {win:4s}"
          f"  ({time.perf_counter() - t0:.0f} s)", flush=True)
    return win


def main_():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catn", required=True)
    catn = os.path.abspath(ap.parse_args().catn)
    import catn_compare as cc
    tally = {}
    for n in (900, 1600):
        for seed in range(1, 11):
            name = f"supremacy2d{n}-d10-seed{seed}-rank3-new"
            w = compare(f"authors n={n} seed={seed}", lambda: cc.build("pz", name, catn))
            tally.setdefault(f"authors n={n}", []).append(w)
    for seed in range(10):
        w = compare(f"ours L6 d16 seed={seed}", lambda: cc.build("ours", f"6:16:{seed}", catn))
        tally.setdefault("ours L6 d16", []).append(w)
    print()
    for k, ws in tally.items():
        print(f"{k:16s} fifo better {ws.count('fifo'):2d}   id better {ws.count('id'):2d}   same {ws.count('same'):2d}")


if __name__ == "__main__":
    main_()
