# Stage 3: exact references by raising D_max. For every circuit our kernel runs at several D_max
# (chi_max fixed); a run with eps_SVD ~ 0 (no truncation at all) is an exact reference, and every
# truncated run can then be scored by its TRUE error |ln|amp| - ln|amp|_ref|.
#   python stage3_reference.py --catn catn                       # all circuits, D_max 50 64 100 128 200
#   python stage3_reference.py --catn catn --dheads 50,128 --only ours
# Writes Data/validation/3_dmax_reference.csv (resumes: finished rows are skipped).
import argparse, csv, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.join(HERE, "TensorPrimitive"), os.path.join(HERE, "Test")):
    if p not in sys.path:
        sys.path.insert(0, p)
import numpy as np

XHEAD = 500
CIRCUITS = ([("ours", f"{L}:8:0") for L in (10, 20, 30, 40)] +
            [("pz", f"supremacy2d{n}-d10-seed1-rank3-new") for n in (100, 400, 900, 1600)])
OUT = os.path.join(HERE, "Data", "validation", "3_dmax_reference.csv")
FIELDS = ["family", "circuit", "n_qubits", "dmax", "eps", "eps_chi", "eps_D", "hits_chi", "hits_D",
          "max_chi", "max_D", "ln_abs_amp", "seconds"]
EXACT = 1e-20                                           # eps below this: nothing was truncated


def n_of(family, circ):
    return int(circ.split("supremacy2d")[1].split("-")[0]) if family == "pz" else int(circ.split(":")[0]) ** 2


def run(family, circ, dmax, catn_dir):
    import catn_compare as cc
    from main_redblack import MPSCalculus
    A, legs = cc.build(family, circ, catn_dir)          # fresh network: MPSCalculus changes legs in place
    st = {}
    t0 = time.perf_counter()
    lnZ, eps = MPSCalculus(A, legs, dmax, XHEAD, stats=st)
    return dict(eps=eps, ln_abs_amp=float(np.real(lnZ)), seconds=time.perf_counter() - t0, **st)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catn", required=True)
    ap.add_argument("--dheads", default="50,64,100,128,200")
    ap.add_argument("--only", choices=["ours", "pz"])
    a = ap.parse_args()
    catn_dir = os.path.abspath(a.catn)
    dheads = [int(x) for x in a.dheads.split(",")]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    done = set()
    if os.path.exists(OUT):
        done = {(r["family"], r["circuit"], r["dmax"]) for r in csv.DictReader(open(OUT))}
    for family, circ in CIRCUITS:
        if a.only and family != a.only:
            continue
        for dmax in dheads:
            if (family, circ, str(dmax)) in done:
                continue
            r = run(family, circ, dmax, catn_dir)
            row = dict(family=family, circuit=circ, n_qubits=n_of(family, circ), dmax=dmax, **r)
            new = not os.path.exists(OUT)
            with open(OUT, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
                if new:
                    w.writeheader()
                w.writerow(row)
            tag = "exact" if r["eps"] < EXACT else f"hits_D={r['hits_D']} hits_chi={r['hits_chi']}"
            print(f"{family:4s} n={row['n_qubits']:5d} D_max={dmax:4d}: eps={r['eps']:9.2e}  "
                  f"ln|amp|={r['ln_abs_amp']:18.10f}  max_D={r['max_D']:4d} max_chi={r['max_chi']:4d}  "
                  f"{r['seconds']:7.1f} s  {tag}", flush=True)
            if r["eps"] < EXACT and dmax != dheads[-1]:
                print(f"     -> exact from D_max={dmax}; larger D_max cannot change the result, skipping them")
                break


if __name__ == "__main__":
    main()
