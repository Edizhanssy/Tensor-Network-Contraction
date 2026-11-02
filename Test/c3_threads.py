# C3 prototype: how much of the ideal C2 speedup do real Python threads deliver?
# Two kinds of parallelism, each measured on its own:
#   inter-step: the steps of a batch on P Python threads (K = None), BLAS forced to 1 thread
#               -> compare with C2's ideal S_P (LPT + serial part, no overhead)
#   intra-step: serial steps (K = 1), but BLAS/LAPACK inside every SVD/QR on B threads
#               -> the "tiling" inside one big SVD; only matters where matrices are big (pz n=1600)
# Each configuration is a fresh child process (the BLAS thread count must be set BEFORE numpy is
# imported), REPEATS times, minimum wall time. Circuit loading is not timed.
#   python c3_threads.py                          # our L40 only
#   python c3_threads.py --catn C:\path\to\catn   # + the authors' n = 900 and 1600
#   python c3_threads.py report
import argparse, json, os, subprocess, sys, time

REPEATS = 3
DHEAD, XHEAD = 50, 500
OUT = "c3_threads.json"
CIRCUITS = [("ours", "40:8:0")]
PZ = [("pz", "supremacy2d900-d10-seed1-rank3-new"), ("pz", "supremacy2d1600-d10-seed1-rank3-new")]
# (label, K, Python threads, BLAS threads)
CONFIGS = ([("serial K=1", 1, None, 1), ("serial K=None", None, None, 1)]
           + [(f"inter P={P}", None, P, 1) for P in (2, 4, 8, 24)]
           + [(f"intra BLAS={B}", 1, None, B) for B in (4, 24)])


def child(family, circuit, K, threads, catn_dir):
    import numpy as np
    import catn_compare as cc
    import main_parallel
    A, legs = cc.build(family, circuit, catn_dir)
    t0 = time.perf_counter()
    lnZ, eps = main.MPSCalculus(A, legs, DHEAD, XHEAD, K=K, threads=threads)
    return dict(seconds=time.perf_counter() - t0, lnabs=float(np.real(lnZ)), eps=float(eps))


def run_all(catn_dir):
    rows = json.load(open(OUT)) if os.path.exists(OUT) else []
    done = {(r["family"], r["circuit"], r["label"], r["rep"]) for r in rows}
    for family, circ in CIRCUITS + (PZ if catn_dir else []):
        for label, K, P, B in CONFIGS:
            for rep in range(REPEATS):
                if (family, circ, label, rep) in done:
                    continue
                env = dict(os.environ, OPENBLAS_NUM_THREADS=str(B), OMP_NUM_THREADS=str(B), MKL_NUM_THREADS=str(B))
                cmd = [sys.executable, __file__, "one", family, circ, str(K), str(P), catn_dir or "-"]
                out = subprocess.run(cmd, capture_output=True, text=True, env=env)
                if out.returncode != 0:
                    print(out.stderr[-800:]); sys.exit(f"{family} {circ} {label} crashed")
                r = dict(json.loads(out.stdout.strip().splitlines()[-1]), family=family, circuit=circ,
                         label=label, rep=rep)
                rows.append(r)
                json.dump(rows, open(OUT, "w"))
                print(f"[{family:4s} {circ:34s}] {label:15s} rep {rep}: {r['seconds']:7.2f} s  eps {r['eps']:.1e}", flush=True)


def report():
    rows = json.load(open(OUT))
    out = ["# C3: real threads vs. the ideal (C2)\n",
           f"Minimum wall time over {REPEATS} runs. inter = steps of a batch on P Python threads (BLAS 1 thread). "
           "intra = serial steps, BLAS on B threads. speedup vs 'serial K=None' for inter, vs 'serial K=1' for intra. "
           "same = identical ln|amp| as the serial run with the same K.\n",
           "| circuit | config | time [s] | speedup | same result |", "|---|---|---|---|---|"]
    for family, circ in CIRCUITS + PZ:
        sel = [r for r in rows if r["family"] == family and r["circuit"] == circ]
        if not sel:
            continue
        best = {}
        for r in sel:
            if r["label"] not in best or r["seconds"] < best[r["label"]]["seconds"]:
                best[r["label"]] = r
        nm = f"authors n={circ.split('supremacy2d')[1].split('-')[0]}" if family == "pz" else f"ours L{circ.split(':')[0]}"
        for label, K, P, B in CONFIGS:
            if label not in best:
                continue
            base = best.get("serial K=1" if (K == 1) else "serial K=None")
            r = best[label]
            sp = base["seconds"] / r["seconds"] if base else float("nan")
            same = "–" if base is None or base is r else ("yes" if r["lnabs"] == base["lnabs"] else f"Δ {abs(r['lnabs'] - base['lnabs']):.1e}")
            out.append(f"| {nm} | {label} | {r['seconds']:.2f} | {sp:.2f}× | {same} |")
    text = "\n".join(out)
    open("c3_threads_report.md", "w", encoding="utf-8").write(text + "\n")
    print(text)


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "one":
        K = None if a[3] == "None" else int(a[3])
        P = None if a[4] == "None" else int(a[4])
        print(json.dumps(child(a[1], a[2], K, P, None if a[5] == "-" else a[5])))
    elif a and a[0] == "report":
        report()
    else:
        ap = argparse.ArgumentParser()
        ap.add_argument("--catn", default=None)
        c = ap.parse_args().catn
        run_all(os.path.abspath(c) if c else None)
        report()
