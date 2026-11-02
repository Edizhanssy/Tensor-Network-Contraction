# Phase A benchmark: every kernel version on the same circuits, against exact references.
#
#   python bench_A.py all            # all versions, small + big cases (on an idle PC), then the table
#   python bench_A.py all --small    # only the small cases (a few minutes)
#   python bench_A.py all --only B1,A2   # only these versions
#   python bench_A.py report         # table from bench_A.csv
#
# Versions (each a separate file, so nothing has to be swapped by hand):
#   baseline = main_baseline.py   (kernel used for all sweeps so far)
#   A1       = main_A1.py         (+ center tracking, swaps truncated in canonical gauge)
#   A2       = main_A2.py         (+ bond cut always: lossless rank reduction, QR trick)
#   A3       = main_A2.py, compress=True (+ recompress the merged MPS after every merge)
#   B1       = main_B1.py         (+ logSize cache)
#   B2       = main_B2.py         (+ min-heap of edge costs, lazy invalidation)
#   B3       = main_B3.py         (+ CATN rule: log2 cost bucket, ties -> most triangles)
#   B4       = main.py         (+ tensorSwap: rank computed once, no isfinite check)
#   main_redblack.py now has C1 (batches); its default K=1 is bit-identical to B4
# Each run is a child process with a time limit; within a version the big cases stop at the first timeout.
# References: exact ln|amp| from CATN (no truncation), exact contraction orders, or the state vector.
import csv, json, os, subprocess, sys, time
import numpy as np

SMALL = [  # name, L, d, seed, exact ln|amp|, source of the reference
    ("L5 d12", 5, 12, 0, -10.863674078189362, "state vector"),
    ("L6 d14", 6, 14, 0, -13.533747486762000, "exact orders"),
    ("L6 d16", 6, 16, 0, -12.552317591793326, "exact orders"),
    ("L10 d8", 10, 8, 0, -37.58825906084385, "CATN"),
    ("L20 d8", 20, 8, 0, -146.4967154741864, "CATN"),
]
BIG = [
    ("L30 d8", 30, 8, 0, -348.830739446184, "CATN"),
    ("L40 d8", 40, 8, 0, -663.4292474226539, "CATN"),
]
VERSIONS = [("B4", "main_B4", False), ("B3", "main_B3", False), ("B2", "main_B2", False), ("B1", "main_B1", False), ("A2", "main_A2", False), ("A3", "main_A2", True), ("A1", "main_A1", False),
            ("baseline", "main_baseline", False)]
SKIP = {("A1", "L40 d8"), ("baseline", "L40 d8")}      # known to take > 1 h (sweep A, catn_compare)
DHEAD, XHEAD = 50, 500
TIME_LIMIT = 3600
CSV_FILE = "bench_A.csv"
FIELDS = ["label", "case", "status", "seconds", "eps", "lnabs", "abs_err", "ref_source"]


def child(module, compress, name):
    import importlib
    from TensorPrimitive.circuits import randomCircuit
    from TensorPrimitive.network import circuitToEdges, edgesToLegs
    _, L, d, seed, ref, src = next(c for c in SMALL + BIG if c[0] == name)
    m = importlib.import_module(module)
    A, el = circuitToEdges(randomCircuit(L, d, seed=seed), [0] * (L * L))
    legs = edgesToLegs(el)
    t0 = time.time()
    if compress:
        lnZ, eps = m.MPSCalculus(A, legs, DHEAD, XHEAD, compress=True)
    else:
        lnZ, eps = m.MPSCalculus(A, legs, DHEAD, XHEAD)
    return dict(seconds=time.time() - t0, eps=float(eps), lnabs=float(np.real(lnZ)),
                abs_err=abs(float(np.real(lnZ)) - ref), ref_source=src)


def append_row(row):
    new = not os.path.exists(CSV_FILE)
    with open(CSV_FILE, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


def check_csv():
    # an older bench_A.csv with a different header would silently shift the new columns
    if os.path.exists(CSV_FILE):
        with open(CSV_FILE) as f:
            header = f.readline().strip().split(",")
        if header != FIELDS:
            sys.exit(f"{CSV_FILE} has an old header {header}; rename it (e.g. bench_A_old.csv) and run again")


def run_all(cases, only=None):
    check_csv()
    done = set()
    if os.path.exists(CSV_FILE):
        done = {(r["label"], r["case"]) for r in csv.DictReader(open(CSV_FILE))}
    for label, module, compress in VERSIONS:
        if only and label not in only:
            continue
        for name, *_ in cases:
            if (label, name) in done or (label, name) in SKIP:
                continue
            cmd = [sys.executable, __file__, "one", module, str(int(compress)), name]
            row = dict(label=label, case=name)
            t0 = time.time()
            try:
                out = subprocess.run(cmd, capture_output=True, text=True, timeout=TIME_LIMIT)
                if out.returncode != 0:
                    row.update(status="crash", seconds=round(time.time() - t0, 1))
                    print(out.stderr[-600:])
                else:
                    row.update(status="ok", **json.loads(out.stdout.strip().splitlines()[-1]))
            except subprocess.TimeoutExpired:
                row.update(status="timeout", seconds=TIME_LIMIT)
            append_row(row)
            if row["status"] == "ok":
                print(f"[{label:8s}] {name}: {row['seconds']:7.1f} s   eps = {row['eps']:.2e}   "
                      f"|Δ ln|amp|| = {row['abs_err']:.2e}   (ref: {row['ref_source']})", flush=True)
            else:
                print(f"[{label:8s}] {name}: {row['status']} after {row['seconds']} s -> rest of {label} skipped", flush=True)
                break


def report():
    check_csv()
    rows = list(csv.DictReader(open(CSV_FILE)))
    labels = [v[0] for v in VERSIONS[::-1] if any(r["label"] == v[0] for r in rows)]
    labels += [l for l in dict.fromkeys(r["label"] for r in rows) if l not in labels]
    cases = [c[0] for c in SMALL + BIG if any(r["case"] == c[0] for r in rows)]
    out = ["| case | " + " | ".join(f"{l}: time / ε / true err" for l in labels) + " |",
           "|---|" + "---|" * len(labels)]
    for c in cases:
        cells = []
        for l in labels:
            r = [x for x in rows if x["label"] == l and x["case"] == c]
            if not r:
                cells.append("–")
            elif r[-1]["status"] != "ok":
                cells.append(f"{r[-1]['status']} ({float(r[-1]['seconds']):.0f} s)")
            else:
                cells.append(f"{float(r[-1]['seconds']):.1f} s / {float(r[-1]['eps']):.1e} / {float(r[-1]['abs_err']):.1e}")
        out.append(f"| {c} | " + " | ".join(cells) + " |")
    text = "\n".join(out)
    with open("bench_A_report.md", "w", encoding="utf-8") as f:
        f.write("# Phase A benchmark\n\nD̂ = 50, χ̂ = 500. true err = |ln|amp| − exact|.\n\n" + text + "\n")
    print(text)


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "one":
        print(json.dumps(child(a[1], a[2] == "1", " ".join(a[3:]))))
    elif a and a[0] == "report":
        report()
    elif a and a[0] == "all":
        only = a[a.index("--only") + 1].split(",") if "--only" in a else None
        run_all(SMALL if "--small" in a else SMALL + BIG, only)
        report()
    else:
        sys.exit("usage: python bench_A.py all [--small]  |  python bench_A.py report")
