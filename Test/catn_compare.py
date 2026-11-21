# AI usage is avaliable for this particular code!

# Apples to apples: our MPS-calculus kernel vs. the authors' CATN code (github.com/panzhang83/catn),
# on the SAME circuits with the SAME D_hat / chi_hat.
#
#                       | our kernel | CATN
#   their circuits (pz) |     ?      |  ?        -> isolates the algorithm
#   our circuits        |     ?      |  ?        -> isolates the circuit protocol
#
#   python catn_compare.py --catn C:\path\to\catn          # run the matrix (resumes), then report
#   python catn_compare.py report                           # only the report from the CSV
#
# Needs: pip install networkx   (CATN imports it; scipy is already there)
# Every run is a child process with a time limit. Within one (circuit family, code, D_hat, chi_hat)
# series the sizes grow; the series stops at the first timeout / crash.
import argparse, csv, json, os, subprocess, sys, time, tracemalloc
import numpy as np

# ------------------------------------------------------------------ settings
PARAMS = [(50, 500)]                         # (D_hat, chi_hat); add (600, 1500) = CATN README example
PZ_CIRCUITS = [f"supremacy2d{n}-d10-seed1-rank3-new" for n in (100, 400, 900, 1600)]   # d10 = 1+8+1
OUR_CIRCUITS = [(L, 8, 0) for L in (10, 20, 30, 40)]                                     # same sizes, d = 8
CODES = ["ours", "catn"]
TIME_LIMIT = 3600
CSV_FILE = "catn_compare.csv"
REPORT_FILE = "catn_compare_report.md"
FIELDS = ["family", "circuit", "n_qubits", "code", "dhead", "xhead", "status", "seconds", "peak_MB",
          "err", "amp_re", "amp_im", "log_abs_amp", "note"]


# --------------------------------------------------------------- circuits
def load_pz(prefix):
    """CATN's graph files -> (A, legs) in OUR format. Files: 1-based edge labels, Fortran order,
    complex numbers stored as interleaved (re, im) floats."""
    def rows(ext, typ):
        with open(prefix + ext) as f:
            return [np.array(line.split(), dtype=typ) for line in f if line.strip()]
    vals, labels, sizes = rows(".tensors.dat", float), rows(".labels.dat", int), rows(".sizes.dat", int)
    A = [v.view(complex).reshape(s, order="F") for v, s in zip(vals, sizes)]
    from TensorPrimitive.network import edgesToLegs
    legs = edgesToLegs([list(l - 1) for l in labels])      # 1-based -> 0-based edge IDs
    return A, legs


def load_ours(L, d, seed):
    from TensorPrimitive.circuits import randomCircuit
    from TensorPrimitive.network import circuitToEdges, edgesToLegs
    A, el = circuitToEdges(randomCircuit(L, d, seed=seed), [0] * (L * L))
    return A, edgesToLegs(el)


def build(family, circuit, catn_dir):
    if family == "pz":
        return load_pz(os.path.join(catn_dir, "graphs", circuit))
    L, d, seed = map(int, circuit.split(":"))
    return load_ours(L, d, seed)


# ------------------------------------------------------------ the two codes
def run_ours(A, legs, dhead, xhead):
    from main_redblack import MPSCalculus
    lnZ, eps = MPSCalculus(A, legs, dhead, xhead)
    return complex(np.exp(lnZ)), float(eps)                 # eps = sum of discarded weights


def run_catn(A, legs, dhead, xhead, catn_dir):
    sys.path.insert(0, catn_dir)
    # NumPy >= 2 compatibility: CATN (2019) turns size-1 arrays into Python floats ("%.3e" % arr,
    # math.log(arr)), which newer NumPy refuses. Load their two modules with exactly these two spots
    # patched (size-1 array -> scalar); the algorithm itself is untouched.
    import types
    def load_patched(name, old, new):
        path = os.path.join(catn_dir, name + ".py")
        with open(path, encoding="utf-8") as f:
            src = f.read()
        assert src.count(old) == 1, f"unexpected CATN version ({name})"
        mod = types.ModuleType(name)
        mod.__file__ = path
        sys.modules[name] = mod
        exec(compile(src.replace(old, new), path, "exec"), mod.__dict__)
        return mod
    load_patched("mps_node_np", "lognorm = math.log(np.abs(result))", "lognorm = math.log(np.abs(result).item())")
    Tensor_Network_np = load_patched("tensor_network_np", "np.array([2])", "np.array(2)").Tensor_Network_np
    tn = Tensor_Network_np([np.asarray(a, dtype=complex) for a in A], [np.array(l) for l in legs],
                           seed=1, maxdim=10**9, verbose=-1, Dmax=dhead, chi=xhead, node_type="mps",
                           norm_method=1, svdopt=True, swapopt=True, reverse=True, cutoff=1e-14, select=1)
    # CATN stores the neighbour lists as Python lists, but compares them elementwise (np.argwhere(nb == j));
    # make them arrays and rebuild the edge-cost buckets that __init__ computed with the lists.
    for node in tn.tensors.values():
        node.neighbor = np.array(node.neighbor)
    tn.select_edge_init()
    lnZ, error, psi = tn.contraction()
    amp = complex(np.exp(np.asarray(lnZ).ravel()[0]) * np.asarray(psi).ravel()[0])   # as in their qc.py
    return amp, float(np.linalg.norm(error))               # CATN error = sum of discarded singular values


def child(family, circuit, code, dhead, xhead, catn_dir):
    A, legs = build(family, circuit, catn_dir)
    tracemalloc.start()
    t0 = time.time()
    if code == "ours":
        amp, err = run_ours(A, legs, dhead, xhead)
    else:
        amp, err = run_catn(A, legs, dhead, xhead, catn_dir)
    seconds = time.time() - t0
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return dict(seconds=seconds, peak_MB=peak / 1e6, err=err, amp_re=amp.real, amp_im=amp.imag,
                log_abs_amp=float(np.log(abs(amp))) if amp != 0 else float("-inf"))


# ----------------------------------------------------------------- driver
def load_rows():
    if not os.path.exists(CSV_FILE):
        return []
    with open(CSV_FILE) as f:
        return list(csv.DictReader(f))


def append_row(row):
    new = not os.path.exists(CSV_FILE)
    with open(CSV_FILE, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


def n_of(family, circuit):
    if family == "pz":
        return int(circuit.split("supremacy2d")[1].split("-")[0])
    return int(circuit.split(":")[0]) ** 2


def run_all(catn_dir):
    rows = load_rows()
    done = {(r["family"], r["circuit"], r["code"], r["dhead"], r["xhead"]) for r in rows}
    stopped = {(r["family"], r["code"], r["dhead"], r["xhead"]) for r in rows if r["status"] != "ok"}
    series = [("pz", PZ_CIRCUITS), ("ours", [f"{L}:{d}:{s}" for L, d, s in OUR_CIRCUITS])]
    for dhead, xhead in PARAMS:
        for family, circuits in series:
            for code in CODES:
                key = (family, code, str(dhead), str(xhead))
                if key in stopped:
                    print(f"[{family} circuits / {code} / D={dhead} chi={xhead}] stopped earlier, skipping")
                    continue
                for circ in circuits:
                    if (family, circ, code, str(dhead), str(xhead)) in done:
                        continue
                    cmd = [sys.executable, __file__, "one", family, circ, code, str(dhead), str(xhead), catn_dir]
                    row = dict(family=family, circuit=circ, n_qubits=n_of(family, circ), code=code,
                               dhead=dhead, xhead=xhead)
                    t0 = time.time()
                    try:
                        out = subprocess.run(cmd, capture_output=True, text=True, timeout=TIME_LIMIT)
                        if out.returncode != 0:
                            last = ((out.stderr.strip() or out.stdout.strip()).splitlines() or ["unknown"])[-1]
                            row.update(status="crash", seconds=round(time.time() - t0, 1), note=last[:200])
                            print((out.stderr or out.stdout)[-800:])
                        else:
                            res = json.loads(out.stdout.strip().splitlines()[-1])
                            res.pop("n_qubits", None)
                            row.update(status="ok", **res)
                    except subprocess.TimeoutExpired:
                        row.update(status="timeout", seconds=TIME_LIMIT, note=f"exceeded {TIME_LIMIT} s")
                    append_row(row)
                    print(f"[{family:4s} {circ:34s}] {code:4s} D={dhead} chi={xhead}: {row['status']:7s} "
                          f"t={float(row.get('seconds', 0)):7.1f}s  peak={row.get('peak_MB', 'n/a')}  "
                          f"err={row.get('err', 'n/a')}  ln|amp|={row.get('log_abs_amp', 'n/a')}", flush=True)
                    if row["status"] != "ok":
                        print("  -> series stops here")
                        break


# ----------------------------------------------------------------- report
def _f(x, fmt):
    try:
        return format(float(x), fmt)
    except (TypeError, ValueError):
        return "–"


def report():
    rows = load_rows()
    out = ["# Our kernel vs. CATN on identical circuits\n",
           "err: ours = Σ discarded weight Σσ²/Σσ² (ε_SVD); CATN = Σ discarded singular values (absolute). "
           "Not the same measure; both are 0 when nothing is truncated.\n"]
    for dhead, xhead in sorted({(int(r["dhead"]), int(r["xhead"])) for r in rows}):
        out.append(f"## D̂ = {dhead}, χ̂ = {xhead}\n")
        out.append("| circuits | qubits | code | status | time [s] | peak [MB] | err | ln\\|amp\\| | Δ ln\\|amp\\| ours vs CATN |")
        out.append("|---|---|---|---|---|---|---|---|---|")
        sel = [r for r in rows if int(r["dhead"]) == dhead and int(r["xhead"]) == xhead]
        for fam in ("pz", "ours"):
            for n in sorted({int(r["n_qubits"]) for r in sel if r["family"] == fam}):
                pair = {r["code"]: r for r in sel if r["family"] == fam and int(r["n_qubits"]) == n}
                diff = "–"
                if all(c in pair and pair[c]["status"] == "ok" for c in CODES):
                    diff = f"{abs(float(pair['ours']['log_abs_amp']) - float(pair['catn']['log_abs_amp'])):.1e}"
                for c in CODES:
                    if c in pair:
                        r = pair[c]
                        out.append(f"| {'authors (d10 = 1+8+1)' if fam == 'pz' else 'ours (d = 8)'} | {n} | {c} | "
                                   f"{r['status']} | {_f(r['seconds'], '.1f')} | {_f(r['peak_MB'], '.1f')} | "
                                   f"{_f(r['err'], '.1e')} | {_f(r['log_abs_amp'], '.6f')} | {diff if c == 'ours' else ''} |")
        out.append("")
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "one":
        fam, circ, code, dh, xh, cdir = sys.argv[2:8]
        print(json.dumps(child(fam, circ, code, int(dh), int(xh), cdir)))
    elif len(sys.argv) > 1 and sys.argv[1] == "report":
        report()
    else:
        ap = argparse.ArgumentParser()
        ap.add_argument("--catn", required=True, help="path to the cloned catn repository")
        a = ap.parse_args()
        catn_dir = os.path.abspath(a.catn)
        missing = [p for p in ("tensor_network_np.py", "mps_node_np.py", os.path.join("graphs", PZ_CIRCUITS[0] + ".tensors.dat"))
                   if not os.path.exists(os.path.join(catn_dir, p))]
        if missing:                                   # a wrong path must not be recorded as crashed runs
            sys.exit(f"--catn {catn_dir} does not look like the catn repository (missing: {', '.join(missing)})")
        run_all(catn_dir)
        report()
