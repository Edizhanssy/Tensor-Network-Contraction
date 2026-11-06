# Accuracy gap on the authors' circuits (n >= 900): we truncate (eps 7.7e-10 at n=900), CATN does not.
# Two hypotheses, tested by changing ONE parameter at a time:
#   H1  CATN's reverse=True (reverse an MPS when that needs fewer swaps -> fewer truncations; our B6)
#   H2  the "lossless" tolerance: ours drops singular values < 1e-12 * s_max, CATN < 1e-14
# Configurations (D_max = 50, chi_max = 500, one run each, every run a fresh child process):
#   ours  tol=1e-12           reference (today's kernel)
#   ours  tol=1e-14           H2: does matching CATN's cutoff close the gap?
#   catn  reverse=T cut=1e-14 reference (CATN as published)
#   catn  reverse=F cut=1e-14 H1: does CATN start to truncate without reverse?
#   catn  reverse=T cut=1e-12 H2 the other way: does CATN get worse with our tolerance?
# Note: "tol" for ours applies to swaps (tensorSwap) and bond cuts (cutBond). The initial TT-SVD of every
# tensor keeps its fixed 1e-12 (exact anyway: it only drops numerically zero singular values).
# CATN's "err" is its own measure (sum of discarded singular values), ours is the discarded weight;
# compare "exact (~1e-13 or below) or not", and ln|amp| between runs.
#   python exp_gap.py --catn C:\path\to\catn            # runs (resumes from exp_gap.csv), then the table
#   python exp_gap.py --catn C:\path\to\catn --small    # only n = 100 (a quick check, ~1 min)
#   python exp_gap.py report
import argparse, csv, json, os, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.dirname(HERE), os.path.join(os.path.dirname(HERE), "TensorPrimitive"),
          os.path.join(os.path.dirname(HERE), "Source")):
    if p not in sys.path:
        sys.path.insert(0, p)
import numpy as np

DHEAD, XHEAD = 50, 500
OUT = "exp_gap.csv"
FIELDS = ["n", "config", "status", "seconds", "err", "lnabs"]
CIRCS = {100: "supremacy2d100-d10-seed1-rank3-new", 900: "supremacy2d900-d10-seed1-rank3-new",
         1600: "supremacy2d1600-d10-seed1-rank3-new"}
CONFIGS = [("ours tol=1e-12", "ours", dict(tol=1e-12)),
           ("ours tol=1e-14", "ours", dict(tol=1e-14)),
           ("catn reverse=T cut=1e-14", "catn", dict(reverse=True, cutoff=1e-14)),
           ("catn reverse=F cut=1e-14", "catn", dict(reverse=False, cutoff=1e-14)),
           ("catn reverse=T cut=1e-12", "catn", dict(reverse=True, cutoff=1e-12))]


def run_ours(A, legs, tol):
    import main
    d = list(main.tensorSwap.__defaults__)          # (right, Xhead, err, tol): swaps get tol via the default
    d[-1] = tol
    main.tensorSwap.__defaults__ = tuple(d)
    lnZ, eps = main.MPSCalculus(A, legs, DHEAD, XHEAD, tol=tol)
    return float(np.real(lnZ)), float(eps)


def run_catn(A, legs, catn_dir, reverse, cutoff):
    # same loader as catn_compare.run_catn (two NumPy-2 patches, algorithm untouched), with reverse/cutoff free
    import types
    sys.path.insert(0, catn_dir)
    def load_patched(name, old, new):
        path = os.path.join(catn_dir, name + ".py")
        src = open(path, encoding="utf-8").read()
        assert src.count(old) == 1, f"unexpected CATN version ({name})"
        mod = types.ModuleType(name); mod.__file__ = path; sys.modules[name] = mod
        exec(compile(src.replace(old, new), path, "exec"), mod.__dict__)
        return mod
    load_patched("mps_node_np", "lognorm = math.log(np.abs(result))", "lognorm = math.log(np.abs(result).item())")
    TN = load_patched("tensor_network_np", "np.array([2])", "np.array(2)").Tensor_Network_np
    tn = TN([np.asarray(a, dtype=complex) for a in A], [np.array(l) for l in legs],
            seed=1, maxdim=10**9, verbose=-1, Dmax=DHEAD, chi=XHEAD, node_type="mps",
            norm_method=1, svdopt=True, swapopt=True, reverse=reverse, cutoff=cutoff, select=1)
    for node in tn.tensors.values():
        node.neighbor = np.array(node.neighbor)
    tn.select_edge_init()
    lnZ, error, psi = tn.contraction()
    amp = complex(np.exp(np.asarray(lnZ).ravel()[0]) * np.asarray(psi).ravel()[0])
    return float(np.log(abs(amp))), float(np.linalg.norm(error))


def child(n, label, catn_dir):
    import catn_compare as cc
    code, opts = next((c, o) for l, c, o in CONFIGS if l == label)
    A, legs = cc.build("pz", CIRCS[n], catn_dir)
    t0 = time.perf_counter()
    if code == "ours":
        lnabs, err = run_ours(A, legs, opts["tol"])
    else:
        lnabs, err = run_catn(A, legs, catn_dir, opts["reverse"], opts["cutoff"])
    return dict(seconds=time.perf_counter() - t0, err=err, lnabs=lnabs)


def rows():
    return list(csv.DictReader(open(OUT))) if os.path.exists(OUT) else []


def run_all(catn_dir, ns):
    done = {(r["n"], r["config"]) for r in rows()}
    for n in ns:
        for label, _, _ in CONFIGS:
            if (str(n), label) in done:
                continue
            cmd = [sys.executable, os.path.abspath(__file__), "one", str(n), label, catn_dir]
            t0 = time.time()
            out = subprocess.run(cmd, capture_output=True, text=True)
            row = dict(n=n, config=label)
            if out.returncode != 0:
                row.update(status="crash", seconds=round(time.time() - t0, 1))
                print(out.stderr[-800:])
            else:
                row.update(status="ok", **json.loads(out.stdout.strip().splitlines()[-1]))
            new = not os.path.exists(OUT)
            with open(OUT, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
                if new:
                    w.writeheader()
                w.writerow(row)
            if row["status"] == "ok":
                print(f"n={n:5d}  {label:26s} {row['seconds']:8.1f} s   err {row['err']:.2e}   ln|amp| {row['lnabs']:.10f}", flush=True)
            else:
                print(f"n={n:5d}  {label:26s} CRASH", flush=True)


def report():
    rs = rows()
    out = ["# Accuracy gap on the authors' circuits: which hypothesis?\n",
           f"D_max = {DHEAD}, chi_max = {XHEAD}. err: ours = discarded weight, CATN = sum of discarded singular values. "
           "Δ = |ln|amp| − ln|amp| of CATN as published| (same n).\n",
           "| n | config | time [s] | err | ln\\|amp\\| | Δ vs CATN ref |", "|---|---|---|---|---|---|"]
    for n in sorted({int(r["n"]) for r in rs}):
        ref = next((float(r["lnabs"]) for r in rs if int(r["n"]) == n and r["config"] == "catn reverse=T cut=1e-14" and r["status"] == "ok"), None)
        for label, _, _ in CONFIGS:
            r = next((r for r in rs if int(r["n"]) == n and r["config"] == label), None)
            if r is None:
                continue
            if r["status"] != "ok":
                out.append(f"| {n} | {label} | crash | – | – | – |"); continue
            d = "–" if ref is None else f"{abs(float(r['lnabs']) - ref):.1e}"
            out.append(f"| {n} | {label} | {float(r['seconds']):.1f} | {float(r['err']):.1e} | {float(r['lnabs']):.10f} | {d} |")
    text = "\n".join(out)
    open("exp_gap_report.md", "w", encoding="utf-8").write(text + "\n")
    print(text)


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "one":
        print(json.dumps(child(int(a[1]), a[2], a[3])))
    elif a and a[0] == "report":
        report()
    else:
        ap = argparse.ArgumentParser()
        ap.add_argument("--catn", required=True)
        ap.add_argument("--small", action="store_true")
        args = ap.parse_args()
        catn = os.path.abspath(args.catn)
        if not os.path.exists(os.path.join(catn, "tensor_network_np.py")):
            sys.exit(f"--catn {catn} does not look like the catn repository")
        run_all(catn, [100] if args.small else [900, 1600])
        report()
