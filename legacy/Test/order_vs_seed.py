# AI usage is avaliable for this particular code!

# Why is d=12 truncated but d=14 exact (L=5)?  Two hypotheses, two experiments:
#   Experiment 1 "seed":  different circuits (circuit seed 0..N-1), same method (deterministic greedy order)
#                         -> large spread in eps  => the circuit realization matters
#   Experiment 2 "order": same circuit (seed 0), different but equally greedy orders
#                         (random tie-breaking in ContractionOrder, order seed 0..N-1)
#                         -> large spread in eps  => the contraction order matters
# Both can be true; the spreads tell how much each one matters.
#
#   python order_vs_seed.py            # run everything (resumes), then summary + figure
#   python order_vs_seed.py --sv       # additionally check L=5 runs against the exact state vector
#                                      #   (~70 s and ~1.6 GB per distinct circuit)
#   python order_vs_seed.py report     # only summary + figure from the CSV
import csv, json, os, subprocess, sys, time
import numpy as np

# ------------------------------------------------------------------ settings
CASES = [(5, 12), (5, 14), (6, 14), (6, 16)]   # (L, d): around the truncation onset of the baseline
N = 10                                         # circuits per case (exp. 1) and orders per case (exp. 2)
DHEAD, XHEAD = 50, 500
TIME_LIMIT = 600                               # s per run; these cases took 1-30 s in the baseline
SV_MAX_QUBITS = 25
CSV_FILE = "order_vs_seed.csv"
FIG_FILE = "order_vs_seed.png"
REPORT_FILE = "order_vs_seed_report.md"
FIELDS = ["exp", "L", "d", "circuit_seed", "order_seed", "status", "seconds", "eps",
          "lnZ_re", "lnZ_im", "sv_rel_err", "P_times_2n", "note"]


# ------------------------------------------------------------- children
def build(L, d, circuit_seed):
    from TensorPrimitive.circuits import randomCircuit
    from TensorPrimitive.network import circuitToEdges, edgesToLegs
    circuit = randomCircuit(L, d, seed=circuit_seed)
    x = [0] * (L * L)
    A, el = circuitToEdges(circuit, x)
    return circuit, x, A, edgesToLegs(el)


def child_mps(L, d, circuit_seed, order_seed):
    from main_heap import MPSCalculus
    _, _, A, legs = build(L, d, circuit_seed)
    rng = None if order_seed < 0 else np.random.default_rng(order_seed)
    t0 = time.time()
    lnZ, eps = MPSCalculus(A, legs, DHEAD, XHEAD, rng=rng)
    return dict(seconds=time.time() - t0, eps=eps, lnZ_re=float(np.real(lnZ)), lnZ_im=float(np.imag(lnZ)))


def child_sv(L, d, circuit_seed):
    from TensorPrimitive.network import stateVectorAmplitude
    circuit, x, _, _ = build(L, d, circuit_seed)
    ref = stateVectorAmplitude(circuit, x)
    return dict(re=float(ref.real), im=float(ref.imag))


def run_child(args):
    cmd = [sys.executable, __file__] + [str(a) for a in args]
    t0 = time.time()
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=TIME_LIMIT)
    except subprocess.TimeoutExpired:
        return "timeout", None, f"exceeded {TIME_LIMIT} s", TIME_LIMIT
    if out.returncode != 0:
        last = (out.stderr.strip().splitlines() or ["unknown error"])[-1]
        return "crash", None, last[:200], time.time() - t0
    for line in out.stdout.splitlines():                # forward [safeSVD] warnings
        if line.startswith("[safeSVD]"):
            print("     ", line)
    return "ok", json.loads(out.stdout.strip().splitlines()[-1]), "", None


# --------------------------------------------------------------- driver
def load_rows():
    if not os.path.exists(CSV_FILE):
        return []
    with open(CSV_FILE) as f:
        return list(csv.DictReader(f))


def append_row(row):
    new = not os.path.exists(CSV_FILE)
    with open(CSV_FILE, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def run_all(with_sv):
    done = {(r["exp"], r["L"], r["d"], r["circuit_seed"], r["order_seed"]) for r in load_rows()}
    sv_cache = {}
    for L, d in CASES:
        jobs = [("seed", s, -1) for s in range(N)] + [("order", 0, o) for o in range(N)]
        for exp, cs, os_ in jobs:
            key = (exp, str(L), str(d), str(cs), str(os_))
            if key in done:
                continue
            status, res, note, secs = run_child(["mps", L, d, cs, os_])
            row = dict(exp=exp, L=L, d=d, circuit_seed=cs, order_seed=os_, status=status, note=note)
            if res:
                row.update(res)
            elif secs is not None:
                row["seconds"] = round(secs, 1)
            if with_sv and status == "ok" and L * L <= SV_MAX_QUBITS:
                if (L, d, cs) not in sv_cache:
                    st, ref, _, _ = run_child(["sv", L, d, cs])
                    sv_cache[(L, d, cs)] = complex(ref["re"], ref["im"]) if st == "ok" else None
                ref = sv_cache[(L, d, cs)]
                if ref is not None:
                    amp = np.exp(complex(row["lnZ_re"], row["lnZ_im"]))
                    row["sv_rel_err"] = abs(amp - ref) / max(abs(ref), 1e-300)
                    row["P_times_2n"] = abs(ref) ** 2 * 2.0 ** (L * L)
            append_row(row)
            tag = "greedy" if os_ < 0 else f"order {os_}"
            extra = f"  true rel.err={row['sv_rel_err']:.1e}" if "sv_rel_err" in row else ""
            print(f"[{exp:5s}] L={L} d={d:>2} circuit {cs} {tag:8s}: {status:7s} "
                  f"t={float(row.get('seconds', 0) or 0):6.1f}s  eps={row.get('eps', 'n/a')}{extra}",
                  flush=True)


# --------------------------------------------------------------- report
EXACT = 1e-10          # eps below this: no truncation happened


def _eps(r):
    return max(float(r["eps"]), 1e-32)


def exact_refs(rows):
    """ln|amp| of every circuit that at least one run contracted exactly -> free ground truth,
    also for circuits far too large for a state vector (36 qubits ~ 1 TB)."""
    refs = {}
    for r in rows:
        if float(r["eps"]) < EXACT:
            refs.setdefault((r["L"], r["d"], r["circuit_seed"]), []).append(float(r["lnZ_re"]))
    return {k: float(np.median(v)) for k, v in refs.items()}


def true_err(r, refs):
    """|Delta ln|amp|| against the exact reference (state vector if available, else exact order)."""
    if r.get("sv_rel_err"):
        return None                       # state-vector error is reported separately
    k = (r["L"], r["d"], r["circuit_seed"])
    if float(r["eps"]) >= EXACT and k in refs:
        return abs(float(r["lnZ_re"]) - refs[k])
    return None


def report():
    rows = [r for r in load_rows() if r["status"] == "ok"]
    refs = exact_refs(rows)
    out = ["# Seed vs. order: what makes a circuit hard for MPS calculus?\n",
           f"D̂ = {DHEAD}, χ̂ = {XHEAD}. Experiment 1 'seed': circuit seeds 0..{N-1}, deterministic greedy order. "
           f"Experiment 2 'order': circuit seed 0, random tie-breaking with order seeds 0..{N-1}. "
           f"'exact' = ε_SVD < {EXACT:.0e}.\n",
           "| L | d | experiment | runs | exact | ε of truncated runs: min / median / max | time [s] exact runs | time [s] truncated runs |",
           "|---|---|---|---|---|---|---|---|"]
    for L, d in CASES:
        for exp in ("seed", "order"):
            g = [r for r in rows if r["exp"] == exp and int(r["L"]) == L and int(r["d"]) == d]
            if not g:
                continue
            ex = [r for r in g if float(r["eps"]) < EXACT]
            tr = [r for r in g if float(r["eps"]) >= EXACT]
            e = np.array([float(r["eps"]) for r in tr])
            def trange(rs):
                t = [float(r["seconds"]) for r in rs]
                return f"{min(t):.1f}–{max(t):.1f}" if t else "–"
            es = f"{e.min():.1e} / {np.median(e):.1e} / {e.max():.1e}" if tr else "–"
            out.append(f"| {L} | {d} | {exp} | {len(g)} | {len(ex)} | {es} | {trange(ex)} | {trange(tr)} |")

    out.append("\n## Is ε_SVD a good error estimate?\n")
    out.append("Every circuit that some run contracted exactly gives a reference value of ln|amp| "
               "(all exact runs of the same circuit agree to ~1e-13). Truncated runs of such circuits:\n")
    out.append("| L | d | experiment | circuit | order | ε_SVD | true abs(Δ ln abs(amp)) | ratio true / ε |")
    out.append("|---|---|---|---|---|---|---|---|")
    pairs = []
    for r in rows:
        t = true_err(r, refs)
        if t is not None:
            pairs.append((float(r["eps"]), t))
            o = "greedy" if int(r["order_seed"]) < 0 else r["order_seed"]
            out.append(f"| {r['L']} | {r['d']} | {r['exp']} | {r['circuit_seed']} | {o} | "
                       f"{float(r['eps']):.1e} | {t:.1e} | {t / float(r['eps']):.1f} |")
    if pairs:
        rat = np.array([t / e for e, t in pairs])
        out.append(f"\n{len(pairs)} truncated runs with a reference: true error / ε between "
                   f"{rat.min():.1f} and {rat.max():.1f} (median {np.median(rat):.1f}).")
    sv = [r for r in rows if r.get("sv_rel_err")]
    if sv:
        out.append(f"\nState-vector checks: {len(sv)} runs, see the CSV column sv_rel_err.")

    bad = [r for r in load_rows() if r["status"] != "ok"]
    if bad:
        out.append(f"\n{len(bad)} run(s) did not finish: " +
                   ", ".join(f"{r['exp']} L={r['L']} d={r['d']} c{r['circuit_seed']} o{r['order_seed']} ({r['status']})"
                             for r in bad))
    out.append(f"\n![seed vs order]({FIG_FILE})\n")
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print("\n".join(out))
    plot(rows, pairs)


def plot(rows, pairs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw=dict(width_ratios=[2.2, 1]))
    colors = {"seed": "#2a78d6", "order": "#e8613c"}
    labels = []
    for k, (L, d) in enumerate(CASES):
        labels.append(f"L={L}, d={d}")
        for j, exp in enumerate(("seed", "order")):
            g = [r for r in rows if r["exp"] == exp and int(r["L"]) == L and int(r["d"]) == d]
            if not g:
                continue
            xs = k + (-0.15 if j == 0 else 0.15) + np.linspace(-0.06, 0.06, len(g))
            ax.scatter(xs, [_eps(r) for r in g], color=colors[exp], s=28,
                       label=("exp. 1: different circuits, greedy order" if exp == "seed"
                              else "exp. 2: circuit 0, random tie-breaking") if k == 0 else None)
    ax.set_yscale("log")
    ax.axhline(EXACT, color="gray", lw=1, ls="--")
    ax.text(-0.4, EXACT * 0.1, "exact below this line", color="gray", fontsize=8, ha="left", va="top")
    ax.set_xticks(range(len(CASES)), labels)
    ax.set_ylabel("ε_SVD (accumulated)")
    ax.set_title(f"Seed vs. order, D̂ = {DHEAD}, χ̂ = {XHEAD}  (ε < 1e-32 drawn at 1e-32)", fontsize=10)
    ax.legend(loc="center right", fontsize=9, framealpha=1)

    if pairs:
        e, t = np.array(pairs).T
        bx.scatter(e, t, color="#2a78d6", s=28)
        lo, hi = min(e.min(), t.min()) / 2, max(e.max(), t.max()) * 2
        bx.plot([lo, hi], [lo, hi], color="gray", lw=1, ls="--", label="true error = ε")
        bx.set_xscale("log"); bx.set_yscale("log")
        bx.set_xlim(lo, hi); bx.set_ylim(lo, hi)
        bx.set_xlabel("ε_SVD"); bx.set_ylabel("true |Δ ln|amp|| (exact-order reference)")
        bx.set_title("Truncated runs: estimate vs. truth", fontsize=10)
        bx.legend(loc="upper left", fontsize=9)
    else:
        bx.axis("off")
    fig.tight_layout()
    fig.savefig(FIG_FILE, dpi=200)
    print("saved", FIG_FILE)


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "mps":
        print(json.dumps(child_mps(*map(int, a[1:5]))))
    elif a and a[0] == "sv":
        print(json.dumps(child_sv(*map(int, a[1:4]))))
    elif a and a[0] == "report":
        report()
    else:
        run_all(with_sv="--sv" in a)
        report()
