# AI usage is avaliable for this particular code!

# Random-circuit sweep for MPS calculus: how far can one machine go?
#   Sweep A ("Fig. 5"):  L grows at fixed depth d   -> time / memory vs. #qubits
#   Sweep B ("depth"):   d grows at fixed L         -> memory and eps_SVD explode
# Every run is a separate process with a time limit, so a blow-up never kills the sweep.
# Results are appended to circuits_sweep.csv; re-running skips finished rows (resume).
# At the end everything is collected in a report:
#   circuits_sweep_report.md   (setup, machine, tables, limits, validation, figure)
#   circuits_sweep.png         (figure)
#   circuits_sweep.csv         (raw data)
#   circuits_sweep_meta.json   (machine info + session times, used by the report)
#
#   python sweep_circuits.py            # run both sweeps, then figure + report
#   python sweep_circuits.py plot       # only the figure from the CSV
#   python sweep_circuits.py report     # only figure + report from the CSV
import csv, datetime, json, os, platform, subprocess, sys, time, tracemalloc
import numpy as np

# ------------------------------------------------------------------ settings
DHEAD, XHEAD = 50, 500              # paper: D_hat = 50, chi_hat = 500
TIME_LIMIT = 3600                   # seconds per single run (paper: "in an hour")
MEM_LIMIT_GB = 56                   # stop a sweep once a run's peak exceeds this
SEED = 0
SWEEP_A = dict(d=8, Ls=[4, 5, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32, 36, 40])
SWEEP_B = dict(Ls=[5, 6, 8], ds=[8, 10, 12, 14, 16, 18, 20, 22, 24])
SV_MAX_QUBITS = 16                  # state-vector reference only up to here (cost ~ 2^n per gate)
CSV_FILE = "circuits_sweep.csv"
META_FILE = "circuits_sweep_meta.json"
REPORT_FILE = "circuits_sweep_report.md"
FIG_FILE = "circuits_sweep.png"
FIELDS = ["sweep", "L", "n_qubits", "d", "dhead", "xhead", "status", "seconds",
          "peak_MB", "eps", "lnZ_re", "lnZ_im", "sv_abs_err", "n_gates", "n_tensors", "note"]


# --------------------------------------------------------- one run (child)
def single_run(L, d, dhead, xhead):
    from TensorPrimitive.circuits import randomCircuit
    from TensorPrimitive.network import circuitToEdges, edgesToLegs, stateVectorAmplitude
    from Source.main_redblack import MPSCalculus

    n = L * L
    circuit = randomCircuit(L, d, seed=SEED)
    x = [0] * n                                         # amplitude <0...0|C|0...0>
    A, el = circuitToEdges(circuit, x)
    legs = edgesToLegs(el)

    tracemalloc.start()
    t0 = time.time()
    lnZ, eps = MPSCalculus(A, legs, dhead, xhead)
    seconds = time.time() - t0
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    sv_err = ""
    if n <= SV_MAX_QUBITS:
        ref = stateVectorAmplitude(circuit, x)
        sv_err = abs(np.exp(lnZ) - ref) / max(abs(ref), 1e-300)
    return dict(seconds=seconds, peak_MB=peak / 1e6, eps=eps,
                lnZ_re=float(np.real(lnZ)), lnZ_im=float(np.imag(lnZ)), sv_abs_err=sv_err,
                n_gates=len(circuit), n_tensors=len(A))


# ----------------------------------------------------------- driver (parent)
def done_rows():
    if not os.path.exists(CSV_FILE):
        return set()
    with open(CSV_FILE) as f:
        return {(r["sweep"], int(r["L"]), int(r["d"])) for r in csv.DictReader(f)}


def append_row(row):
    new = not os.path.exists(CSV_FILE)
    with open(CSV_FILE, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def launch(sweep, L, d):
    cmd = [sys.executable, __file__, "single", str(L), str(d), str(DHEAD), str(XHEAD)]
    row = dict(sweep=sweep, L=L, n_qubits=L * L, d=d, dhead=DHEAD, xhead=XHEAD)
    t0 = time.time()
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=TIME_LIMIT)
        if out.returncode != 0:
            last = (out.stderr.strip().splitlines() or ["unknown error"])[-1]
            # wall time until the crash (includes Python start-up; peak memory unknown)
            row.update(status="crash", seconds=round(time.time() - t0, 1), note=last[:200])
            print(out.stderr[-600:])
        else:
            row.update(status="ok", **json.loads(out.stdout.strip().splitlines()[-1]))
    except subprocess.TimeoutExpired:
        row.update(status="timeout", seconds=TIME_LIMIT, note=f"exceeded {TIME_LIMIT} s")
    append_row(row)
    peak = row.get("peak_MB")
    peak_s = f"{float(peak):9.1f} MB" if peak not in (None, "") else "      n/a   "
    print(f"[{sweep}] L={L:>2} n={L*L:>4} d={d:>2}: {row['status']:7s} "
          f"t={float(row.get('seconds', 0)):8.1f}s  peak={peak_s}  "
          f"eps={row.get('eps', '') if row['status'] == 'ok' else 'n/a'}", flush=True)
    return row


def system_info():
    info = dict(os=f"{platform.system()} {platform.release()}", machine=platform.machine(),
                cpu=platform.processor() or "unknown", cores_logical=os.cpu_count(),
                python=platform.python_version(), numpy=np.__version__)
    try:
        import psutil
        info["cores_physical"] = psutil.cpu_count(logical=False)
        info["ram_GB"] = round(psutil.virtual_memory().total / 1e9, 1)
    except ImportError:
        info["ram_GB"] = "unknown (pip install psutil)"
    try:
        blas = np.show_config(mode="dicts")["Build Dependencies"]["blas"]
        info["blas"] = f"{blas.get('name', '?')} {blas.get('version', '')}".strip()
    except Exception:
        info["blas"] = "unknown"
    threads = {k: os.environ[k] for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
               if k in os.environ}
    info["thread_env"] = threads if threads else "not set (library default)"
    return info


def load_meta():
    if os.path.exists(META_FILE):
        with open(META_FILE) as f:
            return json.load(f)
    return dict(system=system_info(), sessions=[], stops={})


def save_meta(meta):
    with open(META_FILE, "w") as f:
        json.dump(meta, f, indent=2)


def stop_reason(row):
    if row["status"] != "ok":
        return f"{row['status']} ({row.get('note', '')})"
    if row["peak_MB"] / 1e3 > MEM_LIMIT_GB:
        return f"memory {row['peak_MB'] / 1e3:.1f} GB > {MEM_LIMIT_GB} GB"
    return None


def run_sweeps():
    meta = load_meta()
    meta["sessions"].append(dict(start=datetime.datetime.now().isoformat(timespec="seconds")))
    save_meta(meta)
    finished = done_rows()
    # A run that did not finish "ok" (crash/timeout) is a stop point even if the meta file lost it
    if os.path.exists(CSV_FILE):
        with open(CSV_FILE) as f:
            for r in csv.DictReader(f):
                if r["status"] != "ok":
                    key = "A" if r["sweep"] == "A" else f"B_L{r['L']}"
                    meta["stops"].setdefault(key, dict(L=int(r["L"]), d=int(r["d"]),
                                                      reason=f"{r['status']} ({r['note']})"))
        save_meta(meta)
    current = None                       # (key, L, d) of the run in progress, for Ctrl+C
    try:
        # Sweep A: grow L at fixed depth; stop at the first run that hits a limit
        if "A" in meta["stops"]:
            print("sweep A already stopped earlier:", meta["stops"]["A"]["reason"])
        else:
            for L in SWEEP_A["Ls"]:
                if ("A", L, SWEEP_A["d"]) in finished:
                    continue
                current = ("A", L, SWEEP_A["d"])
                row = launch("A", L, SWEEP_A["d"])
                reason = stop_reason(row)
                if reason:
                    meta["stops"]["A"] = dict(L=L, d=SWEEP_A["d"], reason=reason)
                    save_meta(meta)
                    print("  -> sweep A stops here:", reason)
                    break
        # Sweep B: grow depth at fixed L
        for L in SWEEP_B["Ls"]:
            if f"B_L{L}" in meta["stops"]:
                print(f"sweep B, L={L} already stopped earlier:", meta["stops"][f"B_L{L}"]["reason"])
                continue
            for d in SWEEP_B["ds"]:
                if ("B", L, d) in finished:
                    continue
                current = (f"B_L{L}", L, d)
                row = launch("B", L, d)
                reason = stop_reason(row)
                if reason:
                    meta["stops"][f"B_L{L}"] = dict(L=L, d=d, reason=reason)
                    save_meta(meta)
                    print(f"  -> sweep B, L={L} stops here:", reason)
                    break
    except KeyboardInterrupt:
        if current:
            key, L, d = current
            meta["stops"][key] = dict(L=L, d=d, reason="stopped manually (Ctrl+C)")
            print(f"\n  -> {key} stopped manually at L={L}, d={d}")
    meta["sessions"][-1]["end"] = datetime.datetime.now().isoformat(timespec="seconds")
    save_meta(meta)


# ---------------------------------------------------------------- plot
def plot():
    import matplotlib.pyplot as plt
    rows = [r for r in csv.DictReader(open(CSV_FILE)) if r["status"] == "ok"]
    SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
    BLUE, ORANGE = "#2a78d6", "#eb6834"
    ramp = ["#86b6ef", "#2a78d6", "#0d366b"]
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#c3c2b7",
                         "xtick.color": MUTED, "ytick.color": MUTED})
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.2), facecolor=SURFACE)

    def style(ax, title, xl, yl):
        ax.set_facecolor(SURFACE); ax.set_yscale("log"); ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
        ax.set_title(title, loc="left", fontsize=10, color=INK)
        ax.set_xlabel(xl, color=INK2); ax.set_ylabel(yl, color=INK2)

    A = sorted([r for r in rows if r["sweep"] == "A"], key=lambda r: int(r["n_qubits"]))
    n = [int(r["n_qubits"]) for r in A]
    axs[0].plot(n, [float(r["seconds"]) for r in A], color=BLUE, lw=2, marker="o", label="time [s]")
    axs[0].plot(n, [max(float(r["peak_MB"]), 1e-3) for r in A], color=ORANGE, lw=2, marker="s", label="peak memory [MB]")
    axs[0].legend(frameon=False, fontsize=9, labelcolor=INK2)
    style(axs[0], f"Sweep A: depth {SWEEP_A['d']}, growing lattice", "number of qubits  n = L²", "")

    for L, c in zip(SWEEP_B["Ls"], ramp):
        B = sorted([r for r in rows if r["sweep"] == "B" and int(r["L"]) == L], key=lambda r: int(r["d"]))
        if not B: continue
        dd = [int(r["d"]) for r in B]
        axs[1].plot(dd, [max(float(r["peak_MB"]), 1e-3) for r in B], color=c, lw=2, marker="o", label=f"L = {L}")
        axs[2].plot(dd, [max(float(r["eps"]), 1e-30) for r in B], color=c, lw=2, marker="o", label=f"L = {L}")
    axs[1].legend(frameon=False, fontsize=9, labelcolor=INK2)
    style(axs[1], "Sweep B: peak memory vs depth", "depth d", "peak memory [MB]")
    style(axs[2], "Sweep B: truncation estimate vs depth", "depth d", "ε_SVD (accumulated)")
    fig.suptitle(f"MPS calculus on random circuits, D̂ = {DHEAD}, χ̂ = {XHEAD}", x=0.01, ha="left", color=INK)
    fig.tight_layout()
    fig.savefig(FIG_FILE, dpi=200, facecolor=SURFACE)
    print("saved", FIG_FILE)


# --------------------------------------------------------------- report
def _f(x, fmt):
    try:
        return format(float(x), fmt)
    except (TypeError, ValueError):
        return "–"


def _quality(eps):
    try:
        e = float(eps)
    except (TypeError, ValueError):
        return "–"
    if e < 1e-10:
        return "exact"
    if e < 1e-3:
        return "truncated, reliable"
    return "UNRELIABLE"


def _stop_for(meta, key, rows, par, planned):
    """Earliest stop point, from the CSV (crash/timeout rows) or the meta file (memory, Ctrl+C)."""
    cands = []
    stop = meta.get("stops", {}).get(key)
    if stop:
        cands.append((int(stop[par]), stop["reason"]))
    for r in rows:
        if r["status"] != "ok":
            cands.append((int(r[par]), f"{r['status']} ({r['note']})"))
    if cands:
        return min(cands, key=lambda c: c[0])
    done = {int(r[par]) for r in rows}
    left = [v for v in planned if v > max(done)] if done else list(planned)
    if left:
        return left[0], "not run: sweep was interrupted before this point"
    return None


def report():
    rows = list(csv.DictReader(open(CSV_FILE)))
    meta = load_meta()
    ok = [r for r in rows if r["status"] == "ok"]
    out = []
    w = out.append

    w("# MPS calculus on random quantum circuits: sweep report\n")
    w(f"Generated {datetime.datetime.now().isoformat(timespec='seconds')}. "
      f"Raw data: `{CSV_FILE}`.\n")

    # 1. setup
    w("## 1. Setup\n")
    w("| Setting | Value |\n|---|---|")
    w(f"| Max physical bond dimension D̂ | {DHEAD} |")
    w(f"| Max virtual bond dimension χ̂ | {XHEAD} |")
    w(f"| Time limit per run | {TIME_LIMIT} s |")
    w(f"| Memory limit (stops a sweep) | {MEM_LIMIT_GB} GB |")
    w(f"| Circuit | Pan et al. protocol: H layer, d CZ layers (Fig. 13 layouts, order 1,3,5,7,2,4,6,8), "
      f"random T / X^½ / Y^½ on idle qubits, H layer; seed {SEED} |")
    w("| Quantity | single amplitude ⟨0…0│C│0…0⟩ |")
    w(f"| State-vector check | for n ≤ {SV_MAX_QUBITS} qubits |\n")

    # 2. machine
    w("## 2. Machine\n")
    w("| Item | Value |\n|---|---|")
    for k, v in meta.get("system", {}).items():
        w(f"| {k} | {v} |")
    for i, s in enumerate(meta.get("sessions", []), 1):
        w(f"| session {i} | {s.get('start', '?')} → {s.get('end', 'interrupted')} |")
    w("")

    # 3. sweep A
    A = sorted([r for r in rows if r["sweep"] == "A"], key=lambda r: int(r["n_qubits"]))
    w(f"## 3. Sweep A: growing lattice at depth {SWEEP_A['d']}\n")
    if A:
        w("| L | qubits | gates | tensors | status | time [s] | peak mem [MB] | ε_SVD | quality | rel. err vs state vector |")
        w("|---|---|---|---|---|---|---|---|---|---|")
        for r in A:
            w(f"| {r['L']} | {r['n_qubits']} | {r['n_gates'] or '–'} | {r['n_tensors'] or '–'} | {r['status']} | "
              f"{_f(r['seconds'], '.1f')} | {_f(r['peak_MB'], '.1f')} | {_f(r['eps'], '.1e')} | "
              f"{_quality(r['eps']) if r['status'] == 'ok' else '–'} | {_f(r['sv_abs_err'], '.1e')} |")
        okA = [r for r in A if r["status"] == "ok"]
        w("")
        if okA:
            big = max(okA, key=lambda r: int(r["n_qubits"]))
            w(f"- Largest circuit completed: **{big['n_qubits']} qubits** (L = {big['L']}) "
              f"in {_f(big['seconds'], '.1f')} s, peak {_f(big['peak_MB'], '.1f')} MB.")
            fit = [r for r in okA if int(r["n_qubits"]) >= 16 and float(r["seconds"]) > 0]
            if len(fit) >= 3:
                p = np.polyfit(np.log([int(r["n_qubits"]) for r in fit]),
                               np.log([float(r["seconds"]) for r in fit]), 1)[0]
                w(f"- Empirical time scaling over this range: time ∝ n^{p:.2f}.")
        stop = _stop_for(meta, "A", A, "L", SWEEP_A["Ls"])
        w(f"- Stop point: L = {stop[0]}: {stop[1]}." if stop else "- No limit reached.")
        w("")
    else:
        w("No data.\n")

    # 4. sweep B
    w("## 4. Sweep B: growing depth at fixed lattice\n")
    for L in SWEEP_B["Ls"]:
        B = sorted([r for r in rows if r["sweep"] == "B" and int(r["L"]) == L], key=lambda r: int(r["d"]))
        if not B:
            continue
        w(f"### L = {L} ({L * L} qubits)\n")
        w("| depth | gates | status | time [s] | peak mem [MB] | ε_SVD | quality | rel. err vs state vector |")
        w("|---|---|---|---|---|---|---|---|")
        for r in B:
            w(f"| {r['d']} | {r['n_gates'] or '–'} | {r['status']} | {_f(r['seconds'], '.1f')} | "
              f"{_f(r['peak_MB'], '.1f')} | {_f(r['eps'], '.1e')} | "
              f"{_quality(r['eps']) if r['status'] == 'ok' else '–'} | {_f(r['sv_abs_err'], '.1e')} |")
        okB = [r for r in B if r["status"] == "ok"]
        w("")
        if okB:
            w(f"- Deepest circuit completed: d = {max(int(r['d']) for r in okB)}.")
            bad = [int(r["d"]) for r in okB if _quality(r["eps"]) == "UNRELIABLE"]
            w(f"- First depth with ε_SVD ≥ 1e-3 (truncation no longer negligible): d = {min(bad)}."
              if bad else f"- ε_SVD stays below 1e-3 for all completed depths.")
        stop = _stop_for(meta, f"B_L{L}", B, "d", SWEEP_B["ds"])
        w(f"- Stop point: d = {stop[0]}: {stop[1]}." if stop else "- No limit reached.")
        w("")

    # 5. validation
    w("## 5. Validation against the state vector\n")
    val = [r for r in ok if r["sv_abs_err"] not in ("", None)]
    if val:
        worst = max(float(r["sv_abs_err"]) for r in val)
        w(f"{len(val)} runs with n ≤ {SV_MAX_QUBITS} qubits were checked against an independent "
          f"state-vector simulation. Largest relative error: **{worst:.1e}**.\n")
    else:
        w("No run was small enough for a state-vector check.\n")

    # 6. figure
    w("## 6. Figure\n")
    w(f"![sweep]({FIG_FILE})\n")

    # 7. notes
    w("## 7. How to read this\n")
    w("- **ε_SVD** is the accumulated discarded weight Σδ of all truncations (swaps and D̂ cuts). "
      "It is an a-priori estimate, not a bound; on the 16×16 Ising benchmark it tracked the true error "
      "within roughly one order of magnitude. 'exact' < 1e-10, 'reliable' < 1e-3. "
      "Because it is a sum over many truncations it can exceed 1; values that large only say that "
      "truncation dominated the run, not by how much the amplitude is off.")
    w("- **Peak memory** is measured with `tracemalloc` (all Python and NumPy allocations of the run); "
      "the interpreter's own baseline is not included.")
    w("- **Time** is wall-clock for `MPSCalculus` only (building the network is excluded). "
      "For crashed runs it is the wall time until the crash, including start-up; their peak memory is unknown.")
    w("- Each point is a single random circuit (one seed) and a single amplitude.")
    w("- The CZ layout order is not specified in the paper; the order used here alternates horizontal and "
      "vertical layouts so that consecutive layers never act on the same pair.")

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print("saved", REPORT_FILE)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "single":
        L, d, dh, xh = map(int, sys.argv[2:6])
        print(json.dumps(single_run(L, d, dh, xh)))
    elif len(sys.argv) > 1 and sys.argv[1] == "plot":
        plot()
    elif len(sys.argv) > 1 and sys.argv[1] == "report":
        plot()
        report()
    else:
        run_sweeps()
        plot()
        report()
