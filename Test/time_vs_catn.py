# Wall-clock timing: our final kernel (main_redblack.py, B4) vs. CATN, on identical circuits and D_hat/chi_hat.
# Unlike catn_compare.py there is NO tracemalloc here (it inflated the times several-fold), so these are
# the numbers for the report. Circuit loading and the CATN wrapper are reused from catn_compare.py.
# Each run is a child process (fresh interpreter, no warm caches shared between codes), REPEATS times;
# the minimum is reported (least disturbed by the OS). A series stops at its first timeout.
#   python time_vs_catn.py --catn C:\path\to\catn      # runs (resumes from time_vs_catn.csv), then the table
#   python time_vs_catn.py report
import argparse, csv, json, os, subprocess, sys, time
import numpy as np

DHEAD, XHEAD = 50, 500
REPEATS = 3
TIME_LIMIT = 1800
CSV_FILE = "time_vs_catn.csv"
REPORT_FILE = "time_vs_catn_report.md"
FIELDS = ["family", "circuit", "n_qubits", "code", "rep", "status", "seconds", "log_abs_amp", "err"]
SERIES = [("ours", [f"{L}:8:0" for L in (10, 20, 30, 40)]),
          ("pz", [f"supremacy2d{n}-d10-seed1-rank3-new" for n in (100, 400, 900, 1600)])]
CODES = ["ours", "catn"]


def child(family, circuit, code, catn_dir):
    import catn_compare as cc
    A, legs = cc.build(family, circuit, catn_dir)            # loading is NOT timed
    t0 = time.perf_counter()
    if code == "ours":
        amp, err = cc.run_ours(A, legs, DHEAD, XHEAD)
    else:
        amp, err = cc.run_catn(A, legs, DHEAD, XHEAD, catn_dir)   # includes CATN's own setup (part of its cost)
    sec = time.perf_counter() - t0
    return dict(seconds=sec, err=err, log_abs_amp=float(np.log(abs(amp))) if amp != 0 else float("-inf"))


def n_of(family, circuit):
    return int(circuit.split("supremacy2d")[1].split("-")[0]) if family == "pz" else int(circuit.split(":")[0]) ** 2


def load_rows():
    return list(csv.DictReader(open(CSV_FILE))) if os.path.exists(CSV_FILE) else []


def run_all(catn_dir):
    rows = load_rows()
    done = {(r["family"], r["circuit"], r["code"], r["rep"]) for r in rows}
    stopped = {(r["family"], r["code"]) for r in rows if r["status"] != "ok"}
    for family, circuits in SERIES:
        for code in CODES:
            if (family, code) in stopped:
                print(f"[{family} / {code}] stopped earlier (timeout/crash), skipping")
                continue
            for circ in circuits:
                failed = False
                for rep in range(REPEATS):
                    if (family, circ, code, str(rep)) in done:
                        continue
                    row = dict(family=family, circuit=circ, n_qubits=n_of(family, circ), code=code, rep=rep)
                    cmd = [sys.executable, __file__, "one", family, circ, code, catn_dir]
                    try:
                        out = subprocess.run(cmd, capture_output=True, text=True, timeout=TIME_LIMIT)
                        if out.returncode != 0:
                            row.update(status="crash")
                            print((out.stderr or out.stdout)[-800:])
                        else:
                            row.update(status="ok", **json.loads(out.stdout.strip().splitlines()[-1]))
                    except subprocess.TimeoutExpired:
                        row.update(status="timeout", seconds=TIME_LIMIT)
                    new = not os.path.exists(CSV_FILE)
                    with open(CSV_FILE, "a", newline="") as f:
                        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
                        if new:
                            w.writeheader()
                        w.writerow(row)
                    print(f"[{family:4s} n={row['n_qubits']:5d}] {code:4s} rep {rep}: {row['status']:7s} "
                          f"{float(row.get('seconds', 0)):8.2f} s", flush=True)
                    if row["status"] != "ok":
                        failed = True
                        break
                if failed:
                    print("  -> series stops here")
                    break


def report():
    rows = [r for r in load_rows() if r["status"] == "ok"]
    out = [f"# Wall-clock time: our kernel (B4) vs. CATN\n",
           f"D̂ = {DHEAD}, χ̂ = {XHEAD}. Minimum over {REPEATS} runs, each in a fresh process, no profiler/tracemalloc. "
           "Circuit loading not timed; CATN's network setup is included.\n",
           "| circuits | qubits | ours [s] | CATN [s] | CATN / ours | Δ ln\\|amp\\| |", "|---|---|---|---|---|---|"]
    for family, circuits in SERIES:
        for circ in circuits:
            t, z = {}, {}
            for code in CODES:
                rr = [r for r in rows if r["family"] == family and r["circuit"] == circ and r["code"] == code]
                if rr:
                    t[code] = min(float(r["seconds"]) for r in rr)
                    z[code] = float(rr[0]["log_abs_amp"])
            if not t:
                continue
            fmt = lambda c: f"{t[c]:.2f}" if c in t else "–"
            ratio = f"{t['catn'] / t['ours']:.1f}×" if len(t) == 2 else "–"
            dz = f"{abs(z['ours'] - z['catn']):.1e}" if len(z) == 2 else "–"
            name = "authors (d10)" if family == "pz" else "ours (d8)"
            out.append(f"| {name} | {n_of(family, circ)} | {fmt('ours')} | {fmt('catn')} | {ratio} | {dz} |")
    out.append("\nΔ ln|amp| is only meaningful for n ≤ 400 on our circuits (L30/L40: double-precision cancellation).")
    text = "\n".join(out)
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(text)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "one":
        fam, circ, code, cdir = sys.argv[2:6]
        print(json.dumps(child(fam, circ, code, cdir)))
    elif len(sys.argv) > 1 and sys.argv[1] == "report":
        report()
    else:
        ap = argparse.ArgumentParser()
        ap.add_argument("--catn", required=True, help="path to the cloned catn repository")
        catn_dir = os.path.abspath(ap.parse_args().catn)
        if not os.path.exists(os.path.join(catn_dir, "tensor_network_np.py")):
            sys.exit(f"--catn {catn_dir} does not look like the catn repository")
        run_all(catn_dir)
        report()
