# C2: how much parallelism does the batched MPS calculus have? (ideal machine: no thread overhead)
# For every circuit and batch cap K, one run of main.MPSCalculus with per-step timing:
#   steps, rounds      -> steps/rounds  = parallelism if all steps took equally long
#   W  = sum of all step times            (work: one thread)
#   S  = sum over rounds of the longest step   (span: infinitely many threads)
#   T_serial = selectBatch + heap pushes  (bookkeeping, stays serial -> Amdahl)
#   T_P = sum over rounds of an LPT schedule of the round on P threads, + T_serial
#   speedup_P = (W + T_serial) / T_P ;   upper bound for P -> inf: (W + T_serial) / (S + T_serial)
#   work inflation = W(K) / W(K=1): a different order may make the steps themselves cheaper or dearer
#   accuracy: eps and |ln|amp| - exact| (exact reference only where valid: L <= 20 of our circuits)
# Every (circuit, K) is a fresh child process. Times are from one run (noise ~ few %).
#   python c2_parallelism.py                         # our circuits L = 10, 20, 30, 40
#   python c2_parallelism.py --catn C:\path\to\catn  # + the authors' circuits n = 100 ... 1600
#   python c2_parallelism.py report                  # table + figure from c2_parallelism.json
import argparse, heapq, json, os, subprocess, sys
import numpy as np

KS = [1, 4, 8, 24, None]
PS = [4, 8, 24]
DHEAD, XHEAD = 50, 500
OUT = "c2_parallelism.json"
REF = {"ours 10:8:0": -37.58825906084385, "ours 20:8:0": -146.4967154741864}     # exact, valid references
CIRCUITS = [("ours", f"{L}:8:0") for L in (10, 20, 30, 40)]
PZ = [("pz", f"supremacy2d{n}-d10-seed1-rank3-new") for n in (100, 400, 900, 1600)]


def lpt(times, P):
    # longest processing time first: a good (and standard) schedule of independent tasks on P threads
    load = [0.0] * P
    for t in sorted(times, reverse=True):
        heapq.heapreplace(load, load[0] + t)
    return max(load)


def child(family, circuit, K, catn_dir):
    import catn_compare as cc
    A, legs = cc.build(family, circuit, catn_dir)
    st = {}
    lnZ, eps = main.MPSCalculus(A, legs, DHEAD, XHEAD, stats=st, K=K)
    rt, ser = st["round_times"], st["serial_time"]
    W = sum(map(sum, rt))
    S = sum(max(r) for r in rt)
    res = dict(family=family, circuit=circuit, K=K, steps=sum(map(len, rt)), rounds=len(rt),
               W=W, S=S, serial=ser, eps=float(eps), lnabs=float(np.real(lnZ)),
               hits_D=st["hits_D"], max_batch=max(map(len, rt)),
               T_P={P: sum(lpt(r, P) for r in rt) + ser for P in PS},
               batch_sizes=[len(r) for r in rt] if K is None else None)
    ref = REF.get(f"{family} {circuit}")
    res["abs_err"] = None if ref is None else abs(res["lnabs"] - ref)
    return res


def load():
    return json.load(open(OUT)) if os.path.exists(OUT) else []


def run_all(catn_dir):
    rows = load()
    done = {(r["family"], r["circuit"], r["K"]) for r in rows}
    for family, circ in CIRCUITS + (PZ if catn_dir else []):
        for K in KS:
            if (family, circ, K) in done:
                continue
            cmd = [sys.executable, __file__, "one", family, circ, str(K), catn_dir or "-"]
            out = subprocess.run(cmd, capture_output=True, text=True)
            if out.returncode != 0:
                print(out.stderr[-800:]); sys.exit(f"{family} {circ} K={K} crashed")
            r = json.loads(out.stdout.strip().splitlines()[-1])
            rows.append(r)
            json.dump(rows, open(OUT, "w"))
            print(f"[{family:4s} {circ:34s}] K={str(K):4s}  steps {r['steps']:6d}  rounds {r['rounds']:6d}  "
                  f"W {r['W']:6.2f} s  S {r['S']:6.2f} s  serial {r['serial']:5.2f} s  eps {r['eps']:.1e}", flush=True)


def name(family, circ):
    if family == "pz":
        return f"authors n={circ.split('supremacy2d')[1].split('-')[0]}"
    return f"ours L{circ.split(':')[0]}"


def report():
    rows = load()
    out = ["# C2: parallelism of the batched MPS calculus (ideal machine)\n",
           f"D̂ = {DHEAD}, χ̂ = {XHEAD}. W = Σ step times, S = Σ_rounds max step, serial = selectBatch + pushes. "
           "S_P = (W + serial) / T_P with an LPT schedule per round. ∞ = (W + serial) / (S + serial). "
           "infl. = W(K) / W(K=1).\n",
           "| circuit | K | steps/rounds | max batch | W/S | serial / total | S_4 | S_8 | S_24 | S_∞ | infl. | ε | err |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    w1 = {(r["family"], r["circuit"]): r["W"] for r in rows if r["K"] == 1}
    for r in rows:
        tot = r["W"] + r["serial"]
        sp = {P: tot / r["T_P"][str(P)] if str(P) in r["T_P"] else tot / r["T_P"][P] for P in PS}
        sinf = tot / (r["S"] + r["serial"])
        infl = r["W"] / w1[(r["family"], r["circuit"])] if (r["family"], r["circuit"]) in w1 else float("nan")
        err = "–" if r["abs_err"] is None else f"{r['abs_err']:.1e}"
        out.append(f"| {name(r['family'], r['circuit'])} | {r['K']} | {r['steps'] / r['rounds']:.1f} | {r['max_batch']} | "
                   f"{r['W'] / r['S']:.1f} | {r['serial'] / tot:.0%} | {sp[4]:.1f} | {sp[8]:.1f} | {sp[24]:.1f} | "
                   f"{sinf:.1f} | {infl:.2f} | {r['eps']:.1e} | {err} |")
    text = "\n".join(out)
    open("c2_parallelism_report.md", "w", encoding="utf-8").write(text + "\n")
    print(text)
    figure(rows)


def figure(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    big = [r for r in rows if r["K"] is None and r["batch_sizes"]]
    for r in big:
        ax[0].plot(r["batch_sizes"], lw=1, label=name(r["family"], r["circuit"]))
    ax[0].set(xlabel="round", ylabel="batch size (edges)", yscale="log", title="(a) batch size per round, K = None")
    ax[0].legend(fontsize=8)
    for r in big:
        tot = r["W"] + r["serial"]
        sp = [1.0] + [tot / r["T_P"][str(P)] for P in PS]
        ax[1].plot([1] + PS, sp, "o-", label=name(r["family"], r["circuit"]))
    ax[1].plot([1, 24], [1, 24], "k:", lw=1, label="linear")
    ax[1].set(xlabel="threads P", ylabel="ideal speedup", xscale="log", yscale="log",
              title="(b) ideal speedup (LPT per round + serial part), K = None")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("c2_parallelism.png", dpi=150)
    print("saved c2_parallelism.png")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "one":
        K = None if a[3] == "None" else int(a[3])
        print(json.dumps(child(a[1], a[2], K, None if a[4] == "-" else a[4])))
    elif a and a[0] == "report":
        report()
    else:
        ap = argparse.ArgumentParser()
        ap.add_argument("--catn", default=None)
        catn = ap.parse_args().catn
        catn = os.path.abspath(catn) if catn else None
        run_all(catn)
        report()
