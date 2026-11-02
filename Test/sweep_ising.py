# 16x16 Ising (h=0): relative error of ln Z vs beta, MPS calculus vs exact Kac-Ward
# Reproduces Fig. 4(a) of Pan et al. (2020): D_hat = 50, chi_hat = 500 in the paper.
# Run from the project root:  python sweep_ising.py
import csv
import time
import numpy as np
import matplotlib.pyplot as plt

from main_redblack import lexicographicmapping, createIDs, defineTensors, MPSCalculus
from Test.kac_ward import kacWardLogZ

# ----------------------------------------------------------------- settings
ROWS, COLS = 16, 16
J = 1.0
BETAS = np.round(np.arange(0.1, 1.01, 0.1), 2)
DHEADS = [2, 4, 8, 16, 50]                     # 50 = paper setting (Fig. 4a)
XHEAD = 500
BETA_C = 0.5 * np.log(1 + np.sqrt(2))        # 0.4407, critical point
FLOOR = 1e-16                                  # plot floor for exact-to-machine results


def build(rows, cols, K):
    a = np.sqrt(np.cosh(K) / 2.0)
    b = np.sqrt(np.sinh(K) / 2.0)
    B = np.array([[a + b, a - b], [a - b, a + b]])
    x, y = lexicographicmapping(rows, cols)
    legs = createIDs(x, y, rows, cols)
    return defineTensors(B, legs), legs


# -------------------------------------------------------------------- sweep
rows_out = []
for beta in BETAS:
    K = beta * J
    lnZ_exact = kacWardLogZ(ROWS, COLS, K)
    for dhead in DHEADS:
        A, legs = build(ROWS, COLS, K)          # legs is mutated -> rebuild every run
        t0 = time.time()
        lnZ, acc_err = MPSCalculus(A, legs, dhead, XHEAD)
        rel_err = abs(lnZ - lnZ_exact) / abs(lnZ_exact)
        rows_out.append(dict(beta=beta, dhead=dhead, lnZ=lnZ, lnZ_exact=lnZ_exact,
                             rel_err=rel_err, acc_err=acc_err, seconds=time.time() - t0))
        print(f"beta={beta:.1f} dhead={dhead:>2}: rel.err={rel_err:.2e}  "
              f"acc.err={acc_err:.2e}  ({rows_out[-1]['seconds']:.1f}s)", flush=True)

with open("ising_sweep_16x16.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows_out[0].keys())
    w.writeheader()
    w.writerows(rows_out)

# --------------------------------------------------------------------- plot
# Ordinal blue ramp (light -> dark = small -> large dhead), validated for a light surface
COLORS = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
MARKERS = ["o", "s", "^", "D", "v"]

plt.rcParams.update({"font.family": "sans-serif", "font.size": 10,
                     "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2,
                     "xtick.color": MUTED, "ytick.color": MUTED})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4), facecolor=SURFACE,
                               gridspec_kw={"width_ratios": [1.35, 1]})


def style(ax, title, ylabel):
    ax.set_facecolor(SURFACE)
    ax.axvline(BETA_C, color=MUTED, lw=1.2, ls="--", zorder=0)
    ax.text(BETA_C, 0.98, " $\\beta_c$", transform=ax.get_xaxis_transform(),
            va="top", ha="left", color=INK2, fontsize=9)
    ax.set_yscale("log")
    ax.set_xlabel("$\\beta$", color=INK2)
    ax.set_xticks(BETAS[::2])
    ax.grid(axis="y", color=GRID, lw=0.8, ls="-")
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.set_ylabel(ylabel, color=INK2)


# left: the paper's quantity -- relative error of ln Z (i.e. of the free energy)
placed = []
for d, c, mk in zip(DHEADS, COLORS, MARKERS):
    sel = [r for r in rows_out if r["dhead"] == d]
    xs = [r["beta"] for r in sel]
    ys = [max(r["rel_err"], FLOOR) for r in sel]
    ax1.plot(xs, ys, color=c, lw=2, marker=mk, ms=6, mec=SURFACE, mew=1.2,
             label=f"$\\hat D$ = {d}")
    # direct label at the line end; nudge apart labels whose end values coincide
    dy = 0
    while any(abs(np.log10(ys[-1]) - p) < 0.4 and abs(dy - q) < 9 for p, q in placed):
        dy -= 10
    placed.append((np.log10(ys[-1]), dy))
    ax1.annotate(f"$\\hat D$={d}", (xs[-1], ys[-1]), xytext=(6, dy),
                 textcoords="offset points", va="center", fontsize=9, color=INK2)
style(ax1, "Free energy vs exact Kac-Ward", "Relative error of ln Z")
ax1.set_xlim(0.05, 1.15)

# right: how good is the a-priori estimate?  ratio  eps_SVD / |ln Z - ln Z_KW|
for d, c, mk in zip(DHEADS, COLORS, MARKERS):
    sel = [r for r in rows_out if r["dhead"] == d
           and r["rel_err"] > 1e-9]       # only where truncation, not rounding, dominates
    if not sel:
        continue
    xs = [r["beta"] for r in sel]
    ys = [r["acc_err"] / abs(r["lnZ"] - r["lnZ_exact"]) for r in sel]
    ax2.plot(xs, ys, color=c, lw=2, marker=mk, ms=6, mec=SURFACE, mew=1.2)
ax2.axhline(1.0, color=INK2, lw=1)
ax2.text(1.0, 1.0, "estimate = actual ", transform=ax2.get_yaxis_transform(),
         ha="right", va="bottom", fontsize=8, color=INK2)
style(ax2, "Error estimate $\\varepsilon_{SVD}$ / actual |$\\Delta$ ln Z|", "Ratio")
ax2.set_xlim(0.05, 1.05)

fig.suptitle(f"MPS calculus, {ROWS}x{COLS} Ising (h = 0), $\\hat\\chi$ = {XHEAD}",
             x=0.01, ha="left", fontsize=12, color=INK)
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, frameon=False, fontsize=9, labelcolor=INK2,
           loc="upper right", ncol=5, bbox_to_anchor=(0.99, 1.0))
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig("ising_sweep_16x16.png", dpi=200, facecolor=SURFACE)
print("saved ising_sweep_16x16.png and ising_sweep_16x16.csv")
