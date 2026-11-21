# Stage 3 table: true error |ln|amp| - ln|amp|_ref| of our code and of CATN at D_max = 50, chi_max = 500.
# Reference per circuit = our run with the smallest eps_SVD in Data/validation/3_dmax_reference.csv
# (eps ~ 0: exact). For the authors' n = 1600 circuit our greedy order is not exact even at D_max = 200
# (eps 1.8e-8); there the exact value from CATN's order replayed in our kernel is used (D_max >= 128, eps ~1e-27).
#   python stage3_table.py            -> prints the table, writes Data/validation/3_true_error.csv
import csv, os
HERE = os.path.dirname(os.path.abspath(__file__))
V = os.path.join(HERE, "Data", "validation")
EXACT_PZ1600 = -553.5334527020           # figdata/dmax_sweep_n1600.csv: CATN order, D_max = 128, eps 9.7e-28

refrows = list(csv.DictReader(open(os.path.join(V, "3_dmax_reference.csv"))))
runs = list(csv.DictReader(open(os.path.join(V, "2_vs_catn.csv"))))
out = []
for fam, circ in dict.fromkeys((r["family"], r["circuit"]) for r in refrows):
    rr = [r for r in refrows if (r["family"], r["circuit"]) == (fam, circ)]
    best = min(rr, key=lambda r: float(r["eps"]))
    ref, ref_eps, ref_src = float(best["ln_abs_amp"]), float(best["eps"]), f"ours D_max={best['dmax']}"
    if fam == "pz" and circ.startswith("supremacy2d1600"):
        ref, ref_eps, ref_src = EXACT_PZ1600, 0.0, "CATN order in our kernel, D_max=128"
    row = dict(family=fam, circuit=circ, n_qubits=best["n_qubits"], ref_ln_abs_amp=f"{ref:.10f}",
               ref_eps=f"{ref_eps:.2e}", ref_source=ref_src)
    for code in ("ours", "catn"):
        c = [r for r in runs if (r["family"], r["circuit"], r["code"]) == (fam, circ, code) and r["status"] == "ok"]
        if c:
            row[f"{code}_ln_abs_amp"] = f"{float(c[0]['log_abs_amp']):.10f}"
            row[f"{code}_err_reported"] = f"{float(c[0]['err']):.2e}"
            row[f"{code}_true_error"] = f"{abs(float(c[0]['log_abs_amp']) - ref):.2e}"
    out.append(row)
fields = list(dict.fromkeys(k for r in out for k in r))
with open(os.path.join(V, "3_true_error.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(out)
print(f"{'circuits':8s} {'n':>5s}  {'reference (source, eps)':44s} {'ours eps':>9s} {'ours true':>9s} {'CATN true':>9s}")
for r in out:
    print(f"{r['family']:8s} {r['n_qubits']:>5s}  {r['ref_ln_abs_amp']:>16s} ({r['ref_source']}, {r['ref_eps']})"[:62].ljust(62)
          + f" {r.get('ours_err_reported','-'):>9s} {r.get('ours_true_error','-'):>9s} {r.get('catn_true_error','-'):>9s}")
