# AI usage is avaliable for this particular code!

# Record the order in which CATN contracts the authors' circuit, as (i, j) = "tensor i eats tensor j".
# The indices are positions in the tensor list A that catn_compare.build() returns, the same list our
# kernel gets, so the recorded order can be replayed in MPSCalculus (next step).
# CATN runs exactly as in exp_gap / catn_compare (published settings, same two NumPy-2 patches);
# only MPSNode.eat is wrapped to log its two operands, nothing else changes.
#   python exp_order.py --catn /path/to/catn --n 100     # record (seconds)
#   python exp_order.py --catn /path/to/catn --n 900     # record (~2 min) -> order_900.json
#   python exp_order.py --catn /path/to/catn --replay order_900.json
#   python exp_order.py --catn /path/to/catn --replay order_1600.json --dheads 50,64,80,100
#       D_max sweep (greedy and CATN order only, chi_max stays 500): which D_max does each order need?
# Replay runs OUR kernel (main.MPSCalculus, same D_max / chi_max) three times:
#   greedy                our own order (reference, = exp_gap "ours tol=1e-12")
#   CATN order            CATN's pairs, CATN's orientation (i survives, as in CATN)
#   CATN order, min-id    same edges in the same order, but the smaller id survives (our orientation)
# and prints eps (split chi_hat / D_hat), ln|amp| and the first step where our greedy order leaves CATN's.
# Needs main.MPSCalculus(..., order=list of (i, j)).
import argparse, contextlib, io, json, os, sys, types
HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.dirname(HERE), os.path.join(os.path.dirname(HERE), "TensorPrimitive"),
          os.path.join(os.path.dirname(HERE), "Source")):
    if p not in sys.path:
        sys.path.insert(0, p)
import numpy as np

DHEAD, XHEAD = 50, 500
CIRCS = {100: "supremacy2d100-d10-seed1-rank3-new", 900: "supremacy2d900-d10-seed1-rank3-new",
         1600: "supremacy2d1600-d10-seed1-rank3-new"}
SETTINGS = dict(Dmax=DHEAD, chi=XHEAD, reverse=True, cutoff=1e-14, swapopt=True, svdopt=True, select=1)


def load_catn(catn_dir):
    """Both CATN modules, with the same two NumPy-2 patches as exp_gap (algorithm untouched)."""
    sys.path.insert(0, catn_dir)
    def load_patched(name, old, new):
        path = os.path.join(catn_dir, name + ".py")
        src = open(path, encoding="utf-8").read()
        assert src.count(old) == 1, f"unexpected CATN version ({name})"
        mod = types.ModuleType(name); mod.__file__ = path; sys.modules[name] = mod
        exec(compile(src.replace(old, new), path, "exec"), mod.__dict__)
        return mod
    mps_mod = load_patched("mps_node_np", "lognorm = math.log(np.abs(result))",
                           "lognorm = math.log(np.abs(result).item())")
    tn_mod = load_patched("tensor_network_np", "np.array([2])", "np.array(2)")
    return mps_mod, tn_mod


def install_recorder(mps_mod):
    """Wrap MPSNode.eat: log (eater, eaten) before every call, then run the original unchanged."""
    order = []
    original = mps_mod.MPSNode.eat
    def eat(self, other, *args, **kwargs):
        order.append((int(self.index), int(other.index)))
        return original(self, other, *args, **kwargs)
    mps_mod.MPSNode.eat = eat
    return order


def record(catn_dir, n):
    import catn_compare as cc
    A, legs = cc.build("pz", CIRCS[n], catn_dir)
    mps_mod, tn_mod = load_catn(catn_dir)
    order = install_recorder(mps_mod)                      # before the network exists

    tn = tn_mod.Tensor_Network_np([np.asarray(t, dtype=complex) for t in A], [np.array(l) for l in legs],
                                  seed=1, maxdim=10**9, verbose=-1, node_type="mps", norm_method=1, **SETTINGS)
    for node in tn.tensors.values():                       # as in catn_compare.run_catn
        node.neighbor = np.array(node.neighbor)
    tn.select_edge_init()
    with contextlib.redirect_stdout(io.StringIO()):        # CATN prints one progress line per step
        lnZ, error, psi = tn.contraction()
    amp = complex(np.exp(np.asarray(lnZ).ravel()[0]) * np.asarray(psi).ravel()[0])
    lnabs, err = float(np.log(abs(amp))), float(np.linalg.norm(error))

    eaten = [j for _, j in order]
    n_isolated = sum(1 for l in legs if len(l) == 0)
    print(f"n = {n}   tensors = {len(A)}   isolated = {n_isolated}   recorded eats = {len(order)}")
    print(f"ln|amp| = {lnabs:.10f}   CATN err = {err:.2e}")
    print(f"every eaten tensor appears once: {len(set(eaten)) == len(eaten)}")
    print(f"first steps: {order[:5]}")

    out = f"order_{n}.json"
    with open(out, "w") as f:
        json.dump(dict(n=n, circuit=CIRCS[n], settings=SETTINGS, n_tensors=len(A), n_isolated=n_isolated,
                       lnabs=lnabs, err=err, order=[list(e) for e in order]), f)
    print(f"written: {out}")


def min_id_orientation(order):
    """Same edges, same sequence, but the smaller current id survives. CATN's later steps name the
    survivor by CATN's id, so keep a map CATN id -> our id while relabelling."""
    alias, out = {}, []
    for i, j in order:
        a, b = alias.get(i, i), alias.get(j, j)
        s, e = (a, b) if a < b else (b, a)
        out.append((s, e))
        alias[i] = s                                       # CATN keeps calling the merged tensor i
    return out


def run_ours(catn_dir, circuit, order=None, log=None, dhead=DHEAD):
    """Fresh circuit every run: MPSCalculus changes legs in place."""
    import catn_compare as cc, main
    A, legs = cc.build("pz", circuit, catn_dir)
    original = main.mergeMPSTensors
    if log is not None:                                    # record the pairs our run actually contracts
        def merge(A_m, legs_, cano, i, j):
            log.append((int(i), int(j)))
            return original(A_m, legs_, cano, i, j)
        main.mergeMPSTensors = merge
    stats = {}
    try:
        lnZ, eps = main.MPSCalculus(A, legs, dhead, XHEAD, stats=stats, order=order)
    finally:
        main.mergeMPSTensors = original
    return float(np.real(lnZ)), float(eps), stats


def replay(catn_dir, path):
    d = json.load(open(path))
    catn_order = [tuple(e) for e in d["order"]]
    print(f"n = {d['n']}   {d['circuit']}   steps = {len(catn_order)}   CATN: ln|amp| {d['lnabs']:.10f}  err {d['err']:.2e}")
    greedy = []
    runs = [("ours greedy", None, greedy),
            ("ours, CATN order", catn_order, None),
            ("ours, CATN order, min-id", min_id_orientation(catn_order), None)]
    print(f"{'run':28s} {'eps':>9s} {'eps_chi':>9s} {'eps_D':>9s} {'hits_chi':>8s} {'hits_D':>6s} "
          f"{'max_chi':>7s} {'max_D':>5s}  {'ln|amp|':>16s}  {'Δ vs CATN':>9s}")
    for name, order, log in runs:
        lnabs, eps, st = run_ours(catn_dir, d["circuit"], order, log)
        print(f"{name:28s} {eps:9.2e} {st['eps_chi']:9.2e} {st['eps_D']:9.2e} {st['hits_chi']:8d} {st['hits_D']:6d} "
              f"{st['max_chi']:7d} {st['max_D']:5d}  {lnabs:16.10f}  {abs(lnabs - d['lnabs']):9.1e}", flush=True)
    same = [frozenset(a) == frozenset(b) for a, b in zip(greedy, catn_order)]
    first = next((k for k, s in enumerate(same) if not s), None)
    print(f"greedy vs CATN order: {sum(same)} / {len(catn_order)} steps contract the same edge; "
          f"first difference at step {first}" + ("" if first is None else
          f" (ours {greedy[first]}, CATN {catn_order[first]})"))
    oriented = sum(1 for a, b in zip(greedy, catn_order) if a == b)
    print(f"same edge AND same survivor: {oriented} / {len(catn_order)} steps")


def sweep(catn_dir, path, dheads):
    import time
    d = json.load(open(path))
    catn_order = [tuple(e) for e in d["order"]]
    print(f"n = {d['n']}   {d['circuit']}   chi_max = {XHEAD}   D_max sweep {dheads}")
    print(f"{'D_max':>5s}  {'order':12s} {'eps':>9s} {'eps_chi':>9s} {'eps_D':>9s} {'hits_chi':>8s} {'hits_D':>6s} "
          f"{'max_chi':>7s} {'max_D':>5s}  {'ln|amp|':>16s}  {'time':>7s}")
    for dh in dheads:
        for name, order in (("greedy", None), ("CATN order", catn_order)):
            t0 = time.perf_counter()
            lnabs, eps, st = run_ours(catn_dir, d["circuit"], order, None, dh)
            print(f"{dh:5d}  {name:12s} {eps:9.2e} {st['eps_chi']:9.2e} {st['eps_D']:9.2e} {st['hits_chi']:8d} "
                  f"{st['hits_D']:6d} {st['max_chi']:7d} {st['max_D']:5d}  {lnabs:16.10f}  {time.perf_counter() - t0:6.1f}s",
                  flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catn", required=True)
    ap.add_argument("--n", type=int, default=900, choices=sorted(CIRCS))
    ap.add_argument("--replay", metavar="ORDER_JSON")
    ap.add_argument("--dheads", help="comma-separated D_max values: run the D_max sweep instead of the replay table")
    a = ap.parse_args()
    catn_dir = os.path.abspath(a.catn)
    if not os.path.exists(os.path.join(catn_dir, "tensor_network_np.py")):
        sys.exit(f"--catn {catn_dir} does not look like the catn repository")
    if a.replay and a.dheads:
        sweep(catn_dir, a.replay, [int(x) for x in a.dheads.split(",")])
    elif a.replay:
        replay(catn_dir, a.replay)
    else:
        record(catn_dir, a.n)


if __name__ == "__main__":
    main()
