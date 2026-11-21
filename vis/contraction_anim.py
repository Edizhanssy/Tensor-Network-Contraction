from TensorPrimitive.circuits import randomCircuit
from TensorPrimitive.network import circuitToEdges, edgesToLegs
import networkx as nx
import plotly.graph_objects as go
from main import MPSCalculus
from collections import defaultdict
import numpy as np
from plotly.colors import sample_colorscale


def key(a, b):
    return (min(a, b), max(a, b))


def build(owners):
    # fresh bond dictionary w[(a, b)] = bond dimension and neighbour sets nbr[a]
    # (merge changes both in place, so every replay needs a new pair)
    w = {}
    for k1, k2 in owners.values():
        w[key(k1, k2)] = w.get(key(k1, k2), 1) * 2     # parallel legs multiply
    nbr = defaultdict(set)
    for a, b in w:
        nbr[a].add(b)
        nbr[b].add(a)
    return w, nbr


def merge(w, nbr, i, j):
    # j is absorbed into i; returns log2 of the dense size of the new tensor i
    w.pop(key(i, j))                                   # the contracted bond disappears
    nbr[i].remove(j)
    nbr[j].remove(i)
    for k in nbr[j]:
        size = w.pop(key(j, k))                        # bond j-k ...
        nbr[i].add(k)
        w[key(i, k)] = w.get(key(i, k), 1) * size      # ... becomes i-k (multiplies if i-k exists)
        nbr[k].remove(j)                               # k now points to i instead of j
        nbr[k].add(i)
    nbr[j] = set()
    return sum(np.log2(w[key(i, k)]) for k in nbr[i])


def replay(owners, trace):
    # replays the kernel's contraction order on the graph
    # dense[s]  = log2 dense size of the tensor produced at step s (no rank reduction)
    # frames[s] = copy of the bond dictionary after step s (the data of one animation frame)
    w, nbr = build(owners)
    dense, frames = [], []
    for i, j, *_ in trace:
        dense.append(merge(w, nbr, i, j))
        frames.append(dict(w))
    return dense, frames


def network(L, d, seed=0):
    A, el = circuitToEdges(randomCircuit(L, d, seed=seed), [0] * (L * L))
    owners = defaultdict(list)
    for k, edges_k in enumerate(el):
        for e in edges_k:
            owners[e].append(k)
    return A, el, owners


def layout(owners, n_nodes, seed=0):
    # fixed 3D positions for every tensor (computed once, the same in every frame)
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    G.add_edges_from(tuple(e) for e in owners.values())
    return nx.spring_layout(G, dim=3, seed=seed)


def frame_data(s, trace, frames, pos):
    # data of animation frame s (state AFTER step s of the trace)
    #   nodes : alive tensors (survivors; the eaten j's are gone)
    #   xyz   : (len(nodes), 3) positions
    #   size  : log2 dense size of every alive tensor (sum of log2 of its bonds)
    #   edges : {log2 bond dimension: [(p_a, p_b), ...]} - edges grouped by bond size
    #   cur   : the tensor that just absorbed another one (i of step s)
    eaten = {t[1] for t in trace[:s + 1]}
    w = frames[s]
    nodes = [v for v in pos if v not in eaten]
    size = defaultdict(float)
    edges = defaultdict(list)
    for (a, b), dim in w.items():
        lb = int(round(np.log2(dim)))
        size[a] += lb
        size[b] += lb
        edges[lb].append((pos[a], pos[b]))
    xyz = np.array([pos[v] for v in nodes])
    return nodes, xyz, [size[v] for v in nodes], dict(edges), trace[s][0]

def make_traces(s, trace, frames, pos, max_lb, cmax, colors):
    #s = 445
    traces = []
    nodes, xyz, size, edges, cur = frame_data(s, trace, frames, pos)
    #edge_trace = []
    #colors = {1: 'lightgray', 2: 'skyblue', 3: 'royalblue', 4: 'darkblue'}
    for lb in range(1, max_lb + 1):
        pairs = edges.get(lb, [])
        xs = []
        ys = []
        zs = []
        for pa, pb in pairs:
            xs.extend([pa[0], pb[0], None])
            ys.extend([pa[1], pb[1], None])
            zs.extend([pa[2], pb[2], None])

        traces.append(go.Scatter3d(x=xs, y=ys, z=zs, mode='lines', line=dict(width = 1 + 2 * lb, color=colors[lb-1]), name=f"bond 2^{lb}"))

    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]

    mode = 'markers'
    marker = dict(size=10, color=size, cmin=0, cmax=cmax, colorscale='Viridis', colorbar=dict(title='log₂ dense size'))
    name = 'tensors'

    traces.append(go.Scatter3d(x=x, y=y, z=z, mode=mode, marker=marker, name=name))
    px, py, pz = pos[cur]
    traces.append(go.Scatter3d(x=[px], y=[py], z=[pz], mode=mode, marker=dict(size=18, color='rgba(0,0,0,0)', line=dict(color='red', width=4)), name='current tensor'))

    return traces


# ---- size comparison over circuits (results are in the notes, kept for reproduction) ----
def compare_sizes(circuits=((4, 8), (5, 12), (6, 16), (8, 16), (10, 16))):
    for L, d in circuits:
        A, el, owners = network(L, d)
        trace = []
        lnZ, eps = MPSCalculus(A, edgesToLegs(el), 50, 500, trace=trace)
        dense, _ = replay(owners, trace)
        print(f"L{L} d{d}: {len(A)} tensors | dense 2^{max(dense):.1f} | "
              f"ls 2^{max(t[2] for t in trace) / np.log(2):.1f} | "
              f"MPS 2^{np.log2(max(t[3] for t in trace)):.1f} | "
              f"chi {max(t[4] for t in trace)} | eps {eps:.1e}")


if __name__ == '__main__':
    L, d = 6, 16
    A, el, owners = network(L, d)

    # ---- kernel run + replay ----
    trace = []
    MPSCalculus(A, edgesToLegs(el), 50, 500, trace=trace)
    dense, frames = replay(owners, trace)
    ls2 = [t[2] / np.log(2) for t in trace]            # kernel's log2 size (after cutBond)

    # ---- checks ----
    print(len(A), "tensors,", len(owners), "edges")
    print("steps:", len(dense))
    print("last dense:", dense[-1], " edges in last frame:", len(frames[-1]))
    print("dense >= ls everywhere:", all(dn >= l - 1e-9 for dn, l in zip(dense, ls2)))
    print(f"max dense 2^{max(dense):.1f}   max ls 2^{max(ls2):.1f}")

    # ---- frame data ----
    pos = layout(owners, len(A))
    max_lb = max(int(round(np.log2(v))) for f in frames for v in f.values())
    first_big = next(s for s, f in enumerate(frames) if max(f.values(), default=2) > 2)
    print(f"max bond size: 2^{max_lb}   first bond > 2 at step: {first_big}")

    colors = sample_colorscale('Blues', [0.25 + 0.75 * k / (max_lb - 1) for k in range(max_lb)])
    steps = list(range(430, 462))                       # short test; full: range(0, first_big, 20) + range(first_big, len(trace))

    # ---- animation  ----
    #fig = go.Figure(make_traces(445, trace, frames, pos, max_lb, max(dense), colors))
    updatemenus = [dict(type='buttons', buttons=[dict(label='Play', method='animate', args=[None, dict(
        frame=dict(duration=100, redraw=True), fromcurrent=True)])])]

    sliders = [dict(steps=[dict(method='animate', label=str(s),
                                args=[[str(s)], dict(mode='immediate', frame=dict(duration=0, redraw=True))]) for s in
                           steps])]
    anim_frames = [go.Frame(data=make_traces(s, trace, frames, pos, max_lb, max(dense), colors),
                            name=str(s),
                            layout=go.Layout(title_text=f"step {s}"))
                   for s in steps]

    fig = go.Figure(data=anim_frames[0].data, frames=anim_frames)
    P = np.array(list(pos.values()))
    rng = dict(range=[P.min() - 0.05, P.max() + 0.05], visible=False)
    scene = dict(xaxis=rng, yaxis=rng, zaxis=rng, aspectmode='cube')
    fig.update_layout(updatemenus=updatemenus, sliders=sliders,
                      title=f"step {steps[0]}", legend=dict(x=0, y=1), scene=scene)

    fig.show()
    fig.write_html("contraction_anim.html")

    # ---- video export: renders the same frames (no buttons/slider) as PNGs, then one MP4 ----
    import os
    os.makedirs("frames", exist_ok=True)
    figs, paths = [], []
    for k, fr in enumerate(anim_frames):
        f = go.Figure(data=fr.data)
        f.update_layout(title=f"step {fr.name}", legend=dict(x=0, y=1), scene=scene)
        figs.append(f)
        paths.append(f"frames/{k:04d}.png")
    # PNGs via a headless Chromium (playwright) instead of kaleido, which kept failing here
    from playwright.sync_api import sync_playwright
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 720})
        for f, path in zip(figs, paths):
            f.update_layout(width=1280, height=720)
            page.set_content(f.to_html(include_plotlyjs=True))
            page.wait_for_function(
                "() => { const gd = document.querySelector('.js-plotly-plot'); return gd && gd._fullLayout; }")
            page.wait_for_timeout(300)                  # let WebGL finish drawing
            page.screenshot(path=path)
            print(f"\r{path}", end="")
        browser.close()
    print()
    # MP4 instead of GIF: PowerPoint (Mac) misplaces the cropped delta frames of a Pillow GIF,
    # and a video can be set to "Start: On Click"
    import imageio.v2 as imageio
    fps = 4                                             # frames per second (lower = slower)
    with imageio.get_writer("contraction.mp4", fps=fps, codec="libx264",
                            pixelformat="yuv420p", quality=8) as video:
        for path in paths:
            video.append_data(imageio.imread(path)[:, :, :3])
        for _ in range(2 * fps):                        # hold the last frame 2 s
            video.append_data(imageio.imread(paths[-1])[:, :, :3])
    print("saved contraction.mp4,", len(paths), "frames")
