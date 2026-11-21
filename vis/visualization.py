from TensorPrimitive.circuits import randomCircuit, H, T, Xh, Yh, CZ
import plotly.graph_objects as go
import quimb.tensor as qtn
import trimesh
from trimesh.visual.material import PBRMaterial
import cotengra as ctg

def drawable():
    L, d = 10, 8
    circuit = randomCircuit(L, d, seed=0)

    n = L * L
    cyls = []

    circ = qtn.Circuit(n, gate_contract=False)

    for gate, qubits in circuit:
        if gate is H:
            circ.apply_gate('H', *qubits)
        elif gate is T:
            circ.apply_gate('T', *qubits)
        elif gate is Xh:
            circ.apply_gate('X_1_2', *qubits)
        elif gate is Yh:
            circ.apply_gate('Y_1_2', *qubits)
        elif gate is CZ:
            circ.apply_gate('CZ', *qubits)

    #print(circ.num_gates, len(circuit))

    tn = circ.amplitude_tn('0'*n, simplify_sequence = '')

    #print(tn.num_tensors)

    color = ['H', 'T', 'X_1/2', 'Y_1/2', 'CZ']
    groups = {}

   #amp = circ.amplitude('0' * n, optimize='greedy')
    #print(amp, abs(amp))

    xs = []
    ys = []
    zs = []
    edges, nodes, opts = tn.draw(color=color, dim=3, get='data')
    for e in edges.values():
        p0, p1 = e['coos']
        xs.extend([p0[0], p1[0], None])
        ys.extend([p0[1], p1[1], None])
        zs.extend([p0[2], p1[2], None])

    for e in edges.values():
        p0, p1 = e['coos']
        cyl = trimesh.creation.cylinder(radius=0.006, segment=[p0, p1], sections=6)
        cyls.append(cyl)

    edge_mesh = trimesh.util.concatenate(cyls)
    edge_mesh.visual = trimesh.visual.TextureVisuals(material=PBRMaterial(baseColorFactor=[90, 90, 90, 255], metallicFactor=0.0, roughnessFactor=0.7))
    print(len(cyls), len(edge_mesh.faces))
    scene = trimesh.Scene()
    scene.add_geometry(edge_mesh, geom_name='edges')

    for nd in nodes.values():
        cat = 'input'
        for g in color:
            if "'" + g + "'" in nd['tags']:
                cat = g
                break
        gx, gy, gz = groups.setdefault(cat, ([], [], []))
        gx.append(nd['coo'][0])
        gy.append(nd['coo'][1])
        gz.append(nd['coo'][2])

    for k, v in groups.items():
        print(k, len(v[0]))

    print(len(xs), len(ys), len(zs))
    print(len(edges), len(nodes))
    print(list(edges.values())[0]['coos'])

    for cat, (gx, gy, gz) in groups.items():
        spheres = []
        for x, y, z in zip(gx, gy, gz):
            spheres.append(trimesh.creation.icosphere(subdivisions=1, radius=0.015).apply_translation((x, y, z)))
        node_mesh = trimesh.util.concatenate(spheres)
        if cat in opts['colors']:
            c = opts['colors'][cat]
            rgba = [int(c[0]*255), int(c[1]*255), int(c[2]*255), 255]
        else:
            rgba = [115, 120, 128, 255]
        if cat == 'CZ':
            rgba = [148, 103, 189, 255]
        node_mesh.visual = trimesh.visual.TextureVisuals(
            material=PBRMaterial(baseColorFactor=rgba, metallicFactor=0.0, roughnessFactor=0.7))
        scene.add_geometry(node_mesh, geom_name=cat.replace('/', '_'))

    #scene.add_geometry(node_mesh, geom_name=cat.replace('/', '_'))

    #dim = 3
    print(list(scene.geometry.keys()))
    scene.export(f'network_L{L}_d{d}_actual.glb')

    #print(nodes[0], len(nodes), len(edges))
    #fig = tn.draw(dim = dim, color = color, node_scale = 3, edge_scale = 2, edge_alpha = 1, backend='plotly', legend = True, figsize = (8, 8))
    #fig.write_html('network_L6_d20.html')
    def to_rgb(c):
        return f"rgb({int(c[0]*255)}, {int(c[1]*255)}, {int(c[2]*255)})"

    labels = {'input': 'input |0⟩', 'H': 'H', 'T': 'T', 'X_1/2': 'X^1/2', 'Y_1/2': 'Y^1/2', 'CZ': 'CZ'}

    opt = ctg.HyperOptimizer(methods=['greedy'], max_repeats=32, parallel=False)
    tree = tn.contraction_tree(optimize=opt)
    print('contraction width: ', tree.contraction_width())  # log2 of the largest intermediate
    #exit()

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode='lines',
                               line=dict(color='gray', width=2), hoverinfo='none', showlegend=False))
    for cat, (gx, gy, gz) in groups.items():
        col = to_rgb(opts['colors'][cat]) if cat in opts['colors'] else 'rgb(115, 120, 128)'
        fig.add_trace(go.Scatter3d(x=gx, y=gy, z=gz, mode='markers', name=labels[cat],
                                   marker=dict(size=4, color=col)))
    fig.update_layout(scene=dict(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False),
                      paper_bgcolor='white')
    fig.write_html('network_L8_d20_edited.html')
    fig.show()


if __name__ == '__main__':
    drawable()

