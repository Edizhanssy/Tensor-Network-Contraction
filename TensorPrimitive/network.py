# circuit -> tensor network  (Edizhan's circuitToEdges + edgesToLegs)
import numpy as np


def circuitToEdges(circuit, x):
    next_edge = 0
    A = []
    n_qubits = len(x)
    wire = [0] * n_qubits
    edge_lists = []
    for q in range(n_qubits):                         # one ket per qubit
        A.append(np.array([1, 0], dtype=complex))
        edge_lists.append([next_edge])
        wire[q] = next_edge
        next_edge += 1
    for tensor, qubits in circuit:                    # gates: outs first, then ins
        ins = [wire[q] for q in qubits]
        outs = []
        for q in qubits:
            outs.append(next_edge)
            next_edge += 1
        A.append(tensor)
        edge_lists.append(outs + ins)
        for q, o in zip(qubits, outs):
            wire[q] = o
    for q in range(n_qubits):                         # one bra per qubit
        bra = np.zeros(2, dtype=complex)
        bra[x[q]] = 1
        A.append(bra)
        edge_lists.append([wire[q]])
    return A, edge_lists


def edgesToLegs(edge_lists):
    n_edges = max(max(edges) for edges in edge_lists) + 1
    edge_to_tensors = [[] for _ in range(n_edges)]
    for t in range(len(edge_lists)):
        for e in edge_lists[t]:
            edge_to_tensors[e].append(t)
    assert all(len(ts) == 2 for ts in edge_to_tensors)
    legs = [[] for _ in range(len(edge_lists))]
    for t in range(len(edge_lists)):
        for e in edge_lists[t]:
            a, b = edge_to_tensors[e]
            legs[t].append(a if a != t else b)
    return legs


def stateVectorAmplitude(circuit, x):
    """Reference (validation only): full state vector, feasible up to ~25 qubits."""
    n = len(x)
    psi = np.zeros([2] * n, dtype=complex)
    psi[(0,) * n] = 1
    for gate, qubits in circuit:
        k = len(qubits)
        psi = np.tensordot(gate, psi, axes=(list(range(k, 2 * k)), qubits))
        psi = np.moveaxis(psi, list(range(k)), qubits)
    return psi[tuple(x)]
