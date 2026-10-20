import os
import csv
from itertools import product
from itertools import combinations

import torch
import numpy as np
import pennylane as qml

from quantum_models import build_vqc_circuit, init_weights


def compute_avg_entanglement(vqc, encoder_proj, sq_weights, ent_weights, n_qubits=4):
    """
    Computes average von Neumann entropy over all balanced bipartitions.
    Works for any n_qubits ≥ 2.
    """
    # Always input |00...0⟩ (amplitude encoding)
    x = torch.zeros(2 ** n_qubits)
    x[0] = 1.0

    # Get statevector output from the circuit
    state = vqc(x, encoder_proj, sq_weights, ent_weights).detach().cpu().numpy()

    # Convert to density matrix
    rho = np.outer(state, state.conj())

    # Find all balanced bipartitions
    mid = n_qubits // 2
    qubit_indices = list(range(n_qubits))
    bipartitions = list(combinations(qubit_indices, mid))

    # To avoid duplicate complements (e.g., (A,B) vs (C,D) and (C,D) vs (A,B)), only use half
    unique_bipartitions = bipartitions[:len(bipartitions)//2]

    # Compute von Neumann entropy for each bipartition
    entropies = []
    for indices_A in unique_bipartitions:
        S = qml.math.vn_entropy(rho, indices=list(indices_A))
        entropies.append(S)

    return float(np.mean(entropies))
    
def run_entanglement_sweep(
    qubit_counts,
    layer_counts,
    single_qubit_ops,
    entanglement_ops,
    topologies,
    output_path,
    n_samples=200
):
    header = [
        "n_qubits", "num_layers", "single_qubit_op",
        "entanglement_op", "topology", "avg_entropy"
    ]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for n_qubits, num_layers, single_op, ent_op, topology in product(
            qubit_counts, layer_counts, single_qubit_ops, entanglement_ops, topologies
        ):
            if n_qubits < 2 and ent_op is not None:
                print(f"⚠️ Skipping: {n_qubits}q circuit cannot use entanglement ({ent_op})")
                continue
            if ent_op is None:
                topology = "none"

            print(f"Running config: {n_qubits}q | {num_layers}L | {single_op} + {ent_op or 'no ent'} ({topology})")

            # Define the QNode builder outside the sampling loop
            def build_ent_circuit():
                dev = qml.device("default.qubit", wires=n_qubits)

                @qml.qnode(dev, interface="torch")
                def ent_circuit(x, encoder_proj, sq_weights, ent_weights):
                    x_proj = x / torch.norm(x, dim=-1, keepdim=True).clamp(min=1e-8)
                    qml.AmplitudeEmbedding(x_proj, wires=range(n_qubits), normalize=True)

                    for layer in range(num_layers):
                        for q in range(n_qubits):
                            w = sq_weights[layer, q]
                            if single_op == "rx":
                                qml.RX(w, wires=q)
                            elif single_op == "ry":
                                qml.RY(w, wires=q)
                            elif single_op == "rz":
                                qml.RZ(w, wires=q)
                            elif single_op == "u3":
                                qml.U3(*w, wires=q)

                        pairs = (
                            [(i, (i + 1) % n_qubits) for i in range(n_qubits)]
                            if topology == "ring"
                            else [(i, j) for i in range(n_qubits) for j in range(i + 1, n_qubits)]
                        )
                        for idx, (i, j) in enumerate(pairs):
                            if ent_weights is not None:
                                w = ent_weights[layer, idx]

                            if ent_op == "cnot":
                                qml.CNOT(wires=[i, j])
                            elif ent_op == "crx":
                                qml.CRX(w, wires=[i, j])
                            elif ent_op == "cry":
                                qml.CRY(w, wires=[i, j])
                            elif ent_op == "cz":
                                qml.CZ(wires=[i, j])
                            elif ent_op == "ising_xx":
                                qml.IsingXX(w, wires=[i, j])
                            elif ent_op == "ising_yy":
                                qml.IsingYY(w, wires=[i, j])
                            elif ent_op == "ising_zz":
                                qml.IsingZZ(w, wires=[i, j])

                    return qml.state()  # Return full statevector

                return ent_circuit

            entropies = []
            try:
                vqc = build_ent_circuit()
                for _ in range(n_samples):
                    weights = init_weights(
                        n_qubits=n_qubits,
                        num_layers=num_layers,
                        single_qubit_op=single_op,
                        entanglement_op=ent_op if ent_op else "cnot",
                        topology=topology,
                        encoder=False
                    )

                    entropy = compute_avg_entanglement(
                        vqc=vqc,
                        encoder_proj=weights["encoder_proj"],
                        sq_weights=weights["sq_weights"],
                        ent_weights=weights["ent_weights"],
                        n_qubits=n_qubits
                    )
                    entropies.append(entropy)

                avg_entropy = float(np.mean(entropies))
                writer.writerow([n_qubits, num_layers, single_op, ent_op, topology, avg_entropy])
                print(f"✅ {n_qubits}q | L={num_layers} | {single_op} + {ent_op or 'no ent'} ({topology}): Avg Entropy={avg_entropy:.4f}")

            except Exception as e:
                print(f"❌ Skipped config ({n_qubits}, {num_layers}, {single_op}, {ent_op}, {topology}): {e}")

# === Example Usage ===
if __name__ == "__main__":
    sweep_cfg = {
        "qubit_counts": [4],
        "layer_counts": [1, 2, 3, 4, 5, 6],
        "single_qubit_ops": ["rx", "ry", "rz", "u3"],
        "entanglement_ops": [None, "cnot", "cz", "crx", "cry", "ising_xx", "ising_yy", "ising_zz"],
        "topologies": ["ring", "all_to_all"],
        "output_path": "/scratch90/chris/cnn_vqc_expressivity/individual_qubits/4_qubit/entanglement_results.csv",
        "n_samples":1000
    }

    run_entanglement_sweep(**sweep_cfg)