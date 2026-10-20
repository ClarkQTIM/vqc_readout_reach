"""
Gradient Norm Sweep for Variational Quantum Circuits (VQCs)

This script evaluates how trainable different VQC architectures are at initialization,
by measuring the L2 norm of the gradient of the output probability with respect to
the model parameters. Each configuration is sampled `n_samples` times to obtain
a reliable average.

This helps detect:
- Vanishing gradients in deep or poorly entangled circuits
- Highly trainable regimes
"""

import os
import csv
from itertools import product

import torch
import numpy as np
from quantum_models import build_vqc_circuit, init_weights


def compute_gradient_norm(vqc, sq_weights, ent_weights, n_qubits):
    """Compute gradient norm of the circuit's output probability with respect to parameters."""
    # Always start from |0⟩
    x = torch.zeros(n_qubits)

    sq_weights.requires_grad_(True)
    if ent_weights is not None:
        ent_weights.requires_grad_(True)

    output = vqc(x, sq_weights, ent_weights)

    # Use probability of measuring |0⟩ as a proxy loss
    loss = output[0]
    loss.backward()

    grad_norm_sq = sq_weights.grad.pow(2).sum()
    if ent_weights is not None:
        grad_norm_sq += ent_weights.grad.pow(2).sum()

    return grad_norm_sq.sqrt().item()


def run_gradient_sweep(
    qubit_counts,
    layer_counts,
    single_qubit_ops,
    entanglement_ops,
    topologies,
    output_path,
    n_samples=200,
):
    """Run a sweep of VQC configurations and log their average gradient norm at initialization."""
    header = [
        "n_qubits", "num_layers", "single_qubit_op",
        "entanglement_op", "topology", "avg_grad_norm"
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

            print(f"🔁 Running: {n_qubits}q | L={num_layers} | {single_op} + {ent_op or 'no ent'} ({topology})")

            try:
                # Build the circuit once
                vqc = build_vqc_circuit(
                    n_qubits=n_qubits,
                    num_layers=num_layers,
                    encoder=False,
                    single_qubit_op=single_op,
                    entanglement_op=ent_op if ent_op else "cnot",
                    topology=topology,
                    return_type="probs"
                )

                grad_norms = []
                for _ in range(n_samples):
                    weights = init_weights(
                        n_qubits=n_qubits,
                        num_layers=num_layers,
                        single_qubit_op=single_op,
                        entanglement_op=ent_op if ent_op else "cnot",
                        topology=topology,
                        encoder=False
                    )
                    grad = compute_gradient_norm(
                        vqc=vqc,
                        sq_weights=weights["sq_weights"],
                        ent_weights=weights["ent_weights"],
                        n_qubits=n_qubits
                    )
                    grad_norms.append(grad)

                avg_grad_norm = float(np.mean(grad_norms))
                writer.writerow([n_qubits, num_layers, single_op, ent_op, topology, avg_grad_norm])
                print(f"✅ Avg Gradient Norm: {avg_grad_norm:.6f}")

            except Exception as e:
                print(f"❌ Skipped config due to error: {e}")


if __name__ == "__main__":
    sweep_set1 = {
        "qubit_counts": [2],
        "layer_counts": [1, 2, 3, 4, 6, 6],
        "single_qubit_ops": ["rx", "ry", "rz", "u3"],
        "entanglement_ops": [None],
        "topologies": [None],
        "output_path": "/scratch90/chris/cnn_vqc_expressivity/individual_qubits/single_qubit/gradient_norm_sweep.csv",
        "n_samples": 1000,
    }

    run_gradient_sweep(**sweep_set1)
