import os
import csv
from itertools import product

import torch
import numpy as np
import scipy.stats

from quantum_models import build_vqc_circuit, init_weights

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_bloch_sphere(bloch_vectors, title=None, save_path=None):
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection='3d')

    # Draw Bloch sphere
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_wireframe(x, y, z, color="lightgray", alpha=0.2)

    # Plot Bloch vectors
    xs, ys, zs = bloch_vectors[:, 0], bloch_vectors[:, 1], bloch_vectors[:, 2]
    ax.scatter(xs, ys, zs, c="blue", s=5, alpha=0.5)

    ax.set_title(title or "Bloch Vectors")
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close()

def compute_expressibility_kl(
    vqc, n_qubits, num_layers, single_op, ent_op, topology,
    n_samples=200, n_bins=75, bloch_output_dir=None, plot_bloch=False
):
    states = []
    bloch_vectors = []

    for _ in range(n_samples):
        weights = init_weights(
            n_qubits=n_qubits,
            num_layers=num_layers,
            single_qubit_op=single_op,
            entanglement_op=ent_op if ent_op else "cnot",
            topology=topology,
            encoder=False
        )
        x = torch.zeros(n_qubits)

        with torch.no_grad():
            state = vqc(x, weights["sq_weights"], weights["ent_weights"]).detach().cpu().numpy()
            states.append(state)

            if n_qubits == 1 and bloch_output_dir:
                alpha, beta = state[0], state[1]
                bx = 2 * np.real(np.conj(alpha) * beta)
                by = 2 * np.imag(np.conj(alpha) * beta)
                bz = np.abs(alpha) ** 2 - np.abs(beta) ** 2
                bloch_vectors.append([bx, by, bz])

    # Save and plot Bloch vectors if 1-qubit
    if n_qubits == 1 and bloch_output_dir:
        bloch_vectors = np.array(bloch_vectors)
        os.makedirs(bloch_output_dir, exist_ok=True)
        fname_base = f"bloch_{single_op}_layers{num_layers}"
        np.save(os.path.join(bloch_output_dir, f"{fname_base}.npy"), bloch_vectors)

        if plot_bloch:
            plot_bloch_sphere(
                bloch_vectors,
                title=f"{single_op} | Layers={num_layers}",
                save_path=os.path.join(bloch_output_dir, f"{fname_base}.png")
            )

    # KL divergence
    fidelities = []
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            f = np.abs(np.vdot(states[i], states[j])) ** 2
            fidelities.append(f)

    hist_pqc, bin_edges = np.histogram(fidelities, bins=n_bins, range=(0, 1), density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    N = 2 ** n_qubits
    haar_pdf = (N - 1) * (1 - bin_centers) ** (N - 2)
    haar_pdf /= np.trapz(haar_pdf, bin_centers)

    epsilon = 1e-10
    kl_div = scipy.stats.entropy(hist_pqc + epsilon, haar_pdf + epsilon)

    return kl_div

def run_expressivity_sweep(
    qubit_counts,
    layer_counts,
    single_qubit_ops,
    entanglement_ops,
    topologies,
    output_path,
    n_samples=200,
    bloch_output_dir=None,
    plot_bloch=False
):

    header = [
        "n_qubits", "num_layers", "single_qubit_op",
        "entanglement_op", "topology", "kl_divergence"
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

            print(f"Running config: {n_qubits} qubits | {num_layers} layers | {single_op} | {ent_op or 'no ent'} | {topology}")

            try:
                vqc = build_vqc_circuit(
                    n_qubits=n_qubits,
                    num_layers=num_layers,
                    encoder=False,
                    single_qubit_op=single_op,
                    entanglement_op=ent_op if ent_op else "cnot",
                    topology=topology,
                    return_type="state"
                )

                kl = compute_expressibility_kl(
                    vqc=vqc,
                    n_qubits=n_qubits,
                    num_layers=num_layers,
                    single_op=single_op,
                    ent_op=ent_op,
                    topology=topology,
                    n_samples=n_samples,
                    bloch_output_dir=bloch_output_dir,
                    plot_bloch=plot_bloch
                )


                writer.writerow([n_qubits, num_layers, single_op, ent_op, topology, kl])
                print(f"✅ {n_qubits}q | L={num_layers} | {single_op} + {ent_op or 'no ent'} ({topology}): KL={kl:.5f}")

            except Exception as e:
                print(f"❌ Skipped config ({n_qubits}, {num_layers}, {single_op}, {ent_op}, {topology}): {e}")


# === Example Usage ===
if __name__ == "__main__":

    sweep_set_1q = {
        "qubit_counts": [4],
        "layer_counts": [1, 2, 3, 4, 5, 6],
        "single_qubit_ops": ["u3", "rx", "ry", "rz"],
        "entanglement_ops": [None, "cnot", "cz", "crx", "cry", "ising_xx", "ising_yy", "ising_zz"],
        "topologies": ["ring", "all_to_all"],
        "output_path": "/scratch90/chris/cnn_vqc_expressivity/individual_qubits/4_qubit/expressibility_results.csv",
        "n_samples": 1000,
        "bloch_output_dir": "/scratch90/chris/cnn_vqc_expressivity/individual_qubits/4_qubit/bloch_plots",
        "plot_bloch": False
    }

    run_expressivity_sweep(**sweep_set_1q)
