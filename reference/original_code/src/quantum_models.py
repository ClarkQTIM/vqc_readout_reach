"""
quantum_models.py

Modular construction and visualization utilities for expressive VQC architectures.

Features:
- Fully parameterized VQC builder supporting:
  - Single-qubit gates: RY, RX, RZ, U3
  - Entangling gates: CNOT, CRX, CRY, CZ, IsingXX, YY, ZZ
  - Topologies: ring or all-to-all
  - Optional learnable encoder projection
- Weight initialization adapted to circuit configuration
- Circuit visualization and diagram export with sample inputs

Designed to support architectural exploration and expressivity evaluation
in quantum machine learning tasks.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pennylane as qml

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pennylane as qml

def build_vqc_circuit(
    n_qubits,
    num_layers,
    single_qubit_op="ry",
    entanglement_op="crx",
    topology="ring",
    interface="torch",
    return_type="probs",
    encoder=False
):

    dev = qml.device("default.qubit", wires=n_qubits)

    def apply_single_qubit_gates(weights, layer):
        for q in range(n_qubits):
            w = weights[layer, q]
            if single_qubit_op == "ry":
                qml.RY(w, wires=q)
            elif single_qubit_op == "rx":
                qml.RX(w, wires=q)
            elif single_qubit_op == "rz":
                qml.RZ(w, wires=q)
            elif single_qubit_op == "u3":
                qml.U3(*w, wires=q)
            else:
                raise ValueError(f"Unsupported single_qubit_op: {single_qubit_op}")

    def apply_entanglement_gates(weights, layer):

        if entanglement_op == "none":
            return  # Skip applying any entanglement gates

        pairs = (
            [(i, (i + 1) % n_qubits) for i in range(n_qubits)]
            if topology == "ring"
            else [(i, j) for i in range(n_qubits) for j in range(i + 1, n_qubits)]
        )
        for idx, (i, j) in enumerate(pairs):
            w = weights[layer, idx] if weights is not None else None
            if entanglement_op == "cnot":
                qml.CNOT(wires=[i, j])
            elif entanglement_op == "crx":
                qml.CRX(w, wires=[i, j])
            elif entanglement_op == "cry":
                qml.CRY(w, wires=[i, j])
            elif entanglement_op == "cz":
                qml.CZ(wires=[i, j])
            elif entanglement_op == "ising_xx":
                qml.IsingXX(w, wires=[i, j])
            elif entanglement_op == "ising_yy":
                qml.IsingYY(w, wires=[i, j])
            elif entanglement_op == "ising_zz":
                qml.IsingZZ(w, wires=[i, j])
            else:
                raise ValueError(f"Unsupported entanglement_op: {entanglement_op}")

    @qml.qnode(dev, interface=interface)
    def circuit(x, sq_weights, ent_weights):
        if x is None:
            raise ValueError("Input x cannot be None during circuit execution")

        if encoder:
            # Use Amplitude Encoding
            x_proj = x / torch.norm(x, dim=-1, keepdim=True).clamp(min=1e-8)
            qml.AmplitudeEmbedding(x_proj, wires=range(n_qubits), normalize=True)
        else:
            # Use Angle Encoding
            if x.shape[-1] != n_qubits:
                raise ValueError(f"Angle encoding requires input of length {n_qubits}, got {x.shape[-1]}")
            for i in range(n_qubits):
                qml.RY(x[i], wires=i)

        for layer in range(num_layers):
            apply_single_qubit_gates(sq_weights, layer)
            apply_entanglement_gates(ent_weights, layer)

        if return_type == "probs":
            return qml.probs(wires=0)
        elif return_type == "state":
            return qml.state()
        elif return_type == "density_matrix":
            return qml.density_matrix(wires=range(n_qubits))
        else:
            raise ValueError(f"Unsupported return_type: {return_type}")
    return circuit


def init_weights(n_qubits, num_layers, single_qubit_op="ry", entanglement_op="crx", topology="ring", encoder=False, input_dim=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    weights_dict = {}

    if single_qubit_op == "u3":
        sq_shape = (num_layers, n_qubits, 3)
    else:
        sq_shape = (num_layers, n_qubits)
    weights_dict["sq_weights"] = torch.nn.Parameter(torch.rand(*sq_shape).to(device) * np.pi)

    if entanglement_op in ["cnot", "cz"]:
        weights_dict["ent_weights"] = None
    else:
        num_pairs = n_qubits if topology == "ring" else n_qubits * (n_qubits - 1) // 2
        weights_dict["ent_weights"] = torch.nn.Parameter(torch.rand(num_layers, num_pairs).to(device) * np.pi)

    if encoder:
        if input_dim is None:
            raise ValueError("input_dim must be provided when encoder=True")
        weights_dict["encoder_proj"] = torch.nn.Linear(input_dim, 2 ** n_qubits, bias=True).to(device)
    else:
        weights_dict["encoder_proj"] = None

    return weights_dict


def draw_vqc_circuit(n_qubits, num_layers, weights, save_path,
                     single_qubit_op="ry", entanglement_op="crx", topology="ring"):

    dev = qml.device("default.qubit", wires=n_qubits)

    encoder_proj = weights["encoder_proj"]
    if encoder_proj is not None:
        encoder_proj = encoder_proj.cpu()
    sq_weights = weights["sq_weights"].cpu().detach()
    ent_weights = weights["ent_weights"]
    ent_weights = ent_weights.cpu().detach() if ent_weights is not None else None

    # === Generate dummy projected input (2^n_qubits)

    with torch.no_grad():
        x = torch.randn(512)
        if encoder_proj is not None:
            x_proj = encoder_proj(x)
            x_proj = x_proj / torch.norm(x_proj, dim=-1, keepdim=True).clamp(min=1e-8)
            x_proj_np = x_proj.numpy()
        else:
            x_proj_np = torch.randn(2 ** n_qubits)
            x_proj_np = x_proj_np / torch.norm(x_proj_np, dim=-1, keepdim=True).clamp(min=1e-8)
            x_proj_np = x_proj_np.numpy()

    def apply_single_qubit_gates(layer):
        for q in range(n_qubits):
            if single_qubit_op == "u3":
                theta, phi, lam = sq_weights[layer, q].tolist()
                qml.U3(theta, phi, lam, wires=q)
            else:
                w = sq_weights[layer, q].item()
                if single_qubit_op == "ry":
                    qml.RY(w, wires=q)
                elif single_qubit_op == "rx":
                    qml.RX(w, wires=q)
                elif single_qubit_op == "rz":
                    qml.RZ(w, wires=q)
                else:
                    raise ValueError(f"Unsupported single_qubit_op: {single_qubit_op}")

    def apply_entanglement_gates(layer):

        pairs = (
            [(i, (i + 1) % n_qubits) for i in range(n_qubits)]
            if topology == "ring"
            else [(i, j) for i in range(n_qubits) for j in range(i + 1, n_qubits)]
        )
        for idx, (i, j) in enumerate(pairs):
            if ent_weights is None:
                continue
            w = ent_weights[layer, idx].item()
            if entanglement_op == "crx":
                qml.CRX(w, wires=[i, j])
            elif entanglement_op == "cry":
                qml.CRY(w, wires=[i, j])
            elif entanglement_op == "cz":
                qml.CZ(wires=[i, j])
            elif entanglement_op == "ising_xx":
                qml.IsingXX(w, wires=[i, j])
            elif entanglement_op == "ising_yy":
                qml.IsingYY(w, wires=[i, j])
            elif entanglement_op == "ising_zz":
                qml.IsingZZ(w, wires=[i, j])
            else:
                print("entanglement_op: {entanglement_op}")


    @qml.qnode(dev, interface="torch")
    def circuit():
        qml.AmplitudeEmbedding(x_proj_np, wires=range(n_qubits), normalize=True)
        for layer in range(num_layers):
            apply_single_qubit_gates(layer)
            apply_entanglement_gates(layer)
        return qml.probs(wires=0)

    # === Draw and save
    fig, ax = qml.draw_mpl(circuit)()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    ax.set_title("VQC Circuit")
    fig.savefig(save_path)
    plt.close()
    print(f"✅ Saved circuit diagram to: {save_path}")
