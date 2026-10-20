import os
import csv
import torch
import numpy as np
import pandas as pd
from itertools import product

# === Local project modules ===
from expressivity_computation import run_expressivity_sweep, compute_expressibility_kl
from entanglement_computation import run_entanglement_sweep, compute_avg_entanglement
from gradient_computation import run_gradient_sweep, compute_gradient_norm

# === Shared sweep configuration ===
SWEEP_CFG = {
    "qubit_counts": [4],
    "layer_counts": [1, 2, 4],
    "single_qubit_ops": ["rx", "ry", "rz", "u3"],
    "entanglement_ops": [None, "cnot", "cz", "crx", "cry", "ising_xx", "ising_yy", "ising_zz"],
    "topologies": ["ring", "all_to_all"]
    # "qubit_counts": [4],
    # "layer_counts": [4],
    # "single_qubit_ops": ["rx"],
    # "entanglement_ops": [None, "cnot"],
    # "topologies": ["ring"]
}


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)

    # === Run Expressivity Sweep ===
    print("🔁 Running expressivity sweep...")
    run_expressivity_sweep(
        **SWEEP_CFG,
        output_path="sweep_results_4_qubits/expressibility_results.csv"
    )

    # === Run Entanglement Sweep ===
    print("🔁 Running entanglement sweep...")
    run_entanglement_sweep(
        **SWEEP_CFG,
        output_path="sweep_results_4_qubits/entanglement_results.csv"
    )

    # === Run Gradient Norm Sweep ===
    print("🔁 Running gradient norm sweep...")
    run_gradient_sweep(
        **SWEEP_CFG,
        output_path="sweep_results_4_qubits/gradient_norm_results.csv"
    )

    print("✅ All sweeps complete!")
