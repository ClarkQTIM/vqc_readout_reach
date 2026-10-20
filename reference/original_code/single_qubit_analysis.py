# single_qubit_analysis.py

import pandas as pd
import matplotlib.pyplot as plt


def load_single_qubit_data(csv_path):
    """Load expressibility results and filter for 1-qubit circuits."""
    df = pd.read_csv(csv_path)
    return df[df["n_qubits"] == 1]


def plot_expressibility_vs_layers(df, output_path):
    """Plot KL divergence vs number of layers for each single-qubit gate."""
    plt.figure(figsize=(8, 6))

    for gate in df["single_qubit_op"].unique():
        df_gate = df[df["single_qubit_op"] == gate].sort_values("num_layers")
        plt.plot(
            df_gate["num_layers"],
            df_gate["kl_divergence"],
            marker="o",
            label=gate.upper()
        )

    plt.title("Single-Qubit Gate Expressibility vs Depth")
    plt.xlabel("Number of Layers")
    plt.ylabel("KL Divergence (↓ = More Expressive)")
    plt.legend(title="Single-Qubit Gate")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


if __name__ == "__main__":
    # === Config ===
    csv_path = "/scratch90/chris/cnn_vqc_expressivity/individual_qubits/single_qubit/expressibility_results.csv"
    output_path = "/scratch90/chris/cnn_vqc_expressivity/individual_qubits/single_qubit/single_qubit_expressibility.png"

    df = load_single_qubit_data(csv_path)
    plot_expressibility_vs_layers(df, output_path)
    print(f"✅ Plot saved to: {output_path}")
