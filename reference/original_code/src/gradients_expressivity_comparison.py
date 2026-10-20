import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_grad_vs_kl_by_qubits_and_layers(expressivity_csv, gradient_csv, save_dir="/path.to/save/"):
    # === Load and preprocess
    df_expr = pd.read_csv(expressivity_csv)
    df_grad = pd.read_csv(gradient_csv)

    df_expr["entanglement_op"] = df_expr["entanglement_op"].fillna("none")
    df_grad["entanglement_op"] = df_grad["entanglement_op"].fillna("none")

    merge_cols = ["n_qubits", "num_layers", "single_qubit_op", "entanglement_op", "topology"]
    df = pd.merge(df_expr, df_grad, on=merge_cols, suffixes=("_kl", "_grad"))

    # === Plot settings
    os.makedirs(save_dir, exist_ok=True)
    marker_styles = {
        "none": "o", "cnot": "X", "cz": "s", "crx": "P",
        "cry": "D", "ising_xx": "^", "ising_yy": "v", "ising_zz": "*"
    }
    edge_colors = {
        "ring": "gray",
        "all_to_all": "black",
        "none": "lightgray"
    }

    # === Loop over all (qubit, layer) combinations
    for nq in sorted(df["n_qubits"].unique()):
        for l in sorted(df["num_layers"].unique()):
            df_sub = df[(df["n_qubits"] == nq) & (df["num_layers"] == l)]
            if df_sub.empty:
                continue

            plt.figure(figsize=(10, 6))
            for _, row in df_sub.iterrows():
                marker = marker_styles.get(row["entanglement_op"], "o")
                edge_color = edge_colors.get(row["topology"], "gray")
                color = sns.color_palette("tab10").as_hex()[hash(row["single_qubit_op"]) % 10]

                plt.scatter(
                    row["kl_divergence"],
                    row["grad_norm"],
                    label=f"{row['single_qubit_op']}",
                    marker=marker,
                    color=color,
                    edgecolor=edge_color,
                    s=100,
                    alpha=0.8,
                    linewidth=1.2
                )

            plt.title(f"KL vs Grad Norm — {nq} Qubits, {l} Layers")
            plt.xlabel("KL Divergence (↓ = More Expressive)")
            plt.ylabel("Gradient Norm")
            plt.grid(True)
            plt.tight_layout()

            filename = f"kl_vs_grad_q{nq}_l{l}.png"
            plt.savefig(os.path.join(save_dir, filename), dpi=300)
            plt.close()

    print(f"✅ Saved plots to: {save_dir}")

if __name__ == "__main__":

    plot_grad_vs_kl_by_qubits_and_layers(
        expressivity_csv="/scratch90/chris/cnn_vqc_expressivity/expressibility_results.csv",
        gradient_csv="/scratch90/chris/cnn_vqc_expressivity/gradient_norm_sweep.csv",
        save_dir="/scratch90/chris/cnn_vqc_expressivity/expressibility_gradient_analysis"
    )
