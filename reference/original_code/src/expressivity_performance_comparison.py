import os
import pandas as pd
import matplotlib.pyplot as plt

# --- Config ---
base_dir = "/scratch90/chris/cnn_vqc_expressivity/expressivity_tiers"
expressivity_csv = "/scratch90/chris/cnn_vqc_expressivity/expressibility_results.csv"

# --- Load expressivity data ---
expressivity_df = pd.read_csv(expressivity_csv)

# --- Collect performance data ---
results = []

for tier in ["least_expressive", "most_expressive"]:
    tier_path = os.path.join(base_dir, tier)
    for config_name in os.listdir(tier_path):
        config_path = os.path.join(tier_path, config_name)
        metrics_file = os.path.join(config_path, "metrics.txt")
        if not os.path.exists(metrics_file):
            continue

        try:
            with open(metrics_file, "r") as f:
                lines = f.readlines()
            roc_line = next(line for line in lines if "Roc Auc" in line)
            roc_auc = float(roc_line.strip().split(":")[-1])
        except Exception:
            continue  # skip broken files

        # Parse configuration from folder name: q2_l1_rx_ising_xx_all_to_all
        try:
            parts = config_name.split("_")
            n_qubits = int(parts[0][1:])
            num_layers = int(parts[1][1:])
            single_qubit_op = parts[2]
            entanglement_op = parts[3] if parts[3] != "none" else ""
            topology = parts[4] if len(parts) > 4 else "none"

            results.append({
                "n_qubits": n_qubits,
                "num_layers": num_layers,
                "single_qubit_op": single_qubit_op,
                "entanglement_op": entanglement_op,
                "topology": topology,
                "roc_auc": roc_auc
            })
        except Exception:
            continue

# --- Merge with expressivity ---
metrics_df = pd.DataFrame(results)
merged = pd.merge(metrics_df, expressivity_df, on=["n_qubits", "num_layers", "single_qubit_op", "entanglement_op", "topology"])

# --- Plot ---
plt.figure(figsize=(8, 6))
plt.scatter(merged["kl_divergence"], merged["roc_auc"], c="blue", s=50)
plt.xlabel("KL Divergence (Expressivity)")
plt.ylabel("ROC AUC")
plt.title("Expressivity vs. Performance")
plt.grid(True)
plt.tight_layout()
plt.savefig("/scratch90/chris/cnn_vqc_expressivity/expressibility_performance_analysis/expressivity_vs_performance.png")
plt.show()
