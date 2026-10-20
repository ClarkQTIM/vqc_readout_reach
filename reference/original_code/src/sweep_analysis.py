import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === Config ===
EXPRESS_CSV = "/scratch90/chris/cnn_vqc_expressivity/sweep_results_4_qubits/expressibility_results.csv"
ENTANGLE_CSV = "/scratch90/chris/cnn_vqc_expressivity/sweep_results_4_qubits/entanglement_results.csv"
GRAD_CSV = "/scratch90/chris/cnn_vqc_expressivity/sweep_results_4_qubits/gradient_norm_results.csv"
PERF_CSV = "/scratch90/chris/cnn_vqc_expressivity/full_4q_sweep_retina/performance_results.csv"
SAVE_DIR = "/scratch90/chris/cnn_vqc_expressivity/sweep_results_4_qubits/retina/analysis_plots"
os.makedirs(SAVE_DIR, exist_ok=True)

# === Load CSVs ===
df_expr = pd.read_csv(EXPRESS_CSV)
df_ent = pd.read_csv(ENTANGLE_CSV)
df_grad = pd.read_csv(GRAD_CSV)
df_perf = pd.read_csv(PERF_CSV)

# === Merge DataFrames ===
common_cols = ["num_layers", "single_qubit_op", "entanglement_op", "topology"]
df = df_expr.merge(df_ent, on=common_cols)
df = df.merge(df_grad, on=common_cols)
df = df.merge(df_perf, on=common_cols)
df = df.rename(columns={
    "kl_divergence": "KL Divergence",
    "avg_entropy": "Entanglement (Entropy)",
    "grad_norm": "Gradient Norm",
    "roc_auc_mean": "ROC AUC"
})

# === Rename for clarity ===
df = df.rename(columns={
    "kl_divergence": "KL Divergence",
    "avg_entropy": "Entanglement (Entropy)",
    "grad_norm": "Gradient Norm",
})

# === Plotting Helper ===
def save_plot(fig, name):
    fig.savefig(os.path.join(SAVE_DIR, f"{name}.png"), bbox_inches="tight")
    plt.close(fig)

# === Plot 1: KL Divergence by single_qubit_op ===
fig, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(data=df, x="single_qubit_op", y="KL Divergence", ax=ax)
ax.set_title("KL Divergence by Single-Qubit Op")
save_plot(fig, "kl_by_sq_op")

# === Plot 2: Entanglement by entanglement_op ===
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(data=df[df["entanglement_op"] != "none"], x="entanglement_op", y="Entanglement (Entropy)", ax=ax)
ax.set_title("Entanglement by Entanglement Op")
ax.tick_params(axis='x', rotation=45)
save_plot(fig, "entropy_by_ent_op")

# === Plot 3: KL Divergence vs Entanglement ===
fig, ax = plt.subplots(figsize=(7, 6))
sns.scatterplot(data=df, x="Entanglement (Entropy)", y="KL Divergence", hue="single_qubit_op", style="topology", ax=ax)
ax.set_title("KL Divergence vs Entanglement")
save_plot(fig, "kl_vs_entropy")

# === Plot 4: Gradient Norm vs KL Divergence ===
fig, ax = plt.subplots(figsize=(7, 6))
sns.scatterplot(data=df, x="KL Divergence", y="Gradient Norm", hue="num_layers", style="topology", ax=ax)
ax.set_title("Gradient Norm vs KL Divergence")
save_plot(fig, "grad_vs_kl")

# === Plot 5: ROC AUC vs Entanglement ===
fig, ax = plt.subplots(figsize=(7, 6))
sns.scatterplot(data=df, x="Entanglement (Entropy)", y="ROC AUC", hue="single_qubit_op", style="topology", ax=ax)
ax.set_title("ROC AUC vs Entanglement")
save_plot(fig, "auc_vs_entropy")

# === Plot 6: ROC AUC vs Expressivity ===
fig, ax = plt.subplots(figsize=(7, 6))
sns.scatterplot(data=df, x="KL Divergence", y="ROC AUC", hue="single_qubit_op", style="topology", ax=ax)
ax.set_title("ROC AUC vs KL Divergence")
save_plot(fig, "auc_vs_kl")

# === Plot 7: ROC AUC vs Gradient Norm ===
fig, ax = plt.subplots(figsize=(7, 6))
sns.scatterplot(data=df, x="Gradient Norm", y="ROC AUC", hue="single_qubit_op", style="topology", ax=ax)
ax.set_title("ROC AUC vs Gradient Norm")
save_plot(fig, "auc_vs_grad")

# === Plot 7: Correlation Matrix ===
corr = df[["ROC AUC", "KL Divergence", "Entanglement (Entropy)", "Gradient Norm"]].corr()

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
ax.set_title("Correlation Matrix")
save_plot(fig, "correlation_matrix")



print("✅ Analysis complete. Plots saved to:", SAVE_DIR)
