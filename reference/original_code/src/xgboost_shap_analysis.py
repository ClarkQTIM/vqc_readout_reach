import os
import shap
import xgboost as xgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

def run_shap_analysis(metric, df, X, display_features, save_dir):
    y = df[metric]
    model = xgb.XGBRegressor(random_state=42)
    model.fit(X, y)

    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    print(f"✅ {metric} | MSE: {mse:.6f}")

    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)

    os.makedirs(save_dir, exist_ok=True)

    # Summary plot
    plt.figure()
    shap.summary_plot(shap_values, X, feature_names=display_features, show=False)
    plt.savefig(os.path.join(save_dir, f"shap_summary_{metric}.png"), bbox_inches="tight")
    plt.close()

    # Bar plot
    plt.figure()
    shap.plots.bar(shap_values, show=False)
    ax = plt.gca()
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_xlabel(r"$\mu(|\mathrm{SHAP}|)$", fontsize=14)
    plt.savefig(os.path.join(save_dir, f"shap_bar_{metric}.png"), bbox_inches="tight")
    plt.close()

    # Force plot
    shap.plots.force(shap_values[0], matplotlib=True, show=False)
    plt.savefig(os.path.join(save_dir, f"shap_force_example_{metric}.png"), bbox_inches="tight")
    plt.close()

    # Save SHAP values
    shap_df = pd.DataFrame(shap_values.values, columns=X.columns)
    shap_df[metric] = y.values
    shap_df.to_csv(os.path.join(save_dir, f"shap_table_{metric}.csv"), index=False)

if __name__ == "__main__":
    # === Paths ===
    BASE_DIR = "/scratch90/chris/cnn_vqc_expressivity"
    RESULT_DIR = os.path.join(BASE_DIR, "individual_qubits/4_qubit")
    SHAP_DIR = os.path.join(BASE_DIR, "individual_qubits/4_qubit")

    # df_expr = pd.read_csv(os.path.join(RESULT_DIR, "expressibility_results.csv"))
    df_ent = pd.read_csv(os.path.join(RESULT_DIR, "entanglement_results.csv"))
    # df_grad = pd.read_csv(os.path.join(RESULT_DIR, "gradient_norm_results.csv"))
    # df_perf = pd.read_csv(os.path.join(BASE_DIR, "full_4q_sweep_iris/performance_results.csv"))

    print('CSVs loaded')
    df = df_ent.copy()

    # # === Merge everything ===
    # on = ["num_layers", "single_qubit_op", "entanglement_op", "topology"]
    # df = df_expr.merge(df_ent, on=on)
    # df = df.merge(df_grad, on=on)
    # df = df.merge(df_perf, on=on)

    print('Df created')
    # === Rename columns ===
    df = df.rename(columns={
        # "kl_divergence": "KL Divergence",
        "avg_entropy": "Entanglement",
        # "grad_norm": "Gradient Norm",
        # "roc_auc_mean": "ROC AUC"
    })

    print('Df renamed')

    # === SHAP 1–4: Circuit design → property
    # targets_stage1 = ["KL Divergence", "Entanglement", "Gradient Norm", "ROC AUC"]
    targets_stage1 = ["Entanglement"]
    # targets_stage1 = ["ROC AUC"]
    X1 = pd.get_dummies(df[["num_layers", "single_qubit_op", "entanglement_op", "topology"]], dtype=float)

    display_features1 = X1.columns.tolist()
    print(f'Display features {display_features1}')
    
    for metric in targets_stage1:

        # Check and drop NaN or inf from target column
        bad_mask = ~np.isfinite(df[metric])
        num_bad = bad_mask.sum()

        if num_bad > 0:
            print(f"\n⚠️ WARNING: Dropping {num_bad} rows from target '{metric}' due to NaN or inf values.")
            df_filtered = df[~bad_mask].copy()
            X_filtered = X1.loc[~bad_mask].copy()
            display_features_filtered = X_filtered.columns.tolist()
        else:
            df_filtered = df.copy()
            X_filtered = X1.copy()
            display_features_filtered = display_features1

        # Now safe to assign target
        y = df_filtered[metric].values

        run_shap_analysis(
            metric,
            df_filtered,
            X_filtered,
            display_features_filtered,
            os.path.join(SHAP_DIR, f"circuit_to_{metric.replace(' ', '_')}")
        )
    # # === SHAP 5: Properties → ROC AUC
    # targets_stage2 = ["KL Divergence", "Entanglement", "Gradient Norm"]
    # X2 = df[targets_stage2]
    # display_features2 = X2.columns.tolist()

    # run_shap_analysis("ROC AUC", df, X2, display_features2, os.path.join(SHAP_DIR, "properties_to_ROC_AUC"))
