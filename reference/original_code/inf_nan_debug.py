import pandas as pd
import numpy as np

def report_nans_infs(df, name):
    print(f"\n=== Analyzing {name} ===")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    # Count NaNs and infs
    nan_counts = df.isna().sum()
    inf_counts = np.isinf(df.select_dtypes(include=[np.number])).sum()

    for col in df.columns:
        if nan_counts[col] > 0 or inf_counts.get(col, 0) > 0:
            print(f"⚠️  {col}: {nan_counts[col]} NaNs, {inf_counts.get(col, 0)} infs")

    # Show rows with NaN or inf
    mask = df.isna().any(axis=1) | np.isinf(df.select_dtypes(include=[np.number])).any(axis=1)
    if mask.any():
        print(f"\n🧪 Rows with issues ({mask.sum()} total):")
        print(df[mask])
    else:
        print("✅ No NaNs or infs found.")


if __name__ == "__main__":
    path_2q = "/scratch90/chris/cnn_vqc_expressivity/individual_qubits/2_qubit/expressibility_results.csv"
    path_4q = "/scratch90/chris/cnn_vqc_expressivity/individual_qubits/4_qubit/expressibility_results.csv"

    df2 = pd.read_csv(path_2q)
    df4 = pd.read_csv(path_4q)

    report_nans_infs(df2, "2-Qubit")
    report_nans_infs(df4, "4-Qubit")

    # Optional: Check schema match
    print("\n=== Schema Match ===")
    schema_match = df2.columns.equals(df4.columns)
    print(f"🧩 Columns match: {schema_match}")

    # Optional: Show value ranges for numeric columns
    print("\n=== Summary Statistics ===")
    print("2-Qubit:\n", df2.describe())
    print("4-Qubit:\n", df4.describe())
