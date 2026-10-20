import os
import re
import pandas as pd

# === CONFIG ===
BASE_DIR = "/scratch90/chris/cnn_vqc_expressivity/full_4q_sweep_retina"
OUTPUT_CSV = os.path.join(BASE_DIR, "performance_results.csv")

results = []

# === Iterate over all experiment folders ===
for exp_name in os.listdir(BASE_DIR):
    exp_path = os.path.join(BASE_DIR, exp_name)
    if not os.path.isdir(exp_path) or not exp_name.startswith("q4_"):
        continue

    # Parse experiment config from folder name
    try:
        parts = exp_name.split("_")
        num_layers = int(parts[1][1:])   # l1 → 1
        sq_op = parts[2]
        ent_op = parts[3]
        topo_raw = parts[4].lower()
        topo = "all_to_all" if topo_raw == "all" else topo_raw
    except Exception as e:
        print(f"❌ Skipping {exp_name} due to name parsing error: {e}")
        continue

    aucs = []
    for run_dir in ["run1", "run2", "run3"]:
        metrics_path = os.path.join(exp_path, run_dir, "metrics.txt")
        if not os.path.exists(metrics_path):
            print(f"⚠️ Missing: {metrics_path}")
            continue

        with open(metrics_path, "r") as f:
            for line in f:
                if line.lower().startswith("roc auc"):
                    try:
                        auc = float(re.findall(r"[-+]?[0-9]*\.?[0-9]+", line)[0])
                        aucs.append(auc)
                    except Exception as e:
                        print(f"❌ Couldn't parse AUC from {metrics_path}: {e}")
                    break

    if len(aucs) == 0:
        print(f"⚠️ No AUC found for {exp_name}, skipping.")
        continue

    avg_auc = sum(aucs) / len(aucs)
    results.append({
        "num_layers": num_layers,
        "single_qubit_op": sq_op,
        "entanglement_op": ent_op,
        "topology": topo,
        "roc_auc_mean": avg_auc
    })

# === Save to CSV ===
df = pd.DataFrame(results)
df.sort_values(by="roc_auc_mean", ascending=False, inplace=True)
df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Saved performance summary to: {OUTPUT_CSV}")
