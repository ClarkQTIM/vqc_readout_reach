import os

BASE_DIR = "/scratch90/chris/cnn_vqc_expressivity/full_4q_sweep"
missing = []

for root, dirs, files in os.walk(BASE_DIR):
    if "metrics.txt" not in files:
        # Only flag leaf directories (which contain your model results)
        if not dirs:
            missing.append(root)

print(f"\nChecked all directories under: {BASE_DIR}")
print(f"Missing metrics.txt in {len(missing)} subdirectories:\n")

for path in missing:
    print(f"❌ {path}")
