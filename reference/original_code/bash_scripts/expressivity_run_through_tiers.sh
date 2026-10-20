#!/bin/bash

# === CONFIGURATION ===
MAIN_SCRIPT="/scratch90/chris/cnn_vqc_expressivity/src/main.py"
SAVE_BASE="/scratch90/chris/cnn_vqc_expressivity/full_4q_sweep_iris"

# Dataset info
DATASET_NAME="iris"
CSV_PATH="/scratch90/chris/pave_consortium_projects/cervical_cancer_diagnosis/csvs/SEED/SEED_w_model_36_splits.csv"
IMG_COL="MASKED_IMG_ID"
LABEL_COL="CC_ST"
SPLIT_COL="dataset"
IMAGE_DIR="/scratch90/chris/pave_consortium_projects/data/SEED"

# Training
N_EPOCHS=25
N_REPEATS=3

# Sweep configs
NUM_LAYERS=(1 2 4)
SINGLE_Q_OPS=("rx" "ry" "rz" "u3")
ENT_OPS=("none" "cnot" "cz" "crx" "cry" "ising_xx" "ising_yy" "ising_zz")
TOPOLOGIES=("ring" "all_to_all")

# === Trap for Ctrl+C to kill all background jobs ===
trap "echo '🛑 Caught Ctrl+C — killing all jobs...'; pkill -P $$; exit 1" INT


# === GRID SEARCH OVER 4-QUBIT CONFIGS ===
for num_layers in "${NUM_LAYERS[@]}"; do
  for sq_op in "${SINGLE_Q_OPS[@]}"; do
    for ent_op in "${ENT_OPS[@]}"; do
      for topo in "${TOPOLOGIES[@]}"; do

        # Force topology to 'none' if there's no entanglement op
        if [ "$ent_op" == "none" ]; then
          topo="none"
        fi

        EXP_NAME="q4_l${num_layers}_${sq_op}_${ent_op}_${topo}"
        BASE_SAVE_DIR="${SAVE_BASE}/${EXP_NAME}"

        echo "🚀 Running config: $EXP_NAME"

        for i in $(seq 1 $N_REPEATS); do
          RUN_DIR="${BASE_SAVE_DIR}/run${i}"
          mkdir -p "$RUN_DIR"

          echo "  ➤ Launching run $i..."
          python "$MAIN_SCRIPT" \
            --dataset_name "$DATASET_NAME" \
            --csv_path "$CSV_PATH" \
            --img_col "$IMG_COL" \
            --label_col "$LABEL_COL" \
            --split_col "$SPLIT_COL" \
            --image_dir "$IMAGE_DIR" \
            --n_qubits 4 \
            --num_layers "$num_layers" \
            --single_qubit_op "$sq_op" \
            --entanglement_op "$ent_op" \
            --topology "$topo" \
            --n_epochs "$N_EPOCHS" \
            --save_dir "$RUN_DIR" &
        done

        wait  # ⏳ Wait for all 3 runs to finish before next config

      done
    done
  done
done
