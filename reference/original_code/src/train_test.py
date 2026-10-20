"""
train_test.py

Training and testing engine for hybrid CNN-VQC and CNN-classical pipelines.

Features:
- Supports classical and quantum heads (VQC)
- Supports PCA-projected or random inputs to heads
- Early stopping, weight saving, and live loss plotting
- Outputs confusion matrix, ROC curve, metrics, and detailed predictions

Main Interfaces:
- train_model(): Runs training loop with optional quantum head
- test_model(): Runs testing + evaluation pipeline with saved model
"""

import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
import copy
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix, roc_curve
)

from feature_extractor import get_feature_extractor
from quantum_models import build_vqc_circuit, init_weights
import torch.nn as nn

def evaluate_vqc_batch(vqc, encoded_features, weights_dict):
    """
    Evaluates a batch of already-encoded features using the modular VQC.
    
    Args:
        vqc: the QNode
        encoded_features: shape (B, 2^n_qubits), output of encoder
        weights_dict: contains 'sq_weights', 'ent_weights'

    Returns:
        Tensor of shape (B, C) with class probabilities
    """
    sq_weights = weights_dict["sq_weights"]
    ent_weights = weights_dict["ent_weights"]

    device = sq_weights.device
    outputs = []

    for x in encoded_features:
        x = x.to(device)
        out = vqc(x, sq_weights, ent_weights)
        outputs.append(out)

    return torch.stack(outputs)

def compute_out(vqc, features, weights_dict):
    """
    Evaluates the VQC circuit on all features and returns class probabilities.
    """
    return evaluate_vqc_batch(vqc, features, weights_dict)

def compute_cost(vqc, weights_dict, features, labels, loss_type="xent"):
    """
    Computes the loss between VQC outputs and true labels using cross-entropy.
    """
    out = compute_out(vqc, features, weights_dict)
    eps = 1e-8
    target_probs = out[torch.arange(len(labels)), labels.long()]
    return -torch.mean(torch.log(target_probs + eps))

def train_model(train_loader,
                val_loader,
                feature_extractor,
                encoder,
                vqc_weights,
                vqc_callable,
                save_dir,
                n_epochs=100,
                encoder_lr=1e-4,
                vqc_lr=1e-1,
                patience=10):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if feature_extractor is not None:
        feature_extractor.to(device).eval()

    if encoder is not None:
        encoder.to(device).train()
        encoder_params = list(encoder.parameters())
    else:
        encoder_params = []

    for k, v in vqc_weights.items():
        if isinstance(v, torch.nn.Parameter):
            vqc_weights[k] = v.to(device)
        elif isinstance(v, nn.Module):
            v.to(device).train()

    vqc_params = [v for v in vqc_weights.values() if isinstance(v, torch.nn.Parameter) and v.requires_grad]

    param_groups = []
    if encoder_params:
        param_groups.append({"params": encoder_params, "lr": encoder_lr})
    if vqc_params:
        param_groups.append({"params": vqc_params, "lr": vqc_lr})

    optimizer = torch.optim.Adam(param_groups)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

    best_val_loss = float('inf')
    best_state = None
    epochs_no_improve = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(n_epochs):
        if encoder is not None:
            encoder.train()

        total_train_loss = 0.0

        for images, labels in tqdm(train_loader, desc=f"[Epoch {epoch+1}]"):
            images, labels = images.to(device), labels.to(device)

            if feature_extractor is not None:
                with torch.no_grad():
                    features = feature_extractor(images)
            else:
                features = images  # For tabular data

            encoded = encoder(features) if encoder is not None else features

            optimizer.zero_grad()
            loss = compute_cost(vqc_callable, vqc_weights, encoded, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder_params + vqc_params, max_norm=1.0)
            optimizer.step()
            total_train_loss += loss.item()

        scheduler.step()
        avg_train_loss = total_train_loss / len(train_loader)
        history["train_loss"].append(avg_train_loss)

        # --- Validation ---
        if encoder is not None:
            encoder.eval()
        else:
            encoder = None
        total_val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                if feature_extractor is not None:
                    features = feature_extractor(images)
                else:
                    features = images  # For tabular data

                encoded = encoder(features) if encoder is not None else features
                loss = compute_cost(vqc_callable, vqc_weights, encoded, labels)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        history["val_loss"].append(avg_val_loss)
        print(f"Epoch {epoch+1}: Train={avg_train_loss:.4f}, Val={avg_val_loss:.4f}")

        # --- Live Loss Plot ---
        plt.figure()
        plt.plot(history["train_loss"], label="Train Loss")
        plt.plot(history["val_loss"], label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training vs Validation Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "loss_curves.png"))
        plt.close()

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = {
                "encoder": encoder.state_dict() if encoder is not None else None,
                "vqc_weights": {
                    k: (
                        v.detach().cpu()
                        if isinstance(v, torch.Tensor)
                        else v.state_dict() if v is not None else None
                    )
                    for k, v in vqc_weights.items()
                }
            }

            epochs_no_improve = 0
            print("✅ New best model saved.")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("⏹️ Early stopping triggered.")
                break

    # Save
    torch.save(best_state, os.path.join(save_dir, "best_model.pth"))
    with open(os.path.join(save_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print("📦 Training complete and saved.")
    return history

def test_model(test_loader, save_dir, n_qubits, num_layers, single_qubit_op, entanglement_op, topology, feature_extractor_type="resnet"):
    """
    Evaluate the trained encoder + VQC head using a frozen feature extractor or raw input.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # === Load model checkpoint ===
    ckpt_path = os.path.join(save_dir, "best_model.pth")
    checkpoint = torch.load(ckpt_path, map_location=device)
    encoder_weights = checkpoint.get("encoder", None)
    vqc_weights = checkpoint["vqc_weights"]

    # === Rebuild feature extractor if applicable ===
    if feature_extractor_type is not None:
        feature_extractor, feature_dim = get_feature_extractor(extractor_type=feature_extractor_type)
        feature_extractor.to(device).eval()
        for param in feature_extractor.parameters():
            param.requires_grad = False
    else:
        feature_extractor = None
        

    # === Rebuild encoder if applicable ===
    if encoder_weights is not None:
        encoder = torch.nn.Linear(feature_dim, 2 ** n_qubits)
        encoder.load_state_dict(encoder_weights)
        encoder.to(device)
        encoder.eval()
    else:
        encoder = None

    # === Rebuild VQC ===
    vqc = build_vqc_circuit(
        n_qubits=n_qubits,
        num_layers=num_layers,
        single_qubit_op=single_qubit_op,
        entanglement_op=entanglement_op,
        topology=topology,
        return_type="probs"
    )

    for k, v in vqc_weights.items():
        if v is not None and isinstance(v, torch.Tensor):
            v.requires_grad = False
            vqc_weights[k] = v.to(device)

    # === Evaluation loop ===
    y_true, y_pred, y_probs = [], [], []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)
            features = feature_extractor(images) if feature_extractor is not None else images
            encoded = encoder(features) if encoder is not None else features
            batch_probs = evaluate_vqc_batch(vqc, encoded, vqc_weights)

            predictions = torch.argmax(batch_probs, dim=1).cpu().numpy()
            class1_probs = batch_probs[:, 1].detach().cpu().numpy()

            y_pred.extend(predictions)
            y_probs.extend(class1_probs)
            y_true.extend(labels.cpu().numpy())

    # === Save predictions ===
    predictions_df = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
        "y_probs": y_probs
    })
    predictions_csv_file = os.path.join(save_dir, "detailed_predictions.csv")
    predictions_df.to_csv(predictions_csv_file, index=False)
    print(f"📂 Predictions saved: {predictions_csv_file}")

    # === Compute metrics ===
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_probs)

    # === Classification report ===
    class_report = classification_report(y_true, y_pred, target_names=['Class 0', 'Class 1'])
    print("\n📊 Classification Report:\n", class_report)

    # === Confusion matrix ===
    conf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"))
    plt.close()

    # === ROC curve ===
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "roc_curve.png"))
    plt.close()

    # === Save metrics ===
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc
    }

    metrics_file = os.path.join(save_dir, "metrics.txt")
    with open(metrics_file, "w") as f:
        for k, v in metrics.items():
            f.write(f"{k.replace('_', ' ').title()}: {v:.4f}\n")
        f.write("\nClassification Report:\n" + class_report)
    print(f"📂 Metrics saved: {metrics_file}")

    print(f"\n✅ Final Test Metrics | Accuracy: {accuracy:.2f}, ROC AUC: {roc_auc:.2f}")
    return accuracy, metrics

