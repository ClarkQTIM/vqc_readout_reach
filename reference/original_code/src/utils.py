"""
utils.py

Utility functions for CNN-VQC pipeline.

Includes:
- PCA projection matrix fitting from feature extractor outputs
- Calculation of number of qubits and layers in VQC to match classical parameter budget
"""

import torch
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA


def fit_pca_from_feature_extractor(train_loader, feature_extractor, n_components=64, device='cuda'):
    """
    Fit PCA on features extracted from training data using a frozen feature extractor.

    Args:
        train_loader (DataLoader): Training data loader.
        feature_extractor (nn.Module): Pretrained and frozen feature extractor.
        n_components (int): Number of PCA dimensions to retain.
        device (str): 'cuda' or 'cpu'.

    Returns:
        torch.Tensor: PCA projection matrix of shape (n_components, feature_dim).
    """
    feature_extractor.eval()
    all_features = []

    with torch.no_grad():
        for images, _, _, _ in tqdm(train_loader, desc="Collecting features for PCA"):
            images = images.to(device)
            feats = feature_extractor(images)
            all_features.append(feats.cpu())

    all_features = torch.cat(all_features, dim=0)
    pca = PCA(n_components=n_components)
    pca.fit(all_features.numpy())
    pca_proj_matrix = torch.tensor(pca.components_, dtype=torch.float32).to(device)

    return pca_proj_matrix


def get_vqc_layer_count_matching_classical(num_outputs, vqc_type="basic", include_bias=True):
    """
    Estimate number of qubits and layers required for a VQC to match parameter count
    of a classical linear head.

    Args:
        num_outputs (int): Number of outputs of the classical head.
        vqc_type (str): Type of quantum circuit ('basic', 'basicu3', 'expressive').
        include_bias (bool): Whether to count bias parameter in classical head.

    Returns:
        (int, int, int): (n_qubits, n_layers, param_target)
    """
    assert vqc_type in ["basic", "basicu3", "expressive"], \
        "Only supports 'basic', 'basicu3', or 'expressive' VQC types"

    param_target = num_outputs + (1 if include_bias else 0)
    n_qubits = int(np.ceil(np.log2(num_outputs)))

    if vqc_type == "basic":
        num_layers = int(np.ceil(param_target / n_qubits))
    elif vqc_type == "basicu3":
        num_layers = int(np.ceil(param_target / (3 * n_qubits)))
    elif vqc_type == "expressive":
        num_layers = int(np.ceil(param_target / (6 * n_qubits)))

    return n_qubits, num_layers, param_target