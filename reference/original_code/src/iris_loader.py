import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

def get_iris_dataloaders(batch_size=16, val_split=0.2, test_split=0.2, random_state=42):
    iris = load_iris()
    X = iris['data']
    y = iris['target']

    # Filter to only class 0 and 1 (binary classification)
    mask = (y == 0) | (y == 1)
    X = X[mask]
    y = y[mask]

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Stratified split: train/val/test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_split, stratify=y, random_state=random_state
    )
    val_ratio_adjusted = val_split / (1 - test_split)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio_adjusted, stratify=y_temp, random_state=random_state
    )

    # Convert to torch tensors
    def to_tensor_dataset(X, y):
        return TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))

    train_loader = DataLoader(to_tensor_dataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(to_tensor_dataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(to_tensor_dataset(X_test, y_test), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
