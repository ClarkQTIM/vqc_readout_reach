import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from medmnist import INFO, RetinaMNIST
import numpy as np

class BinaryRetinaSubset(Dataset):
    """
    A wrapper for a filtered subset of RetinaMNIST that:
    - Keeps only class 0 and severe classes (3 and 4)
    - Maps 3 and 4 → 1 for binary classification
    """
    def __init__(self, base_dataset, indices, relabeled_labels):
        self.dataset = base_dataset
        self.indices = indices
        self.labels = torch.tensor(relabeled_labels, dtype=torch.long)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img, _ = self.dataset[self.indices[idx]]  # original label ignored
        return img, self.labels[idx]

def get_retinamnist_dataloaders(img_size=(256, 256), batch_size=4, balance_classes=False, n_samples=None, num_workers=4):
    """
    Returns train/val/test dataloaders for RetinaMNIST, filtered to class 0 and severe classes (3, 4),
    remapped as binary labels (0 vs 1). Applies data augmentation to the training set.

    Args:
        img_size (tuple): Desired image size, e.g., (256, 256)
        batch_size (int): Batch size
        balance_classes (bool): Undersample to balance binary classes
        n_samples (int or None): If specified, limit to this many samples
        num_workers (int): DataLoader workers

    Returns:
        train_loader, val_loader, test_loader
    """

    data_flag = 'retinamnist'
    info = INFO[data_flag]
    DataClass = getattr(__import__('medmnist'), info['python_class'])

    # === Transforms ===
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])

    train_dataset = DataClass(split='train', transform=train_transform, download=True)
    val_dataset   = DataClass(split='val', transform=val_test_transform, download=True)
    test_dataset  = DataClass(split='test', transform=val_test_transform, download=True)

    def filter_binary_subset(dataset, split_name):
        labels_np = dataset.labels.squeeze()
        all_indices = np.arange(len(labels_np))

        # ✅ Keep class 0 and (3 or 4), drop 1 and 2
        keep_mask = (labels_np == 0) | (labels_np == 3) | (labels_np == 4)
        filtered_indices = all_indices[keep_mask]
        filtered_labels = labels_np[filtered_indices]

        # ✅ Map 3 and 4 → 1
        relabeled = np.copy(filtered_labels)
        relabeled[(relabeled == 3) | (relabeled == 4)] = 1

        print(f"✅ Loaded {len(relabeled)} samples for {split_name}. "
              f"Classes: {dict(zip(*np.unique(relabeled, return_counts=True)))}")

        # ✅ Balance if requested
        if balance_classes:
            idx_0 = filtered_indices[relabeled == 0]
            idx_1 = filtered_indices[relabeled == 1]
            min_len = min(len(idx_0), len(idx_1))
            balanced_indices = np.concatenate([idx_0[:min_len], idx_1[:min_len]])
            relabeled_balanced = np.concatenate([
                relabeled[relabeled == 0][:min_len],
                relabeled[relabeled == 1][:min_len]
            ])
            print(f"✅ Balanced Classes: {{0: {min_len}, 1: {min_len}}}")
        else:
            balanced_indices = filtered_indices
            relabeled_balanced = relabeled

        # ✅ Subsample if requested
        if n_samples and len(balanced_indices) > n_samples:
            np.random.shuffle(balanced_indices)
            balanced_indices = balanced_indices[:n_samples]
            relabeled_balanced = relabeled[np.isin(filtered_indices, balanced_indices)]
            print(f"✅ Subsampled to {len(balanced_indices)} samples. "
                  f"New Classes: {dict(zip(*np.unique(relabeled_balanced, return_counts=True)))}")

        return BinaryRetinaSubset(dataset, balanced_indices, relabeled_balanced)

    # Final loaders
    train_loader = DataLoader(filter_binary_subset(train_dataset, "train"), batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader   = DataLoader(filter_binary_subset(val_dataset, "val"), batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(filter_binary_subset(test_dataset, "test"), batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader

