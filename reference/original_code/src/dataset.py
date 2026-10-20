"""
dataset.py

Custom PyTorch Dataset for image classification tasks in CNN-VQC pipelines.

This dataset:
- Loads and filters metadata from a CSV file
- Supports binary relabeling of class labels (e.g., class 0 vs. max class)
- Optionally balances classes via undersampling
- Applies appropriate PyTorch transforms per split ('train', 'val', 'test')
- Supports both standard image files and `.npy` arrays (grayscale or RGB)

Main Class:
- ImageDataset: A torch.utils.data.Dataset subclass for flexible and reproducible image loading.

Example Usage:
    train_set = ImageDataset(csv_file, img_dir, img_col="Image", label_col="label",
                             split_col="split", split_type="train", img_size=(256, 256),
                             balance_classes=True, n_samples=500)
"""

import os
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

'''
4/30: To do: Add some augmentations to the training!
'''


class ImageDataset(Dataset):
    def __init__(self, csv_file, img_dir, img_col, label_col, split_col, split_type, 
                 img_size=(256, 256), balance_classes=False, n_samples=None, random_state=42):
        self.data = pd.read_csv(csv_file)

        # ✅ Filter by split
        self.data = self.data[self.data[split_col] == split_type].copy()

        # ✅ Binary relabeling: class 0 vs max class
        max_label = self.data[label_col].max()
        self.data = self.data[self.data[label_col].isin([0, max_label])].copy()
        self.data.loc[self.data[label_col] == max_label, label_col] = 1

        # ✅ Class balancing (optional)
        if balance_classes:
            self.data = self._balance_classes(self.data, label_col, random_state)

        # ✅ Subsampling (optional)
        if n_samples and len(self.data) > n_samples:
            self.data = self.data.sample(n=n_samples, random_state=random_state).reset_index(drop=True)

        print(f"✅ Loaded {len(self.data)} samples for {split_type}. "
              f"Classes: {self.data[label_col].value_counts().to_dict()}")

        self.img_dir = img_dir
        self.img_col = img_col
        self.label_col = label_col
        self.split_col = split_col
        self.img_size = img_size

        # ✅ Define transforms
        self.train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
        ])

        self.val_test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
            transforms.Resize(img_size),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.data.iloc[idx][self.img_col])

        if img_path.endswith('.npy'):
            image = np.load(img_path)
            image = np.uint8(255 * image)
            if image.ndim == 2:
                image = np.stack([image] * 3, axis=-1)
            image = Image.fromarray(image, mode="RGB")
        else:
            image = Image.open(img_path).convert("RGB")

        label = self.data.iloc[idx][self.label_col]
        dataset_split = self.data.iloc[idx][self.split_col]

        if dataset_split == 'train':
            image = self.train_transform(image)
        else:
            image = self.val_test_transform(image)

        return image, torch.tensor(label, dtype=torch.long)

    def _balance_classes(self, data, label_col, random_state):
        """
        Balances the dataset by undersampling the majority class.
        """
        class_counts = data[label_col].value_counts()
        min_class_count = class_counts.min()
        balanced_data = pd.concat([
            data[data[label_col] == label].sample(min_class_count, random_state=random_state)
            for label in class_counts.index
        ]).sample(frac=1, random_state=random_state).reset_index(drop=True)

        print(f"✅ Balanced Classes: {balanced_data[label_col].value_counts().to_dict()}")
        return balanced_data