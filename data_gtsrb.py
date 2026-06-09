#!/usr/bin/env python3
"""
GTSRB Dataset loader for road sign classification.

This module extends the data.py to support the German Traffic Sign Recognition Benchmark (GTSRB).
Useful for thesis experiments comparing ANN/SNN on real-world road sign detection.

Dataset info:
- URL: https://www.kaggle.com/datasets/meowmeowmeow/gtsrb-german-traffic-sign
- Classes: 43 traffic sign categories
- Training samples: ~39,209
- Test samples: ~12,630
- Image size: 32x32 (already resized)

Usage:
  1. Download GTSRB dataset from Kaggle
  2. Extract to: data/GTSRB/
  3. Call: load_gtsrb(batch_size=128)
"""

import torch
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as transforms
from pathlib import Path
import numpy as np
from typing import Tuple, Optional
import pickle
from PIL import Image

from utils import CIFAR10_MEAN, CIFAR10_STD, INPUT_SIZE


class GTSRBDataset(Dataset):
    """
    GTSRB (German Traffic Sign Recognition Benchmark) dataset.
    
    Structure:
        GTSRB/
        ├── Final_Training/
        │   ├── Images/
        │   └── GT-final_train.csv
        └── Final_Test/
            ├── Images/
            └─┐ GT-final_test.csv
    """
    
    def __init__(self, data_dir: str, split: str = "train", transform=None):
        """
        Args:
            data_dir: Path to GTSRB root directory
            split: "train" or "test"
            transform: Image transforms
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.images = []
        self.labels = []
        
        if split == "train":
            self.root_dir = self.data_dir / "Final_Training" / "Images"
            csv_file = self.data_dir / "Final_Training" / "GT-final_train.csv"
        else:
            self.root_dir = self.data_dir / "Final_Test" / "Images"
            csv_file = self.data_dir / "Final_Test" / "GT-final_test.csv"
        
        # Load image paths and labels from CSV
        with open(csv_file, 'r') as f:
            lines = f.readlines()[1:]  # Skip header
        
        for line in lines:
            parts = line.strip().split(';')
            if len(parts) >= 2:
                img_path = self.root_dir / parts[0]
                label = int(parts[1])
                if img_path.exists():
                    self.images.append(str(img_path))
                    self.labels.append(label)
        
        print(f"Loaded {len(self.images)} {split} images")
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def load_gtsrb(batch_size: int = 128, num_workers: int = 4,
              val_split: float = 0.1, data_dir: str = "./data/GTSRB") -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load GTSRB dataset with train/val/test split.
    
    Args:
        batch_size: Batch size for DataLoader
        num_workers: Number of workers for data loading
        val_split: Fraction of training set for validation
        data_dir: Path to GTSRB directory
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    data_dir = Path(data_dir)
    
    if not data_dir.exists():
        raise FileNotFoundError(
            f"GTSRB dataset not found at {data_dir}\n"
            f"Please download from: https://www.kaggle.com/datasets/meowmeowmeow/gtsrb-german-traffic-sign\n"
            f"And extract to: {data_dir}"
        )
    
    # Training transforms with augmentation
    train_transform = transforms.Compose([
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.GaussianBlur(kernel_size=3),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),  # Use CIFAR-10 stats as baseline
    ])
    
    # Test transforms
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    
    # Load full training set
    train_dataset_full = GTSRBDataset(
        str(data_dir),
        split="train",
        transform=train_transform
    )
    
    # Split into train and val
    train_size = int(len(train_dataset_full) * (1 - val_split))
    val_size = len(train_dataset_full) - train_size
    train_dataset, val_dataset = random_split(
        train_dataset_full,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Apply test transform to val set
    val_dataset.dataset.transform = test_transform
    
    # Load test set
    test_dataset = GTSRBDataset(
        str(data_dir),
        split="test",
        transform=test_transform
    )
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"GTSRB dataset loaded:")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    print(f"  Classes: 43 (traffic signs)")
    
    return train_loader, val_loader, test_loader


def get_gtsrb_class_names() -> list:
    """
    Get GTSRB class names (43 traffic sign categories).
    
    Returns:
        List of class names
    """
    return [
        'Speed limit (20km/h)', 'Speed limit (30km/h)', 'Speed limit (50km/h)',
        'Speed limit (60km/h)', 'Speed limit (70km/h)', 'Speed limit (80km/h)',
        'End of speed limit (80km/h)', 'Speed limit (100km/h)', 'Speed limit (120km/h)',
        'No passing', 'No passing for trucks', 'Right-of-way at intersection',
        'Priority road', 'Yield', 'Stop', 'No entry', 'Prohibited area',
        'No entry for trucks', 'Direction mandatory (straight)', 'Direction mandatory (left)',
        'Direction mandatory (right)', 'Direction mandatory (straight/left)',
        'Direction mandatory (straight/right)', 'Direction mandatory (left/right)',
        'Direction mandatory (straight/left)', 'Go right', 'Keep left',
        'Roundabout mandatory', 'End of no passing', 'End of no passing for trucks',
        'Pedestrian crossing', 'Children crossing', 'Bicycle crossing',
        'Beware of ice/slippery road', 'Wild animals crossing', 'End of all restrictions',
        'Turn right ahead', 'Turn left ahead', 'Ahead only', 'Go straight or right',
        'Go straight or left', 'Keep right', 'Keep left', 'Traffic signal',
    ]
