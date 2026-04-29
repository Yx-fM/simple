"""Data loading utilities for Digit Recognizer with augmentation support."""

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, Callable
import numpy as np


class DigitDataset(Dataset):
    """PyTorch Dataset for digit recognition."""
    
    def __init__(self, csv_path: str, has_labels: bool = True,
                 augment: Optional[Callable] = None,
                 is_training: bool = False):
        """
        Args:
            csv_path: Path to CSV file
            has_labels: Whether the CSV contains labels (train=True, test=False)
            augment: Optional augmentation function
            is_training: Whether this is training data (affects reshape)
        """
        self.data = pd.read_csv(csv_path)
        self.has_labels = has_labels
        self.augment = augment
        self.is_training = is_training
        
        if has_labels:
            self.labels = self.data['label'].values
            self.images = self.data.drop('label', axis=1).values
        else:
            self.images = self.data.values
            self.labels = None
        
        # Normalize pixel values to [0, 1]
        self.images = self.images / 255.0
        self.images = self.images.astype('float32')
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        image = self.images[idx]
        
        # Reshape to (1, 28, 28) for CNN
        image = image.reshape(28, 28)
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # (1, 28, 28)
        
        # Apply augmentation for training
        if self.augment is not None and self.is_training:
            image = self.augment(image)
        
        if self.has_labels:
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            return image, label
        else:
            return image, None


class Augmentation:
    """Data augmentation for MNIST digits."""
    
    @staticmethod
    def random_rotation(image: torch.Tensor, max_angle: float = 15.0) -> torch.Tensor:
        """Random rotation within [-max_angle, max_angle] degrees."""
        angle = np.random.uniform(-max_angle, max_angle)
        angle_rad = angle * np.pi / 180.0
        
        # Create rotation matrix
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        theta = torch.tensor([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0]
        ], dtype=torch.float32)
        
        grid = F.affine_grid(theta.unsqueeze(0), image.unsqueeze(0).size(),
                             align_corners=False)
        image = F.grid_sample(image.unsqueeze(0), grid, align_corners=False,
                              mode='bilinear', padding_mode='zeros')
        return image.squeeze(0)
    
    @staticmethod
    def random_shift(image: torch.Tensor, max_shift: int = 3) -> torch.Tensor:
        """Random translation."""
        dx = np.random.randint(-max_shift, max_shift + 1)
        dy = np.random.randint(-max_shift, max_shift + 1)
        
        # Create translation matrix
        theta = torch.tensor([
            [1, 0, dx / 14.0],  # 14 = 28/2 for normalization
            [0, 1, dy / 14.0]
        ], dtype=torch.float32)
        
        grid = F.affine_grid(theta.unsqueeze(0), image.unsqueeze(0).size(),
                             align_corners=False)
        image = F.grid_sample(image.unsqueeze(0), grid, align_corners=False,
                              mode='bilinear', padding_mode='zeros')
        return image.squeeze(0)
    
    @staticmethod
    def random_zoom(image: torch.Tensor, zoom_range: Tuple[float, float] = (0.9, 1.1)) -> torch.Tensor:
        """Random zoom."""
        zoom = np.random.uniform(zoom_range[0], zoom_range[1])
        
        theta = torch.tensor([
            [zoom, 0, 0],
            [0, zoom, 0]
        ], dtype=torch.float32)
        
        grid = F.affine_grid(theta.unsqueeze(0), image.unsqueeze(0).size(),
                             align_corners=False)
        image = F.grid_sample(image.unsqueeze(0), grid, align_corners=False,
                              mode='bilinear', padding_mode='zeros')
        return image.squeeze(0)
    
    @classmethod
    def compose(cls, image: torch.Tensor) -> torch.Tensor:
        """Apply random augmentations."""
        if np.random.random() < 0.5:
            image = cls.random_rotation(image)
        if np.random.random() < 0.5:
            image = cls.random_shift(image)
        if np.random.random() < 0.3:
            image = cls.random_zoom(image)
        return image


def get_dataloaders(
    train_path: str,
    test_path: str,
    batch_size: int = 64,
    train_split: float = 0.85,
    use_augmentation: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.

    Args:
        train_path: Path to train.csv
        test_path: Path to test.csv
        batch_size: Batch size for training
        train_split: Fraction of data for training (rest for validation)
        use_augmentation: Whether to use data augmentation for training

    Returns:
        train_loader, val_loader, test_loader
    """
    augment = Augmentation.compose if use_augmentation else None

    # Load full training dataset
    full_train = DigitDataset(train_path, has_labels=True,
                             augment=augment, is_training=True)

    # Split into train and validation
    train_size = int(train_split * len(full_train))
    val_size = len(full_train) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_train, [train_size, val_size]
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    # Test dataset (no labels, no augmentation)
    test_dataset = DigitDataset(test_path, has_labels=False)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    return train_loader, val_loader, test_loader