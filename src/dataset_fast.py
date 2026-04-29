"""Fast data loading with torchvision transforms for augmentation."""

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional
import torchvision.transforms as transforms


class DigitDataset(Dataset):
    """PyTorch Dataset for digit recognition."""

    def __init__(self, csv_path: str, has_labels: bool = True,
                 transform=None):
        """
        Args:
            csv_path: Path to CSV file
            has_labels: Whether the CSV contains labels
            transform: Optional torchvision transform
        """
        self.data = pd.read_csv(csv_path)
        self.has_labels = has_labels
        self.transform = transform

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
        # Reshape to (28, 28)
        image = self.images[idx].reshape(28, 28)
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # (1, 28, 28)

        # Apply transforms
        if self.transform is not None:
            image = self.transform(image)

        if self.has_labels:
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            return image, label
        else:
            return image, None


def get_train_transforms():
    """Data augmentation transforms for training."""
    return transforms.Compose([
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    ])


def get_val_transforms():
    """No augmentation for validation/test."""
    return None


def get_dataloaders(
    train_path: str,
    test_path: str,
    batch_size: int = 128,
    train_split: float = 0.85,
    use_augmentation: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test dataloaders."""
    train_transform = get_train_transforms() if use_augmentation else None

    # Full training dataset
    full_train = DigitDataset(train_path, has_labels=True, transform=train_transform)

    # Split
    train_size = int(train_split * len(full_train))
    val_size = len(full_train) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_train, [train_size, val_size]
    )

    # Validation uses no augmentation
    val_dataset.dataset = DigitDataset(train_path, has_labels=True, transform=None)
    val_indices = val_dataset.indices
    val_dataset = torch.utils.data.Subset(
        DigitDataset(train_path, has_labels=True, transform=None),
        val_indices
    )

    # Test dataset
    test_dataset = DigitDataset(test_path, has_labels=False, transform=None)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader