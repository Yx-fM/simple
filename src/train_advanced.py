"""Advanced training with Label Smoothing and stronger augmentation."""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.model_selection import KFold
import torchvision.transforms as T
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from model_cnn import CNN


class AdvancedDataset(Dataset):
    """Dataset with strong augmentation."""
    def __init__(self, images, labels=None):
        self.images = images
        self.labels = labels
        self.transform = T.Compose([
            T.ToPILImage(),
            T.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
            T.RandomPerspective(distortion_scale=0.2, p=0.5),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].reshape(28, 28) * 255  # Convert back to 0-255 for PIL
        image = self.transform(image.astype(np.uint8))

        if self.labels is not None:
            return image, torch.tensor(self.labels[idx], dtype=torch.long)
        return image, None


class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        n_classes = pred.size(1)
        log_preds = F.log_softmax(pred, dim=1)

        with torch.no_grad():
            smooth_labels = torch.full_like(log_preds, self.smoothing / (n_classes - 1))
            smooth_labels.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)

        return (-smooth_labels * log_preds).sum(dim=1).mean()


def load_data():
    train_df = pd.read_csv('digit-recognizer/train.csv')
    test_df = pd.read_csv('digit-recognizer/test.csv')
    train_labels = train_df['label'].values.astype('int64')
    train_images = train_df.drop('label', axis=1).values.astype('float32') / 255.0
    test_images = test_df.values.astype('float32') / 255.0
    return train_images, train_labels, test_images


def train_fold(train_images, train_labels, val_idx, device, epochs=35):
    train_idx = np.array([i for i in range(len(train_images)) if i not in val_idx])

    train_dataset = AdvancedDataset(train_images[train_idx], train_labels[train_idx])
    val_dataset = TensorDataset(
        torch.tensor(train_images[val_idx], dtype=torch.float32).view(-1, 1, 28, 28),
        torch.tensor(train_labels[val_idx], dtype=torch.long)
    )

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

    model = CNN().to(device)
    criterion = LabelSmoothingLoss(smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    best_acc = 0.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        # Validate
        model.eval()
        correct = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                _, predicted = torch.max(model(images), 1)
                correct += (predicted == labels).sum().item()

        val_acc = correct / len(val_idx)
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{epochs} Val Acc: {val_acc:.4f}")

    return best_state, best_acc


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    train_images, train_labels, test_images = load_data()
    print(f"Train: {len(train_images)}, Test: {len(test_images)}")

    n_folds = 5
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    all_states = []
    fold_accs = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_images)):
        print(f"\n--- Fold {fold+1}/{n_folds} ---")
        best_state, best_acc = train_fold(train_images, train_labels, val_idx, device, epochs=35)
        fold_accs.append(best_acc)
        all_states.append(best_state)
        print(f"Fold {fold+1} Best Val Acc: {best_acc:.4f}")
        torch.save(best_state, f'model_adv_fold{fold}.pth')

    print(f"\nMean Val Acc: {np.mean(fold_accs):.4f} (+/- {np.std(fold_accs):.4f})")
    np.save('test_images.npy', test_images)
    print("Saved test_images.npy")


if __name__ == '__main__':
    main()