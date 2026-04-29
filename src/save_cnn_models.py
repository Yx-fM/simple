"""Train and save all CNN fold models."""

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from model_cnn import CNN

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load data
    train_df = pd.read_csv('digit-recognizer/train.csv')
    train_labels = train_df['label'].values.astype('int64')
    train_images = train_df.drop('label', axis=1).values.astype('float32') / 255.0
    print(f"Train: {len(train_images)}")

    n_folds = 5
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_accs = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_images)):
        print(f"\n--- CNN Fold {fold+1}/{n_folds} ---")

        train_tensor = torch.from_numpy(train_images[train_idx]).view(-1, 1, 28, 28)
        train_labels_tensor = torch.from_numpy(train_labels[train_idx])
        train_loader = DataLoader(TensorDataset(train_tensor, train_labels_tensor), batch_size=128, shuffle=True)

        val_tensor = torch.from_numpy(train_images[val_idx]).view(-1, 1, 28, 28)
        val_labels_tensor = torch.from_numpy(train_labels[val_idx])
        val_loader = DataLoader(TensorDataset(val_tensor, val_labels_tensor), batch_size=256, shuffle=False)

        model = CNN().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)

        for epoch in range(15):
            model.train()
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()

        model.eval()
        correct = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                _, predicted = torch.max(model(images), 1)
                correct += (predicted == labels).sum().item()

        val_acc = correct / len(val_idx)
        fold_accs.append(val_acc)
        print(f"Fold {fold+1} Val Acc: {val_acc:.4f}")

        torch.save(model.state_dict(), f'model_cnn_fold{fold}.pth')
        print(f"Saved model_cnn_fold{fold}.pth")

    print(f"\nCNN Mean Val Acc: {np.mean(fold_accs):.4f} (+/- {np.std(fold_accs):.4f})")


if __name__ == '__main__':
    main()