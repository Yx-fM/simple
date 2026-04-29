"""Train CNNv2 with 5-fold CV."""

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).parent))

from model_cnn import CNNv2


def load_data():
    train_df = pd.read_csv('digit-recognizer/train.csv')
    test_df = pd.read_csv('digit-recognizer/test.csv')
    train_labels = train_df['label'].values
    train_images = train_df.drop('label', axis=1).values / 255.0
    test_images = test_df.values / 255.0
    return train_images.astype('float32'), train_labels.astype('int64'), test_images.astype('float32')


def train_fold(model, train_loader, device, epochs=15, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

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


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    train_images, train_labels, test_images = load_data()
    print(f"Train: {len(train_images)}, Test: {len(test_images)}")

    n_folds = 5
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    all_models = []
    fold_accs = []

    print(f"\n{'='*50}")
    print("Training CNNv2 with 5-fold CV")
    print(f"{'='*50}")

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_images)):
        print(f"\n--- CNNv2 Fold {fold+1}/{n_folds} ---")

        train_tensor = torch.tensor(train_images[train_idx]).view(-1, 1, 28, 28)
        train_labels_tensor = torch.tensor(train_labels[train_idx], dtype=torch.long)
        train_loader = DataLoader(TensorDataset(train_tensor, train_labels_tensor), batch_size=128, shuffle=True)

        val_tensor = torch.tensor(train_images[val_idx]).view(-1, 1, 28, 28)
        val_labels_tensor = torch.tensor(train_labels[val_idx], dtype=torch.long)
        val_loader = DataLoader(TensorDataset(val_tensor, val_labels_tensor), batch_size=256, shuffle=False)

        model = CNNv2().to(device)
        train_fold(model, train_loader, device, epochs=15)

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

        all_models.append(model)

    print(f"\nCNNv2 Mean Val Acc: {np.mean(fold_accs):.4f} (+/- {np.std(fold_accs):.4f})")

    # Save models
    for i, m in enumerate(all_models):
        torch.save(m.state_dict(), f'model_cnnv2_fold{i}.pth')
    print("Saved all CNNv2 fold models")


if __name__ == '__main__':
    main()