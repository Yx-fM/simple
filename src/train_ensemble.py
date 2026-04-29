"""Ensemble training with 5-fold CV for Kaggle Digit Recognizer."""

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

from model_cnn import CNN, CNNv2


def load_data():
    """Load train and test data."""
    train_df = pd.read_csv('digit-recognizer/train.csv')
    test_df = pd.read_csv('digit-recognizer/test.csv')

    train_labels = train_df['label'].values
    train_images = train_df.drop('label', axis=1).values / 255.0

    test_images = test_df.values / 255.0

    return train_images.astype('float32'), train_labels.astype('int64'), test_images.astype('float32')


def train_fold(model, train_loader, device, epochs=15, lr=0.001):
    """Train model for one fold."""
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
        print(f"  Epoch {epoch+1}/{epochs} done")


def predict_ensemble(models, test_images, device, batch_size=256):
    """Ensemble prediction by averaging logits."""
    for m in models:
        m.eval()

    test_tensor = torch.tensor(test_images).view(-1, 1, 28, 28)
    test_dataset = TensorDataset(test_tensor, torch.zeros(len(test_tensor)))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    all_predictions = []

    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            logits = torch.stack([m(images) for m in models])
            avg_logits = logits.mean(dim=0)
            _, predicted = torch.max(avg_logits, 1)
            all_predictions.extend(predicted.cpu().numpy().tolist())

    return np.array(all_predictions)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load data
    train_images, train_labels, test_images = load_data()
    print(f"Train: {len(train_images)}, Test: {len(test_images)}")

    # K-Fold setup
    n_folds = 5
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    all_models = []
    fold_accs = []

    # Train CNN
    print(f"\n{'='*50}")
    print("Training CNN with 5-fold CV")
    print(f"{'='*50}")

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_images)):
        print(f"\n--- CNN Fold {fold+1}/{n_folds} ---")

        # Create data loaders
        train_tensor = torch.tensor(train_images[train_idx]).view(-1, 1, 28, 28)
        train_labels_tensor = torch.tensor(train_labels[train_idx], dtype=torch.long)
        train_dataset = TensorDataset(train_tensor, train_labels_tensor)
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

        val_tensor = torch.tensor(train_images[val_idx]).view(-1, 1, 28, 28)
        val_labels_tensor = torch.tensor(train_labels[val_idx], dtype=torch.long)
        val_dataset = TensorDataset(val_tensor, val_labels_tensor)
        val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

        # Create and train model
        model = CNN().to(device)
        train_fold(model, train_loader, device, epochs=15)

        # Evaluate
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

    print(f"\nCNN Mean Val Acc: {np.mean(fold_accs):.4f} (+/- {np.std(fold_accs):.4f})")

    # Ensemble prediction
    print("\nGenerating ensemble predictions...")
    predictions = predict_ensemble(all_models, test_images, device)

    # Save submission
    submission = pd.DataFrame({
        'ImageId': range(1, len(predictions) + 1),
        'Label': predictions
    })
    submission.to_csv('submission.csv', index=False)
    print(f"\nSaved submission.csv with {len(predictions)} predictions")
    print(submission.head(10))


if __name__ == '__main__':
    main()