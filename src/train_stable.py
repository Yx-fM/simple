"""Stable CNN training - minimal augmentation for better generalization."""

import torch
import torch.nn as nn
import torch.optim as optim
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dataset_fast import get_dataloaders
from model_cnn import CNN


def train_stable(epochs=20, batch_size=128, lr=0.001):
    """Train CNN with minimal augmentation."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # NO augmentation - just normalization
    train_loader, val_loader, _ = get_dataloaders(
        train_path='digit-recognizer/train.csv',
        test_path='digit-recognizer/test.csv',
        batch_size=batch_size,
        use_augmentation=False  # NO augmentation for stable training
    )
    print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}")

    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True
    )

    best_acc = 0.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        correct, total = 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = correct / total

        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = val_correct / val_total
        scheduler.step(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = model.state_dict().copy()

        print(f"Epoch {epoch+1}/{epochs} Train: {train_acc:.4f} Val: {val_acc:.4f}")

    if best_state:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), 'model_stable.pth')
    print(f"Best Val Acc: {best_acc:.4f}")
    return model


if __name__ == '__main__':
    train_stable(epochs=20)