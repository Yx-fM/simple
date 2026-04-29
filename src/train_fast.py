"""Simple training script for Digit Recognizer CNN - faster version."""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from dataset import get_dataloaders
from model_cnn import CNN


def train(
    train_path: str = 'digit-recognizer/train.csv',
    epochs: int = 10,
    batch_size: int = 128,
    learning_rate: float = 0.001,
    model_path: str = 'model_cnn.pth',
    use_augmentation: bool = True
):
    """Train CNN with data augmentation."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    train_loader, val_loader, _ = get_dataloaders(
        train_path=train_path,
        test_path='digit-recognizer/test.csv',
        batch_size=batch_size,
        use_augmentation=use_augmentation
    )
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")

    # Initialize model
    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        batch_count = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            batch_count += 1
            if batch_count % 100 == 0:
                print(f"  Batch {batch_count}/{len(train_loader)}, Loss: {loss.item():.4f}")

        train_acc = train_correct / train_total

        # Validation
        model.eval()
        val_correct = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                val_correct += (predicted == labels).sum().item()

        val_acc = val_correct / len(val_loader.dataset)
        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()

        print(f"Epoch [{epoch+1}/{epochs}] Train Acc: {train_acc:.4f} Val Acc: {val_acc:.4f}")

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    torch.save(model.state_dict(), model_path)
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    return model


if __name__ == '__main__':
    train(epochs=10, batch_size=128, use_augmentation=True)