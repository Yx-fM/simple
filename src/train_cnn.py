"""Training script for Digit Recognizer with CNN."""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import sys
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from dataset import get_dataloaders
from model_cnn import CNN, CNNv2


def train(
    train_path: str = 'digit-recognizer/train.csv',
    epochs: int = 30,
    batch_size: int = 128,
    learning_rate: float = 0.001,
    model_path: str = 'model_cnn.pth',
    model_name: str = 'CNN',
    use_augmentation: bool = True
):
    """
    Train the CNN model.

    Args:
        train_path: Path to training data
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate for optimizer
        model_path: Path to save trained model
        model_name: 'CNN' or 'CNNv2'
        use_augmentation: Whether to use data augmentation
    """
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Model: {model_name}")
    print(f"Data augmentation: {use_augmentation}")

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
    if model_name == 'CNNv2':
        model = CNNv2().to(device)
    else:
        model = CNN().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Cosine annealing learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training loop
    best_val_acc = 0.0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_acc = train_correct / train_total

        # Validation
        model.eval()
        val_correct = 0
        val_loss = 0.0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_correct += (predicted == labels).sum().item()

        val_acc = val_correct / len(val_loader.dataset)

        # Update learning rate
        scheduler.step()

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Loss: {train_loss/len(train_loader):.4f} "
              f"Train Acc: {train_acc:.4f} "
              f"Val Acc: {val_acc:.4f} "
              f"LR: {current_lr:.6f}")

    # Save best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    torch.save(model.state_dict(), model_path)
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Model saved to {model_path}")

    return model


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--model', type=str, default='CNN', choices=['CNN', 'CNNv2'])
    parser.add_argument('--no_aug', action='store_true')
    args = parser.parse_args()

    train(epochs=args.epochs, batch_size=args.batch_size,
          learning_rate=args.lr, model_name=args.model,
          use_augmentation=not args.no_aug)