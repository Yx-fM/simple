"""Training script for Digit Recognizer."""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from dataset import get_dataloaders
from model import SimpleDNN


def train(
    train_path: str = 'digit-recognizer/train.csv',
    epochs: int = 10,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    model_path: str = 'model.pth'
):
    """
    Train the DNN model.
    
    Args:
        train_path: Path to training data
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate for optimizer
        model_path: Path to save trained model
    """
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    train_loader, val_loader, _ = get_dataloaders(
        train_path=train_path,
        test_path='digit-recognizer/test.csv',
        batch_size=batch_size
    )
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Initialize model
    model = SimpleDNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        
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
            train_correct += (predicted == labels).sum().item()
        
        train_acc = train_correct / len(train_loader.dataset)
        
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
        
        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Loss: {train_loss/len(train_loader):.4f} "
              f"Train Acc: {train_acc:.4f} "
              f"Val Acc: {val_acc:.4f}")
    
    # Save model
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    return model


if __name__ == '__main__':
    train()