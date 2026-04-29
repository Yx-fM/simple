"""CNN model for Digit Recognizer - High accuracy architecture."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    """
    CNN for MNIST digit classification.
    Architecture based on top Kaggle solutions:
    - 3 convolutional blocks with increasing filters (32→64→128)
    - BatchNormalization for stable training
    - Dropout for regularization
    - Global Average Pooling to reduce parameters
    """
    
    def __init__(self, num_classes: int = 10):
        super(CNN, self).__init__()
        
        # Block 1: 28x28 → 14x14
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        )
        
        # Block 2: 14x14 → 7x7
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        )
        
        # Block 3: 7x7 → 3x3
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        )
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x


class CNNv2(nn.Module):
    """
    Deeper CNN with residual-like connections.
    Architecture inspired by ResNet for MNIST.
    """
    
    def __init__(self, num_classes: int = 10):
        super(CNNv2, self).__init__()
        
        # Initial conv
        self.conv_in = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        # Residual Block 1
        self.res_block1 = self._make_res_block(32, 32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout(0.25)
        
        # Residual Block 2 (with projection for channel change)
        self.res_block2 = self._make_res_block(32, 64)
        self.proj2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=1, stride=1),
            nn.BatchNorm2d(64)
        )
        self.pool2 = nn.MaxPool2d(2, 2)
        self.drop2 = nn.Dropout(0.25)

        # Residual Block 3 (with projection for channel change)
        self.res_block3 = self._make_res_block(64, 128)
        self.proj3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, stride=1),
            nn.BatchNorm2d(128)
        )
        self.pool3 = nn.MaxPool2d(2, 2)
        self.drop3 = nn.Dropout(0.25)
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(128, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def _make_res_block(self, in_ch: int, out_ch: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)
        
        identity = x
        x = self.res_block1(x)
        x = x + identity  # Residual connection (same channels)
        x = self.pool1(x)
        x = self.drop1(x)

        identity = x
        x = self.res_block2(x)
        identity = self.proj2(identity)  # Project to match channels
        x = x + identity
        x = self.pool2(x)
        x = self.drop2(x)

        identity = x
        x = self.res_block3(x)
        identity = self.proj3(identity)  # Project to match channels
        x = x + identity
        x = self.pool3(x)
        x = self.drop3(x)
        
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def load_model(path: str, model_name: str = 'CNN') -> nn.Module:
    """Load trained model from file."""
    if model_name == 'CNNv2':
        model = CNNv2()
    else:
        model = CNN()
    model.load_state_dict(torch.load(path))
    return model