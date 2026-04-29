"""Simple DNN model for Digit Recognizer."""

import torch
import torch.nn as nn


class SimpleDNN(nn.Module):
    """Simple 3-layer fully connected neural network."""
    
    def __init__(self, input_size: int = 784, hidden1: int = 256, 
                 hidden2: int = 128, output_size: int = 10):
        """
        Args:
            input_size: Number of input features (784 for 28x28 images)
            hidden1: First hidden layer size
            hidden2: Second hidden layer size
            output_size: Number of output classes (10 for digits 0-9)
        """
        super(SimpleDNN, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden1),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden2, output_size)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, 784)
        
        Returns:
            Output logits of shape (batch_size, 10)
        """
        return self.network(x)


def load_model(path: str) -> SimpleDNN:
    """Load trained model from file."""
    model = SimpleDNN()
    model.load_state_dict(torch.load(path))
    return model