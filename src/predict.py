"""Prediction script for Digit Recognizer."""

import torch
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from dataset import DigitDataset, get_dataloaders
from model import SimpleDNN


def predict(
    test_path: str = 'digit-recognizer/test.csv',
    model_path: str = 'model.pth',
    submission_path: str = 'submission.csv'
):
    """
    Generate predictions for test data.
    
    Args:
        test_path: Path to test data
        model_path: Path to trained model
        submission_path: Path to save submission file
    """
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = SimpleDNN().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print(f"Model loaded from {model_path}")
    
    # Load test data
    test_dataset = DigitDataset(test_path, has_labels=False)
    print(f"Test samples: {len(test_dataset)}")
    
    # Generate predictions
    predictions = []
    
    with torch.no_grad():
        for i in range(len(test_dataset)):
            image, _ = test_dataset[i]
            image = image.unsqueeze(0).to(device)  # Add batch dimension
            
            output = model(image)
            _, predicted = torch.max(output.data, 1)
            predictions.append(predicted.item())
    
    # Create submission file
    submission = pd.DataFrame({
        'ImageId': range(1, len(predictions) + 1),
        'Label': predictions
    })
    submission.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
    print(f"Total predictions: {len(predictions)}")


if __name__ == '__main__':
    predict()