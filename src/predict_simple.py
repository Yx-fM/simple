"""Simple prediction without TTA - just verify the model works."""

import torch
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dataset_fast import DigitDataset
from model_cnn import CNN
from torch.utils.data import DataLoader


def predict_simple():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model
    model = CNN().to(device)
    model.load_state_dict(torch.load('model_cnn.pth'))
    model.eval()
    print("Model loaded")

    # Load test data
    test_dataset = DigitDataset('digit-recognizer/test.csv', has_labels=False)
    print(f"Test samples: {len(test_dataset)}")

    # Predict WITHOUT TTA
    predictions = []

    with torch.no_grad():
        for i in range(len(test_dataset)):
            image, _ = test_dataset[i]
            image = image.unsqueeze(0).to(device)
            output = model(image)
            _, predicted = torch.max(output, 1)
            predictions.append(predicted.item())

            if (i + 1) % 10000 == 0:
                print(f"Processed {i + 1}/{len(test_dataset)}")

    # Save submission
    submission = pd.DataFrame({
        'ImageId': range(1, len(predictions) + 1),
        'Label': predictions
    })
    submission.to_csv('submission_simple.csv', index=False)
    print(f"Saved submission_simple.csv with {len(predictions)} predictions")
    print(submission.head(10))


if __name__ == '__main__':
    predict_simple()