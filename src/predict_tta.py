"""Generate predictions with TTA."""

import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dataset_fast import DigitDataset
from model_cnn import CNN
from torch.utils.data import DataLoader


def apply_tta(image, idx):
    """Apply TTA transform."""
    if idx == 0:
        return image
    elif idx == 1:  # Rotate +5
        angle = 5 * np.pi / 180
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        theta = torch.tensor([[cos_a, -sin_a, 0], [sin_a, cos_a, 0]], dtype=torch.float32)
        grid = F.affine_grid(theta.unsqueeze(0), image.unsqueeze(0).size(), align_corners=False)
        return F.grid_sample(image.unsqueeze(0), grid, align_corners=False).squeeze(0)
    elif idx == 2:  # Rotate -5
        angle = -5 * np.pi / 180
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        theta = torch.tensor([[cos_a, -sin_a, 0], [sin_a, cos_a, 0]], dtype=torch.float32)
        grid = F.affine_grid(theta.unsqueeze(0), image.unsqueeze(0).size(), align_corners=False)
        return F.grid_sample(image.unsqueeze(0), grid, align_corners=False).squeeze(0)
    elif idx == 3:  # Shift right
        theta = torch.tensor([[1, 0, 0.1], [0, 1, 0]], dtype=torch.float32)
        grid = F.affine_grid(theta.unsqueeze(0), image.unsqueeze(0).size(), align_corners=False)
        return F.grid_sample(image.unsqueeze(0), grid, align_corners=False).squeeze(0)
    elif idx == 4:  # Shift left
        theta = torch.tensor([[1, 0, -0.1], [0, 1, 0]], dtype=torch.float32)
        grid = F.affine_grid(theta.unsqueeze(0), image.unsqueeze(0).size(), align_corners=False)
        return F.grid_sample(image.unsqueeze(0), grid, align_corners=False).squeeze(0)
    return image


def predict():
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

    # Predict with TTA
    predictions = []
    tta_count = 5

    with torch.no_grad():
        for i in range(len(test_dataset)):
            image, _ = test_dataset[i]
            image = image.to(device)

            logits_list = []
            for tta_idx in range(tta_count):
                aug_img = apply_tta(image.cpu(), tta_idx).unsqueeze(0).to(device)
                output = model(aug_img)
                logits_list.append(output)

            avg_logits = torch.stack(logits_list).mean(dim=0)
            _, predicted = torch.max(avg_logits, 1)
            predictions.append(predicted.item())

            if (i + 1) % 5000 == 0:
                print(f"Processed {i + 1}/{len(test_dataset)}")

    # Save submission
    submission = pd.DataFrame({
        'ImageId': range(1, len(predictions) + 1),
        'Label': predictions
    })
    submission.to_csv('submission.csv', index=False)
    print(f"Saved submission.csv with {len(predictions)} predictions")
    print(submission.head(10))


if __name__ == '__main__':
    predict()