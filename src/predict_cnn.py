"""Prediction script for Digit Recognizer with TTA support."""

import torch
import torch.nn.functional as F
import pandas as pd
from pathlib import Path
import sys
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from dataset import DigitDataset
from model_cnn import CNN, CNNv2
from torch.utils.data import DataLoader


def apply_tta_transforms(image: torch.Tensor, transform_idx: int) -> torch.Tensor:
    """
    Apply different augmentations for TTA.
    Returns original or augmented version of the image.
    """
    if transform_idx == 0:
        # Original
        return image
    elif transform_idx == 1:
        # Rotate +5 degrees
        angle = 5.0 * np.pi / 180.0
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        theta = torch.tensor([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0]
        ], dtype=torch.float32)
        grid = F.affine_grid(theta.unsqueeze(0), image.unsqueeze(0).size(),
                             align_corners=False)
        return F.grid_sample(image.unsqueeze(0), grid, align_corners=False,
                             mode='bilinear', padding_mode='zeros').squeeze(0)
    elif transform_idx == 2:
        # Rotate -5 degrees
        angle = -5.0 * np.pi / 180.0
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        theta = torch.tensor([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0]
        ], dtype=torch.float32)
        grid = F.affine_grid(theta.unsqueeze(0), image.unsqueeze(0).size(),
                             align_corners=False)
        return F.grid_sample(image.unsqueeze(0), grid, align_corners=False,
                             mode='bilinear', padding_mode='zeros').squeeze(0)
    elif transform_idx == 3:
        # Shift right
        dx = 2
        theta = torch.tensor([
            [1, 0, dx / 14.0],
            [0, 1, 0]
        ], dtype=torch.float32)
        grid = F.affine_grid(theta.unsqueeze(0), image.unsqueeze(0).size(),
                             align_corners=False)
        return F.grid_sample(image.unsqueeze(0), grid, align_corners=False,
                             mode='bilinear', padding_mode='zeros').squeeze(0)
    elif transform_idx == 4:
        # Shift left
        dx = -2
        theta = torch.tensor([
            [1, 0, dx / 14.0],
            [0, 1, 0]
        ], dtype=torch.float32)
        grid = F.affine_grid(theta.unsqueeze(0), image.unsqueeze(0).size(),
                             align_corners=False)
        return F.grid_sample(image.unsqueeze(0), grid, align_corners=False,
                             mode='bilinear', padding_mode='zeros').squeeze(0)
    else:
        return image


def predict(
    test_path: str = 'digit-recognizer/test.csv',
    model_path: str = 'model_cnn.pth',
    submission_path: str = 'submission.csv',
    model_name: str = 'CNN',
    use_tta: bool = True,
    tta_count: int = 5
):
    """
    Generate predictions for test data with optional TTA.

    Args:
        test_path: Path to test data
        model_path: Path to trained model
        submission_path: Path to save submission file
        model_name: 'CNN' or 'CNNv2'
        use_tta: Whether to use Test Time Augmentation
        tta_count: Number of TTA transforms to average
    """
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"TTA enabled: {use_tta}, transforms: {tta_count}")

    # Load model
    if model_name == 'CNNv2':
        model = CNNv2().to(device)
    else:
        model = CNN().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print(f"Model loaded from {model_path}")

    # Load test data
    test_dataset = DigitDataset(test_path, has_labels=False)
    print(f"Test samples: {len(test_dataset)}")

    # Generate predictions with TTA
    predictions = []
    softmax = torch.nn.Softmax(dim=1)

    with torch.no_grad():
        for i in range(len(test_dataset)):
            image, _ = test_dataset[i]
            image = image.to(device)

            # Apply TTA transforms and average predictions
            logits_list = []

            for tta_idx in range(tta_count):
                aug_image = apply_tta_transforms(image.cpu(), tta_idx)
                aug_image = aug_image.unsqueeze(0).to(device)

                output = model(aug_image)
                logits_list.append(output)

            # Average logits across TTA transforms
            avg_logits = torch.stack(logits_list).mean(dim=0)
            _, predicted = torch.max(avg_logits.data, 1)
            predictions.append(predicted.item())

            if (i + 1) % 5000 == 0:
                print(f"Processed {i + 1}/{len(test_dataset)}")

    # Create submission file
    submission = pd.DataFrame({
        'ImageId': range(1, len(predictions) + 1),
        'Label': predictions
    })
    submission.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
    print(f"Total predictions: {len(predictions)}")

    # Show sample
    print("\nSample predictions:")
    print(submission.head(10))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='CNN', choices=['CNN', 'CNNv2'])
    parser.add_argument('--no_tta', action='store_true')
    parser.add_argument('--tta_count', type=int, default=5)
    args = parser.parse_args()

    predict(model_name=args.model, use_tta=not args.no_tta,
            tta_count=args.tta_count)