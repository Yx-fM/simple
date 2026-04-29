"""Ensemble prediction from 10 models (5 CNN + 5 CNNv2 folds)."""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from model_cnn import CNN, CNNv2


def load_test_data():
    test_df = pd.read_csv('digit-recognizer/test.csv')
    test_images = test_df.values / 255.0
    return test_images.astype('float32')


def ensemble_predict():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load test data
    test_images = load_test_data()
    test_tensor = torch.tensor(test_images).view(-1, 1, 28, 28)
    print(f"Test samples: {len(test_images)}")

    # Load all 10 models
    models = []

    print("\nLoading CNN models...")
    for i in range(5):
        model = CNN().to(device)
        model.load_state_dict(torch.load(f'model_cnn_fold{i}.pth'))
        model.eval()
        models.append(model)
        print(f"  Loaded CNN fold {i}")

    print("\nLoading CNNv2 models...")
    for i in range(5):
        model = CNNv2().to(device)
        model.load_state_dict(torch.load(f'model_cnnv2_fold{i}.pth'))
        model.eval()
        models.append(model)
        print(f"  Loaded CNNv2 fold {i}")

    print(f"\nTotal models: {len(models)}")

    # Ensemble prediction by averaging logits
    all_predictions = []
    batch_size = 256

    with torch.no_grad():
        for start in range(0, len(test_tensor), batch_size):
            end = min(start + batch_size, len(test_tensor))
            batch = test_tensor[start:end].to(device)

            # Get logits from all models
            logits_list = []
            for model in models:
                logits = model(batch)
                logits_list.append(logits)

            # Average logits
            avg_logits = torch.stack(logits_list).mean(dim=0)
            _, predicted = torch.max(avg_logits, 1)
            all_predictions.extend(predicted.cpu().numpy().tolist())

            if (start + batch_size) % 5000 == 0 or end == len(test_tensor):
                print(f"Processed {end}/{len(test_tensor)}")

    # Save submission
    submission = pd.DataFrame({
        'ImageId': range(1, len(all_predictions) + 1),
        'Label': all_predictions
    })
    submission.to_csv('submission.csv', index=False)
    print(f"\nSaved submission.csv with {len(all_predictions)} predictions")
    print("\nSample predictions:")
    print(submission.head(10))


if __name__ == '__main__':
    ensemble_predict()