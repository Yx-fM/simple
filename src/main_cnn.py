"""Entry point for CNN-based Digit Recognizer."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from train_cnn import train
from predict_cnn import predict


def main():
    """Run training and prediction pipeline with CNN."""
    print("=" * 60)
    print("Digit Recognizer - PyTorch CNN (High Accuracy)")
    print("=" * 60)

    # Configuration
    EPOCHS = 30
    BATCH_SIZE = 128
    LEARNING_RATE = 0.001
    MODEL_NAME = 'CNN'  # or 'CNNv2' for deeper model
    USE_AUGMENTATION = True
    USE_TTA = True
    TTA_COUNT = 5

    print(f"\nConfiguration:")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  Data Augmentation: {USE_AUGMENTATION}")
    print(f"  TTA: {USE_TTA} (count={TTA_COUNT})")

    # Train
    print("\n[Phase 1] Training CNN...")
    model_path = f'model_{MODEL_NAME.lower()}.pth'
    train(epochs=EPOCHS, batch_size=BATCH_SIZE,
          learning_rate=LEARNING_RATE,
          model_path=model_path,
          model_name=MODEL_NAME,
          use_augmentation=USE_AUGMENTATION)

    # Predict
    print("\n[Phase 2] Generating predictions with TTA...")
    predict(model_path=model_path,
            submission_path='submission.csv',
            model_name=MODEL_NAME,
            use_tta=USE_TTA,
            tta_count=TTA_COUNT)

    print("\n" + "=" * 60)
    print("[Done] Check submission.csv for results!")
    print("=" * 60)


if __name__ == '__main__':
    main()