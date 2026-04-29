"""Entry point for Digit Recognizer - train and predict."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from train import train
from predict import predict


def main():
    """Run training and prediction pipeline."""
    print("=" * 50)
    print("Digit Recognizer - PyTorch DNN")
    print("=" * 50)
    
    # Train
    print("\n[Phase 1] Training...")
    train(epochs=10, batch_size=64)
    
    # Predict
    print("\n[Phase 2] Predicting...")
    predict()
    
    print("\n[Done] Check submission.csv for results!")


if __name__ == '__main__':
    main()