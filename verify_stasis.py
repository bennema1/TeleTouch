import numpy as np
from demo.predictor import LSTMPredictor
import torch

def test_stasis():
    # Path to the latest model
    model_path = "checkpoints/surgical_lstm_v1_20260117_214747/best_model.pth"
    predictor = LSTMPredictor(model_path)
    
    # 1. Test ACTIVE Movement
    print("\n--- Test 1: Active Movement ---")
    active_history = [(i*0.01, i*0.01) for i in range(10)]
    pred_active = predictor.predict(active_history, steps_ahead=3) # 100ms at 30fps is ~3 steps
    print(f"Input end: {active_history[-1]}")
    print(f"Prediction: {pred_active}")
    
    # 2. Test STASIS (Stationary)
    print("\n--- Test 2: Stasis (Stationary) ---")
    stasis_history = [(0.5, 0.5) for _ in range(10)]
    pred_stasis = predictor.predict(stasis_history, steps_ahead=3)
    print(f"Input end: {stasis_history[-1]}")
    print(f"Prediction: {pred_stasis}")
    
    # Assert stasis prediction matches input exactly
    if pred_stasis == stasis_history[-1]:
        print("\nSUCCESS: No drift during stasis!")
    else:
        print("\nFAILURE: Prediction drifted from stationary input.")

if __name__ == "__main__":
    test_stasis()
