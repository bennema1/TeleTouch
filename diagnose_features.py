import numpy as np
import json
from pathlib import Path

def diagnose():
    data_path = Path("data/processed")
    inputs = np.load(data_path / "train_inputs.npy")
    metadata_path = data_path / "metadata.json"
    
    with open(metadata_path, 'r') as f:
        meta = json.load(f)
        
    print(f"Metadata Features: {meta['features']}")
    print(f"Inputs Shape: {inputs.shape}")
    
    # feature indices: 0:x, 1:y, 2:vx, 3:vy, 4:ax, 5:ay
    for i, feature in enumerate(meta['features']):
        f_data = inputs[:, :, i]
        print(f"Feature: {feature}")
        print(f"  Min: {f_data.min():.4f}")
        print(f"  Max: {f_data.max():.4f}")
        print(f"  Mean: {f_data.mean():.4f}")
        print(f"  Std: {f_data.std():.4f}")

if __name__ == "__main__":
    diagnose()
