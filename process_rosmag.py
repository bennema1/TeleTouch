#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json

def process_rosmag_data(input_dir, output_dir, sequence_length=10, prediction_horizon=30):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    kinematics_dir = input_path / "kinematics"
    if not kinematics_dir.exists():
        print(f"Error: Kinematics directory not found at {kinematics_dir}")
        return

    files = list(kinematics_dir.glob("*.txt"))
    print(f"Processing {len(files)} ROSMAG40 files...")
    
    all_inputs = []
    all_targets = []
    
    dt = 1.0 / 50.0  # Assumed 50Hz sampling
    
    for f in tqdm(files):
        try:
            # Read with space delimiter
            df = pd.read_csv(f, sep='\s+')
            
            # Use PSM1 linear velocity
            vx = df['PSM1_velocity_linear_x'].values
            vy = df['PSM1_velocity_linear_y'].values
            
            # Integrate velocity to get relative position (displacement)
            # Starting at center (0.5, 0.5)
            pos_x = 0.5 + np.cumsum(vx * dt)
            pos_y = 0.5 + np.cumsum(vy * dt)
            
            # Normalize to 0-1 range based on the trial's bounding box
            x_min, x_max = pos_x.min(), pos_x.max()
            y_min, y_max = pos_y.min(), pos_y.max()
            
            if x_max > x_min:
                pos_x = (pos_x - x_min) / (x_max - x_min)
            else:
                pos_x = pos_x * 0 + 0.5
                
            if y_max > y_min:
                pos_y = (pos_y - y_min) / (y_max - y_min)
            else:
                pos_y = pos_y * 0 + 0.5
                
            positions = np.stack([pos_x, pos_y], axis=1) # (N, 2)
            
            # Create sequences
            for i in range(sequence_length, len(positions) - prediction_horizon):
                input_seq = positions[i-sequence_length:i]
                target_pos = positions[i+prediction_horizon]
                
                all_inputs.append(input_seq)
                all_targets.append(target_pos)
                
        except Exception as e:
            print(f"Error processing {f.name}: {e}")
            
    if not all_inputs:
        print("No sequences generated!")
        return
        
    inputs_np = np.array(all_inputs, dtype=np.float32)
    targets_np = np.array(all_targets, dtype=np.float32)
    
    # Split into train and test
    idx = np.arange(len(inputs_np))
    np.random.shuffle(idx)
    split = int(0.8 * len(idx))
    
    train_idx = idx[:split]
    test_idx = idx[split:]
    
    np.save(output_path / "train_inputs.npy", inputs_np[train_idx])
    np.save(output_path / "train_targets.npy", targets_np[train_idx])
    np.save(output_path / "test_inputs.npy", inputs_np[test_idx])
    np.save(output_path / "test_targets.npy", targets_np[test_idx])
    
    # Save metadata
    metadata = {
        "num_files": len(files),
        "total_samples": len(all_inputs),
        "train_samples": len(train_idx),
        "test_samples": len(test_idx),
        "sequence_length": sequence_length,
        "prediction_horizon": prediction_horizon,
        "features": ["x", "y"]
    }
    
    with open(output_path / "metadata.json", 'w') as mf:
        json.dump(metadata, mf, indent=4)
        
    print(f"Processed {len(all_inputs)} sequences. Saved to {output_path}")

if __name__ == "__main__":
    process_rosmag_data(
        "data/rosma",
        "data/processed",
        sequence_length=10,
        prediction_horizon=3
    )
