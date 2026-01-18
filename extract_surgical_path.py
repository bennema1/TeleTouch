#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pathlib import Path
import os

def extract_path(input_file, output_file):
    """Extract and normalize a real surgical path from ROSMA kinematics."""
    print(f"Reading {input_file}...")
    
    # Read with space delimiter
    df = pd.read_csv(input_file, sep='\s+')
    
    # Integrate velocity to get position (dt = 1/50 for 50Hz)
    dt = 1.0 / 50.0
    vx = df['PSM1_velocity_linear_x'].values
    vy = df['PSM1_velocity_linear_y'].values
    
    # Simple integration
    x = np.cumsum(vx * dt)
    y = np.cumsum(vy * dt)
    
    # Normalize to 0-1 range based on the bounds of this specific path
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    
    if x_max > x_min:
        x_norm = (x - x_min) / (x_max - x_min)
    else:
        x_norm = x * 0 + 0.5
        
    if y_max > y_min:
        y_norm = (y - y_min) / (y_max - y_min)
    else:
        y_norm = y * 0 + 0.5
        
    # Stack into (N, 2)
    path = np.stack([x_norm, y_norm], axis=1)
    
    # Save as numpy
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, path)
    
    print(f"✅ Extracted {len(path)} positions.")
    print(f"💾 Saved to {output_file}")

if __name__ == "__main__":
    trial = "X01_Pea_on_a_Peg_01.txt"
    input_p = Path("data/rosma/kinematics") / trial
    output_p = Path("data/processed") / "real_path.npy"
    
    if input_p.exists():
        extract_path(input_p, output_p)
    else:
        print(f"❌ Error: Trial file {input_p} not found.")
