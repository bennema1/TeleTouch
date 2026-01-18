"""
Simple dataset for basic trajectory prediction training.
"""

import torch
from torch.utils.data import Dataset
import numpy as np


class SimpleTrajectoryDataset(Dataset):
    """Simple dataset that loads numpy arrays directly."""

    def __init__(self, inputs_path, targets_path):
        self.inputs = np.load(inputs_path)   # (N, seq_len, 2)
        self.targets = np.load(targets_path) # (N, 2)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return {
            'input': torch.tensor(self.inputs[idx], dtype=torch.float32),
            'target': torch.tensor(self.targets[idx], dtype=torch.float32)
        }