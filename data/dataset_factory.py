import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from typing import Optional, List, Dict, Any
import numpy as np
from pathlib import Path

from .rosma_dataset import CholecystectomyDataset, EnhancedCholecystectomyDataset, CholecystectomySyntheticDataset
from .augmentation import SurgicalDataAugmenter, TemporalFeatureExtractor


class CholecystectomyAugmentedDataset(Dataset):
    """Dataset wrapper that applies cholecystectomy-specialized augmentations."""

    def __init__(self, base_dataset: Dataset, augmenter: SurgicalDataAugmenter,
                 feature_extractor: Optional[TemporalFeatureExtractor] = None,
                 augment_prob: float = 0.7):
        self.base_dataset = base_dataset
        self.augmenter = augmenter
        self.feature_extractor = feature_extractor
        self.augment_prob = augment_prob

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        sample = self.base_dataset[idx]

        # Apply augmentation with probability (disabled for now)
        # if np.random.random() < self.augment_prob:
        #     sample = self.augmenter.augment_batch(sample)

        # Add temporal features if extractor provided
        if self.feature_extractor is not None:
            features_dict = self.feature_extractor(sample['input'])
            # For now, just return the original input - temporal features could be added later
            sample['temporal_features'] = features_dict

        return sample


class CholecystectomyDatasetFactory:
    """Factory for creating cholecystectomy-specialized surgical movement datasets."""

    def __init__(self, use_rosma: bool = True, use_pygame: bool = True,
                 use_synthetic: bool = True, synthetic_samples: int = 50000):
        self.use_rosma = use_rosma
        self.use_pygame = use_pygame
        self.use_synthetic = use_synthetic
        self.synthetic_samples = synthetic_samples

        # Cholecystectomy-specialized augmenters
        self.augmenter = SurgicalDataAugmenter()
        self.feature_extractor = TemporalFeatureExtractor()

        self.datasets = {}

    def create_combined_dataset(self, pygame_file: Optional[str] = None,
                               demo_data_file: Optional[str] = None,
                               sequence_length: int = 10,
                               prediction_horizon: int = 30) -> Dataset:
        """
        Create combined dataset specialized for cholecystectomy movements.

        Args:
            pygame_file: Path to PyGame training data file
            demo_data_file: Path to demo data file
            sequence_length: Input sequence length
            prediction_horizon: Prediction horizon

        Returns:
            Combined dataset with cholecystectomy-specialized augmentation
        """
        datasets_to_combine = []

        # ROSMA cholecystectomy dataset (primary data)
        if self.use_rosma:
            try:
                cholec_dataset = CholecystectomyDataset()
                trial_names = list(cholec_dataset.kinematic_data.keys())[:10]  # Use first 10 trials

                rosma_datasets = []
                for trial in trial_names:
                    try:
                        features = cholec_dataset.get_cholecystectomy_kinematic_features(trial)
                        trial_dataset = self._create_sequence_dataset(
                            features, sequence_length, prediction_horizon,
                            dataset_type='cholec_rosma', trial_name=trial
                        )
                        rosma_datasets.append(trial_dataset)
                    except Exception as e:
                        print(f"Skipping ROSMA trial {trial}: {e}")
                        continue

                if rosma_datasets:
                    rosma_combined = ConcatDataset(rosma_datasets)
                    datasets_to_combine.append(rosma_combined)
                    self.datasets['rosma'] = rosma_combined
                    print(f"Added {len(rosma_combined)} cholecystectomy ROSMA samples")

            except Exception as e:
                print(f"ROSMA cholecystectomy dataset not available: {e}")

        # Enhanced cholecystectomy dataset
        if self.use_pygame and pygame_file and Path(pygame_file).exists():
            try:
                cholecystectomy_pygame_dataset = EnhancedCholecystectomyDataset(
                    pygame_file, sequence_length, prediction_horizon
                )
                datasets_to_combine.append(cholecystectomy_pygame_dataset)
                self.datasets['pygame'] = cholecystectomy_pygame_dataset
                print(f"Added {len(cholecystectomy_pygame_dataset)} enhanced cholecystectomy samples")
            except Exception as e:
                print(f"Cholecystectomy PyGame dataset loading failed: {e}")

        # Demo data (collected from actual demo sessions)
        if demo_data_file and Path(demo_data_file).exists():
            try:
                demo_dataset = EnhancedCholecDataset(
                    demo_data_file, sequence_length, prediction_horizon
                )
                datasets_to_combine.append(demo_dataset)
                self.datasets['demo'] = demo_dataset
                print(f"Added {len(demo_dataset)} cholecystectomy demo samples")
            except Exception as e:
                print(f"Demo dataset loading failed: {e}")

        # Cholecystectomy-specific synthetic dataset
        if self.use_synthetic:
            synthetic_dataset = CholecystectomySyntheticDataset(
                self.synthetic_samples, sequence_length, prediction_horizon
            )
            datasets_to_combine.append(synthetic_dataset)
            self.datasets['synthetic'] = synthetic_dataset
            print(f"Added {len(synthetic_dataset)} cholecystectomy synthetic samples")

        if not datasets_to_combine:
            raise ValueError("No cholecystectomy datasets could be created from available sources")

        # Combine all datasets
        combined_dataset = ConcatDataset(datasets_to_combine)

        # Apply cholecystectomy-specialized augmentation wrapper
        augmented_dataset = CholecystectomyAugmentedDataset(
            combined_dataset, self.augmenter, self.feature_extractor
        )

        return augmented_dataset

    def _create_sequence_dataset(self, features: np.ndarray, sequence_length: int,
                               prediction_horizon: int, dataset_type: str,
                               trial_name: str = "") -> Dataset:
        """Create sequence dataset from cholecystectomy feature array."""

        class CholecSequenceDataset(Dataset):
            def __init__(self, features, seq_len, pred_horizon, ds_type, trial):
                self.features = features
                self.seq_len = seq_len
                self.pred_horizon = pred_horizon
                self.dataset_type = ds_type
                self.trial_name = trial

                self.valid_indices = []
                for i in range(seq_len, len(features) - pred_horizon):
                    self.valid_indices.append(i)

            def __len__(self):
                return len(self.valid_indices)

            def __getitem__(self, idx):
                center_idx = self.valid_indices[idx]

                input_seq = self.features[center_idx - self.seq_len:center_idx]
                target_pos = self.features[center_idx + self.pred_horizon, :2]

                return {
                    'input': torch.tensor(input_seq, dtype=torch.float32),
                    'target': torch.tensor(target_pos, dtype=torch.float32),
                    'metadata': {
                        'dataset_type': self.dataset_type,
                        'trial_name': self.trial_name,
                        'center_idx': center_idx,
                        'target_idx': center_idx + self.pred_horizon
                    }
                }

        return CholecSequenceDataset(features, sequence_length, prediction_horizon, dataset_type, trial_name)

    def create_data_loader(self, dataset: Dataset, batch_size: int = 32,
                          shuffle: bool = True, num_workers: int = 0) -> DataLoader:
        """
        Create DataLoader with optimized settings for cholecystectomy data.

        Args:
            dataset: Dataset to load
            batch_size: Batch size
            shuffle: Whether to shuffle
            num_workers: Number of worker processes

        Returns:
            Configured DataLoader
        """
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
            prefetch_factor=2 if num_workers > 0 else None
        )

    def get_dataset_stats(self) -> Dict[str, Any]:
        """Get statistics about the loaded cholecystectomy datasets."""
        stats = {}
        for name, dataset in self.datasets.items():
            stats[name] = {
                'samples': len(dataset),
                'type': type(dataset).__name__
            }
        return stats


def create_cholecystectomy_training_setup(pygame_file: Optional[str] = None,
                                         demo_data_file: Optional[str] = None,
                                         batch_size: int = 32,
                                         sequence_length: int = 10,
                                         prediction_horizon: int = 30) -> tuple:
    """
    Create complete cholecystectomy-specialized training setup.

    Returns:
        (train_loader, val_loader, dataset_factory)
    """
    factory = CholecystectomyDatasetFactory()

    # Create combined cholecystectomy dataset
    full_dataset = factory.create_combined_dataset(
        pygame_file=pygame_file,
        demo_data_file=demo_data_file,
        sequence_length=sequence_length,
        prediction_horizon=prediction_horizon
    )

    # Split into train/val (80/20)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Create data loaders
    train_loader = factory.create_data_loader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = factory.create_data_loader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, factory