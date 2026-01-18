#!/usr/bin/env python3
"""
Process ROSMA surgical dataset into training-ready numpy arrays.
Converts JSON annotations to normalized coordinate sequences optimized for cholecystectomy.
"""

import json
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import random


class CholecystectomyDataProcessor:
    """Process ROSMA dataset annotations into cholecystectomy-specific training data."""

    def __init__(self, rosma_dir="data/rosma", output_dir="data/processed"):
        self.rosma_dir = Path(rosma_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Video resolution (common for ROSMA)
        self.video_width = 1920
        self.video_height = 1080

    def load_annotations(self, dataset_name):
        """Load JSON annotations for a ROSMA dataset."""
        dataset_path = self.rosma_dir / dataset_name

        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset {dataset_name} not found at {dataset_path}")

        annotations = {}
        annotation_files = list(dataset_path.glob("*_annotations.json"))

        print(f"Loading {len(annotation_files)} cholecystectomy annotation files from {dataset_name}")

        for ann_file in tqdm(annotation_files, desc="Loading annotations"):
            video_name = ann_file.stem.replace('_annotations', '')

            with open(ann_file, 'r') as f:
                data = json.load(f)
                annotations[video_name] = data

        return annotations

    def extract_trajectories(self, annotations):
        """Extract instrument trajectories from annotations with cholecystectomy focus."""
        trajectories = {}

        print(f"Extracting trajectories from {len(annotations)} cholecystectomy videos")

        for video_name, video_data in tqdm(annotations.items(), desc="Processing videos"):
            if 'frames' not in video_data:
                print(f"Warning: No frames data in {video_name}")
                continue

            trajectory = []

            for frame_data in video_data['frames']:
                frame_id = frame_data.get('frame_id', 0)
                instruments = frame_data.get('instruments', [])

                # Find the main instrument (grasper preferred for cholecystectomy)
                if instruments:
                    # Prioritize grasper for cholecystectomy procedures
                    instrument = None
                    for inst in instruments:
                        if inst.get('type') == 'grasper':
                            instrument = inst
                            break
                    if not instrument:
                        instrument = instruments[0]

                    # Get center coordinates
                    center = instrument.get('center', [0, 0])
                    x, y = center

                    # Normalize coordinates to 0-1
                    x_norm = x / self.video_width
                    y_norm = y / self.video_height

                    # Clamp to valid range
                    x_norm = np.clip(x_norm, 0.0, 1.0)
                    y_norm = np.clip(y_norm, 0.0, 1.0)

                    trajectory.append([x_norm, y_norm, frame_id])

            # Only keep videos with substantial trajectories (cholecystectomy procedures)
            if len(trajectory) > 20:  # Require minimum procedure length
                trajectories[video_name] = np.array(trajectory)

        return trajectories

    def create_training_sequences(self, trajectories, sequence_length=10, prediction_horizon=30):
        """Create cholecystectomy-optimized input-output pairs for training."""
        training_pairs = []

        print(f"Creating cholecystectomy training sequences (input: {sequence_length} frames, output: {prediction_horizon} frames ahead)")

        for video_name, trajectory in tqdm(trajectories.items(), desc="Creating sequences"):
            positions = trajectory[:, :2]  # x, y coordinates only

            # Create sliding window sequences with cholecystectomy-aware sampling
            for i in range(sequence_length, len(positions) - prediction_horizon, 2):  # Sample every 2 frames for variety
                # Input: sequence_length consecutive positions
                input_seq = positions[i - sequence_length:i]  # (sequence_length, 2)

                # Output: position prediction_horizon frames ahead
                target_pos = positions[i + prediction_horizon]  # (2,)

                training_pairs.append({
                    'input': input_seq,
                    'target': target_pos,
                    'video_name': video_name,
                    'frame_idx': i
                })

        return training_pairs

    def split_train_test(self, training_pairs, train_ratio=0.8):
        """Split data by video to ensure temporal separation for cholecystectomy procedures."""
        # Group by video
        videos = {}
        for pair in training_pairs:
            video = pair['video_name']
            if video not in videos:
                videos[video] = []
            videos[video].append(pair)

        # Split videos (not individual pairs) - crucial for cholecystectomy temporal validation
        video_names = list(videos.keys())
        random.shuffle(video_names)

        split_idx = int(len(video_names) * train_ratio)
        train_videos = video_names[:split_idx]
        test_videos = video_names[split_idx:]

        # Collect pairs for each split
        train_pairs = []
        test_pairs = []

        for video, pairs in videos.items():
            if video in train_videos:
                train_pairs.extend(pairs)
            else:
                test_pairs.extend(pairs)

        return train_pairs, test_pairs

    def save_processed_data(self, train_pairs, test_pairs):
        """Save cholecystectomy-processed data as numpy arrays."""
        print(f"Saving {len(train_pairs)} training pairs and {len(test_pairs)} test pairs")

        # Convert to numpy arrays
        train_inputs = np.array([p['input'] for p in train_pairs])  # (N, seq_len, 2)
        train_targets = np.array([p['target'] for p in train_pairs])  # (N, 2)
        test_inputs = np.array([p['input'] for p in test_pairs])  # (N, seq_len, 2)
        test_targets = np.array([p['target'] for p in test_pairs])  # (N, 2)

        # Save arrays
        np.save(self.output_dir / 'train_inputs.npy', train_inputs)
        np.save(self.output_dir / 'train_targets.npy', train_targets)
        np.save(self.output_dir / 'test_inputs.npy', test_inputs)
        np.save(self.output_dir / 'test_targets.npy', test_targets)

        # Save metadata with cholecystectomy-specific information
        metadata = {
            'train_samples': len(train_pairs),
            'test_samples': len(test_pairs),
            'sequence_length': train_inputs.shape[1],
            'feature_dims': train_inputs.shape[2],
            'train_videos': list(set(p['video_name'] for p in train_pairs)),
            'test_videos': list(set(p['video_name'] for p in test_pairs)),
            'created_by': 'CholecystectomyDataProcessor',
            'video_resolution': [self.video_width, self.video_height],
            'specialization': 'laparoscopic_cholecystectomy',
            'features': ['x_position', 'y_position'],
            'description': 'ROSMA dataset processed for cholecystectomy latency compensation'
        }

        with open(self.output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        print("✅ Cholecystectomy data saved successfully!")
        print(f"Training samples: {len(train_pairs)}")
        print(f"Test samples: {len(test_pairs)}")
        print(f"Input shape: {train_inputs.shape}")
        print(f"Output shape: {train_targets.shape}")
        print(f"Files saved in: {self.output_dir}")

    def process_dataset(self, dataset_name='ROSMAG40'):
        """Complete cholecystectomy-specific processing pipeline."""
        print("🔬 Cholecystectomy Data Processing Pipeline")
        print("=" * 50)

        # Step 1: Load annotations
        annotations = self.load_annotations(dataset_name)

        # Step 2: Extract trajectories
        trajectories = self.extract_trajectories(annotations)
        print(f"📊 Extracted trajectories from {len(trajectories)} cholecystectomy videos")

        # Step 3: Create training sequences
        training_pairs = self.create_training_sequences(trajectories)
        print(f"🔄 Created {len(training_pairs)} training sequences")

        # Step 4: Split train/test
        train_pairs, test_pairs = self.split_train_test(training_pairs)
        print(f"✂️ Split into {len(train_pairs)} training and {len(test_pairs)} test samples")

        # Step 5: Save processed data
        self.save_processed_data(train_pairs, test_pairs)

        # Print summary
        print("\n✅ Cholecystectomy processing complete!")
        print(f"Training samples: {len(train_pairs)}")
        print(f"Test samples: {len(test_pairs)}")
        print(f"Input shape: {train_pairs[0]['input'].shape}")
        print(f"Output shape: {train_targets[0].shape}")

        return len(train_pairs), len(test_pairs)


def main():
    """Main processing function."""
    parser = argparse.ArgumentParser(description="Process ROSMA dataset for cholecystectomy")
    parser.add_argument('--rosma-dir', type=str, default='data/rosma',
                       help='Directory containing ROSMA datasets')
    parser.add_argument('--output-dir', type=str, default='data/processed',
                       help='Directory to save processed data')
    parser.add_argument('--dataset', type=str, default='ROSMAG40',
                       help='Which ROSMA dataset to process (ROSMAG40 or ROSMAT24)')
    parser.add_argument('--sequence-length', type=int, default=10,
                       help='Length of input sequences (default: 10)')
    parser.add_argument('--prediction-horizon', type=int, default=30,
                       help='Frames to predict ahead (default: 30)')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                       help='Train/test split ratio (default: 0.8)')

    args = parser.parse_args()

    processor = CholecystectomyDataProcessor(args.rosma_dir, args.output_dir)
    processor.process_dataset(args.dataset)


if __name__ == "__main__":
    main()