"""
Generate synthetic training data for cholecystectomy trajectory prediction.
Creates realistic laparoscopic surgical movement patterns.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def generate_cholec_trajectory(duration_seconds=10, fps=30, seed=None):
    """
    Generate a synthetic cholecystectomy trajectory.

    Creates realistic laparoscopic instrument movements including:
    - Smooth dissection motions
    - Tissue retraction patterns
    - Clipping movements
    - Realistic noise and tremor
    """
    if seed is not None:
        np.random.seed(seed)

    n_frames = int(duration_seconds * fps)
    dt = 1.0 / fps

    # Time array
    t = np.arange(n_frames) * dt

    # Base trajectory: combination of smooth curves and procedural movements
    # Phase 1: Approach and initial dissection (0-3s)
    phase1_frames = int(3 * fps)
    t1 = t[:phase1_frames]

    # Smooth approach curve
    x1 = 0.3 + 0.1 * np.sin(2 * np.pi * t1 / 2) * (1 - np.exp(-t1))
    y1 = 0.4 + 0.05 * np.cos(2 * np.pi * t1 / 1.5) * (1 - np.exp(-t1))

    # Phase 2: Tissue dissection and retraction (3-7s)
    phase2_frames = int(4 * fps)
    t2 = t[phase1_frames:phase1_frames + phase2_frames]

    # More complex dissection pattern
    x2 = 0.4 + 0.15 * np.sin(4 * np.pi * t2 / 3) + 0.05 * np.cos(8 * np.pi * t2 / 2)
    y2 = 0.45 + 0.1 * np.cos(3 * np.pi * t2 / 2.5) + 0.03 * np.sin(6 * np.pi * t2 / 1.8)

    # Phase 3: Final positioning and clip placement (7-10s)
    phase3_frames = n_frames - phase1_frames - phase2_frames
    t3 = t[phase1_frames + phase2_frames:]

    # Precise clip placement movements
    x3 = 0.55 + 0.08 * np.sin(10 * np.pi * t3 / 2) * np.exp(-2 * t3)
    y3 = 0.5 + 0.06 * np.cos(12 * np.pi * t3 / 1.5) * np.exp(-2 * t3)

    # Combine phases
    x = np.concatenate([x1, x2, x3])
    y = np.concatenate([y1, y2, y3])

    # Add realistic surgical tremor (high-frequency, low-amplitude)
    tremor_freq = 8  # Hz
    tremor_amplitude = 0.005
    tremor_x = tremor_amplitude * np.sin(2 * np.pi * tremor_freq * t)
    tremor_y = tremor_amplitude * np.cos(2 * np.pi * tremor_freq * t + np.pi/4)

    # Add measurement noise
    noise_std = 0.003
    noise_x = np.random.normal(0, noise_std, n_frames)
    noise_y = np.random.normal(0, noise_std, n_frames)

    # Combine all effects
    x_final = np.clip(x + tremor_x + noise_x, 0.0, 1.0)
    y_final = np.clip(y + tremor_y + noise_y, 0.0, 1.0)

    # Create trajectory array with timestamps
    trajectory = np.column_stack([x_final, y_final, t])

    return trajectory


def create_training_sequences(trajectory, sequence_length=10, prediction_horizon=15):
    """
    Convert trajectory into training sequences.

    Args:
        trajectory: (N, 3) array [x, y, timestamp]
        sequence_length: Input sequence length
        prediction_horizon: Frames to predict ahead

    Returns:
        inputs: (M, sequence_length, 2) input sequences
        targets: (M, 2) target positions
    """
    positions = trajectory[:, :2]  # x, y coordinates
    n_samples = len(positions) - sequence_length - prediction_horizon + 1

    inputs = []
    targets = []

    for i in range(n_samples):
        # Input sequence
        input_seq = positions[i:i + sequence_length]
        inputs.append(input_seq)

        # Target: position at prediction_horizon frames ahead
        target_pos = positions[i + sequence_length + prediction_horizon - 1]
        targets.append(target_pos)

    return np.array(inputs), np.array(targets)


def generate_dataset(n_trajectories=100, output_dir="data/processed"):
    """
    Generate a complete training dataset.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    all_inputs = []
    all_targets = []

    print(f"Generating {n_trajectories} synthetic cholecystectomy trajectories...")

    for i in range(n_trajectories):
        # Vary trajectory parameters for diversity
        duration = np.random.uniform(8, 15)  # 8-15 seconds
        seed = i * 42  # Deterministic but varied

        trajectory = generate_cholec_trajectory(
            duration_seconds=duration,
            fps=30,
            seed=seed
        )

        inputs, targets = create_training_sequences(trajectory)

        all_inputs.extend(inputs)
        all_targets.extend(targets)

        if (i + 1) % 20 == 0:
            print(f"Generated {i + 1}/{n_trajectories} trajectories")

    # Convert to numpy arrays
    inputs_array = np.array(all_inputs, dtype=np.float32)
    targets_array = np.array(all_targets, dtype=np.float32)

    print(f"Total training samples: {len(inputs_array)}")
    print(f"Input shape: {inputs_array.shape}")
    print(f"Target shape: {targets_array.shape}")

    # Split into train/test (80/20)
    n_train = int(0.8 * len(inputs_array))
    indices = np.random.permutation(len(inputs_array))

    train_indices = indices[:n_train]
    test_indices = indices[n_train:]

    # Save datasets
    np.save(f"{output_dir}/train_inputs.npy", inputs_array[train_indices])
    np.save(f"{output_dir}/train_targets.npy", targets_array[train_indices])
    np.save(f"{output_dir}/test_inputs.npy", inputs_array[test_indices])
    np.save(f"{output_dir}/test_targets.npy", targets_array[test_indices])

    print("Dataset saved successfully!")
    print(f"Train samples: {len(train_indices)}")
    print(f"Test samples: {len(test_indices)}")

    return inputs_array.shape, targets_array.shape


def plot_sample_trajectory():
    """Generate and plot a sample trajectory."""
    trajectory = generate_cholec_trajectory(duration_seconds=10, fps=30, seed=42)

    plt.figure(figsize=(12, 8))

    # Plot trajectory
    plt.subplot(2, 2, 1)
    plt.plot(trajectory[:, 0], trajectory[:, 1], 'b-', alpha=0.7, linewidth=2)
    plt.scatter(trajectory[0, 0], trajectory[0, 1], c='green', s=100, label='Start')
    plt.scatter(trajectory[-1, 0], trajectory[-1, 1], c='red', s=100, label='End')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Synthetic Cholecystectomy Trajectory')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot position over time
    plt.subplot(2, 2, 2)
    plt.plot(trajectory[:, 2], trajectory[:, 0], 'r-', label='X position', alpha=0.8)
    plt.plot(trajectory[:, 2], trajectory[:, 1], 'b-', label='Y position', alpha=0.8)
    plt.xlabel('Time (s)')
    plt.ylabel('Position')
    plt.title('Position vs Time')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot velocity
    dt = np.mean(np.diff(trajectory[:, 2]))
    vx = np.gradient(trajectory[:, 0], dt)
    vy = np.gradient(trajectory[:, 1], dt)
    speed = np.sqrt(vx**2 + vy**2)

    plt.subplot(2, 2, 3)
    plt.plot(trajectory[:, 2], speed, 'g-', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Speed')
    plt.title('Instrument Speed')
    plt.grid(True, alpha=0.3)

    # Plot acceleration (jerk)
    ax = np.gradient(vx, dt)
    ay = np.gradient(vy, dt)
    jerk = np.sqrt(ax**2 + ay**2)

    plt.subplot(2, 2, 4)
    plt.plot(trajectory[:, 2], jerk, 'm-', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Jerk')
    plt.title('Instrument Jerk (Movement Smoothness)')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('sample_trajectory.png', dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic cholecystectomy training data")
    parser.add_argument("--n-trajectories", type=int, default=50, help="Number of trajectories to generate")
    parser.add_argument("--output-dir", type=str, default="data/processed", help="Output directory")
    parser.add_argument("--plot-sample", action="store_true", help="Plot a sample trajectory")

    args = parser.parse_args()

    if args.plot_sample:
        plot_sample_trajectory()
    else:
        input_shape, target_shape = generate_dataset(
            n_trajectories=args.n_trajectories,
            output_dir=args.output_dir
        )
        print("\nDataset generation complete!")
        print(f"Input shape: {input_shape}")
        print(f"Target shape: {target_shape}")