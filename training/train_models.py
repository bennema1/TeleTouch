import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import argparse
import json
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm


class SurgicalTrajectoryDataset(Dataset):
    """Dataset for surgical trajectory prediction."""

    def __init__(self, inputs_path, targets_path):
        self.inputs = np.load(inputs_path)   # (N, seq_len, 2)
        self.targets = np.load(targets_path) # (N, 2)

        print(f"Loaded dataset: {len(self)} samples")
        print(f"Input shape: {self.inputs.shape}")
        print(f"Target shape: {self.targets.shape}")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return {
            'input': torch.tensor(self.inputs[idx], dtype=torch.float32),
            'target': torch.tensor(self.targets[idx], dtype=torch.float32)
        }


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in tqdm(dataloader, desc="Training"):
        inputs = batch['input'].to(device)
        targets = batch['target'].to(device)

        optimizer.zero_grad()

        # Forward pass
        predictions, uncertainties = model(inputs)

        # Loss (MSE between prediction and target)
        loss = criterion(predictions, targets)

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch."""
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)

            predictions, uncertainties = model(inputs)

            # Loss
            loss = criterion(predictions, targets)
            total_loss += loss.item()

            # Mean Absolute Error
            mae = torch.mean(torch.abs(predictions - targets)).item()
            total_mae += mae

            num_batches += 1

    avg_loss = total_loss / num_batches
    avg_mae = total_mae / num_batches

    return avg_loss, avg_mae


def plot_training_history(train_losses, val_losses, val_maes, save_path):
    """Plot training history."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss plot
    axes[0].plot(train_losses, label='Training Loss')
    axes[0].plot(val_losses, label='Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('MSE Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)

    # MAE plot
    axes[1].plot(val_maes, label='Validation MAE', color='orange')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Mean Absolute Error')
    axes[1].set_title('Prediction Accuracy (MAE)')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_model_checkpoint(model, optimizer, epoch, loss, save_path):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'timestamp': datetime.now().isoformat()
    }

    torch.save(checkpoint, save_path)
    print(f"Model saved to {save_path}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train surgical trajectory prediction model")
    parser.add_argument('--data-dir', type=str, default='data/processed',
                       help='Directory containing processed data')
    parser.add_argument('--model-version', type=str, default='v1', choices=['v1', 'v2'],
                       help='Model version to train (v1=standard LSTM, v2=with convolution)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Training batch size')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--hidden-size', type=int, default=128,
                       help='LSTM hidden size')
    parser.add_argument('--save-dir', type=str, default='checkpoints',
                       help='Directory to save model checkpoints')
    parser.add_argument('--experiment-name', type=str, default=None,
                       help='Name for this training experiment')

    args = parser.parse_args()

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)

    # Experiment name
    if args.experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"surgical_lstm_{args.model_version}_{timestamp}"
    else:
        experiment_name = args.experiment_name

    experiment_dir = save_dir / experiment_name
    experiment_dir.mkdir(exist_ok=True)

    # Load cholecystectomy training data
    print("Loading cholecystectomy training data...")

    # Use the specialized factory
    from data.dataset_factory import create_cholecystectomy_training_setup

    train_loader, test_loader, dataset_factory = create_cholecystectomy_training_setup(
        pygame_file=None,  # We'll load from processed numpy files
        demo_data_file=None,
        batch_size=args.batch_size,
        sequence_length=10,
        prediction_horizon=30
    )

    # For compatibility with existing code, extract dataset sizes
    train_size = len(train_loader.dataset)
    test_size = len(test_loader.dataset)

    # Create cholecystectomy-specialized model
    print(f"Creating {args.model_version} cholecystectomy-specialized model...")
    from models.surgical_lstm import create_cholecystectomy_lstm
    model = create_cholecystectomy_lstm(args.model_version)
    model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # Training loop
    print("Starting training...")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print("-" * 50)

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    val_maes = []

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        train_losses.append(train_loss)

        # Validate
        val_loss, val_mae = validate_epoch(model, test_loader, criterion, device)
        val_losses.append(val_loss)
        val_maes.append(val_mae)

        print("4.4f"
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model_checkpoint(
                model, optimizer, epoch, val_loss,
                experiment_dir / 'best_model.pth'
            )

        # Early stopping check
        if epoch > 10 and val_loss > max(val_losses[-5:]):
            print("Early stopping triggered")
            break

    # Save final model
    save_model_checkpoint(
        model, optimizer, args.epochs-1, val_losses[-1],
        experiment_dir / 'final_model.pth'
    )

    # Plot training history
    plot_training_history(
        train_losses, val_losses, val_maes,
        experiment_dir / 'training_history.png'
    )

    # Save training results
    results = {
        'experiment_name': experiment_name,
        'model_version': args.model_version,
        'final_train_loss': train_losses[-1],
        'final_val_loss': val_losses[-1],
        'final_val_mae': val_maes[-1],
        'best_val_loss': best_val_loss,
        'epochs_completed': len(train_losses),
        'training_samples': len(train_dataset),
        'test_samples': len(test_dataset),
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'hidden_size': args.hidden_size,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_maes': val_maes
    }

    with open(experiment_dir / 'training_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("
🎉 Training complete!"    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Final validation MAE: {val_maes[-1]:.6f}")
    print(f"Results saved to: {experiment_dir}")

    # Convert MAE to pixels for context
    pixel_mae = val_maes[-1] * 1920  # Assuming 1920px width
    print(".1f"
if __name__ == "__main__":
    main()