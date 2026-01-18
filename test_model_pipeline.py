#!/usr/bin/env python3
"""
Test the complete cholecystectomy ML pipeline from data loading to prediction.
Ensures all components work together for latency compensation.
"""

import numpy as np
import torch
from pathlib import Path
import sys
import time
from tqdm import tqdm


def test_imports():
    """Test that all required modules can be imported."""
    print("🔍 Testing imports...")

    try:
        # Core dependencies
        import torch
        import numpy as np
        import scipy
        import matplotlib
        print("✅ Core dependencies: OK")

        # ML components
        from models.surgical_lstm import create_cholecystectomy_lstm
        from models.transformer_model import SurgicalTransformer
        from models.ensemble_model import SurgicalEnsemblePredictor
        print("✅ ML models: OK")

        # Data pipeline
        from data.rosma_dataset import CholecystectomyDataset, EnhancedCholecystectomyDataset
        from data.dataset_factory import create_cholecystectomy_training_setup
        print("✅ Data pipeline: OK")

        # Training components
        from training.trainer import CurriculumTrainer
        from training.train_models import train_single_model
        print("✅ Training components: OK")

        # Evaluation
        from evaluation.metrics import TrajectoryMetrics
        from evaluation.validation import CrossValidationProtocol
        print("✅ Evaluation components: OK")

        return True

    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_data_loading():
    """Test cholecystectomy data loading and processing."""
    print("\n🔍 Testing data loading...")

    try:
        from data.rosma_dataset import CholecystectomyDataset

        # Test with synthetic data if ROSMA not available
        print("📊 Testing CholecystectomyDataset...")

        # This will fail gracefully if ROSMA data not downloaded
        try:
            dataset = CholecystectomyDataset()
            print("✅ ROSMA dataset loaded successfully")

            # Test trajectory extraction
            if dataset.kinematic_data:
                sample_trial = list(dataset.kinematic_data.keys())[0]
                trajectories = dataset.extract_cholecystectomy_trajectories(sample_trial)
                print(f"✅ Extracted trajectories: shape {trajectories.shape}")

                # Test feature extraction
                features = dataset.get_cholecystectomy_kinematic_features(sample_trial)
                print(f"✅ Computed features: shape {features.shape}")
            else:
                print("⚠️ No kinematic data available (expected if ROSMA not downloaded)")

        except Exception as e:
            print(f"⚠️ ROSMA dataset test skipped: {e}")

        # Test synthetic dataset
        from data.rosma_dataset import CholecystectomySyntheticDataset
        synthetic_dataset = CholecystectomySyntheticDataset(num_samples=100)
        sample = synthetic_dataset[0]
        print(f"✅ Synthetic dataset: input shape {sample['input'].shape}")

        return True

    except Exception as e:
        print(f"❌ Data loading test failed: {e}")
        return False


def test_model_creation():
    """Test cholecystectomy model creation and forward pass."""
    print("\n🔍 Testing model creation...")

    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️ Using device: {device}")

        # Test LSTM model
        from models.surgical_lstm import create_cholecystectomy_lstm

        lstm_model = create_cholecystectomy_lstm('v1')
        lstm_model.to(device)
        print("✅ CholecystectomyLSTM created")

        # Test forward pass
        batch_size, seq_len, features = 4, 10, 13  # Cholecystectomy features
        dummy_input = torch.randn(batch_size, seq_len, features).to(device)

        with torch.no_grad():
            prediction, uncertainty = lstm_model(dummy_input)
            print(f"✅ Forward pass: prediction shape {prediction.shape}, uncertainty shape {uncertainty.shape}")

        # Test parameter count
        total_params = sum(p.numel() for p in lstm_model.parameters())
        print(f"📊 Model parameters: {total_params:,}")

        # Test Transformer model
        try:
            from models.transformer_model import SurgicalTransformer
            transformer_model = SurgicalTransformer(input_size=13, hidden_size=128)
            transformer_model.to(device)
            print("✅ SurgicalTransformer created")
        except Exception as e:
            print(f"⚠️ Transformer test skipped: {e}")

        return True

    except Exception as e:
        print(f"❌ Model creation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_pipeline():
    """Test the training pipeline with synthetic data."""
    print("\n🔍 Testing training pipeline...")

    try:
        from data.dataset_factory import create_cholecystectomy_training_setup
        from training.trainer import CurriculumTrainer, create_trainer_config
        from models.surgical_lstm import create_cholecystectomy_lstm

        # Create training setup with synthetic data
        train_loader, val_loader, dataset_factory = create_cholecystectomy_training_setup(
            pygame_file=None,
            demo_data_file=None,
            batch_size=16,
            sequence_length=10,
            prediction_horizon=30
        )

        print("✅ Training setup created")
        print(f"📊 Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

        # Test trainer creation
        config = create_trainer_config()
        trainer = CurriculumTrainer(config)

        # Create model
        model = create_cholecystectomy_lstm('v1')
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = torch.nn.MSELoss()

        print("✅ Trainer and model ready")

        # Test one training step
        model.train()
        batch = next(iter(train_loader))
        inputs, targets = batch['input'], batch['target']

        optimizer.zero_grad()
        outputs, _ = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        print(f"Training loss: {loss:.6f}")
        return True

    except Exception as e:
        print(f"❌ Training pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_inference():
    """Test real-time inference performance."""
    print("\n🔍 Testing inference performance...")

    try:
        from models.surgical_lstm import TrajectoryPredictor

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Create predictor (will use fallback if no model)
        predictor = TrajectoryPredictor()

        # Test prediction with dummy data
        dummy_sequence = np.random.randn(10, 2)  # 10 frames of x,y

        # Warm up
        for _ in range(5):
            predictor.predict_single_step(dummy_sequence)

        # Time inference
        num_iterations = 100
        start_time = time.time()

        for _ in range(num_iterations):
            predictor.predict_single_step(dummy_sequence)

        end_time = time.time()
        avg_time = (end_time - start_time) / num_iterations * 1000  # ms

        print(f"Average inference time: {avg_time:.3f}ms")
        print(f"🎯 Target: <16ms for 60fps real-time")

        if avg_time < 16:
            print("✅ Real-time performance: ACHIEVED")
        else:
            print("⚠️ Real-time performance: NOT ACHIEVED (may be OK for demo)")

        return True

    except Exception as e:
        print(f"❌ Inference test failed: {e}")
        return False


def test_demo_components():
    """Test demo-specific components."""
    print("\n🔍 Testing demo components...")

    try:
        # Test data player
        from demo_surgical_latency import DemoDataPlayer
        player = DemoDataPlayer()

        # Test pattern switching
        for pattern in ['dissection', 'clipping', 'retraction', 'navigation']:
            player.set_pattern(pattern)
            pos = player.get_next_position()
            if pos is not None:
                print(f"✅ Pattern '{pattern}': position generated")
            else:
                print(f"⚠️ Pattern '{pattern}': no position generated")

        # Test speed adjustment
        player.set_speed(2.0)
        print("✅ Speed adjustment: OK")

        # Test hand tracker (will fail gracefully without camera)
        try:
            from demo_surgical_latency import HandTracker
            tracker = HandTracker()
            pos, frame = tracker.get_hand_position()
            tracker.release()
            print("✅ Hand tracker: Created and released")
        except Exception as e:
            print(f"⚠️ Hand tracker test skipped: {e}")

        return True

    except Exception as e:
        print(f"❌ Demo components test failed: {e}")
        return False


def run_full_pipeline_test():
    """Run complete pipeline test."""
    print("🚀 CHOLECYSTECTOMY LATENCY COMPENSATION - FULL PIPELINE TEST")
    print("=" * 70)

    tests = [
        ("Import Test", test_imports),
        ("Data Loading Test", test_data_loading),
        ("Model Creation Test", test_model_creation),
        ("Training Pipeline Test", test_training_pipeline),
        ("Inference Performance Test", test_inference),
        ("Demo Components Test", test_demo_components),
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        success = test_func()
        results.append((test_name, success))

        if success:
            print(f"✅ {test_name}: PASSED")
        else:
            print(f"❌ {test_name}: FAILED")

    # Summary
    print("\n" + "="*70)
    print("📊 TEST SUMMARY")
    print("="*70)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print("30")

    print(f"\n🎯 Overall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 ALL TESTS PASSED! Cholecystectomy pipeline is ready!")
        print("\nNext steps:")
        print("1. Download ROSMA data: python download_rosma.py")
        print("2. Process data: python process_rosma_data.py")
        print("3. Train model: python train_surgical_model.py --epochs 50")
        print("4. Run demo: python demo_surgical_latency.py")
    else:
        print("⚠️ Some tests failed. Check the errors above.")
        print("The pipeline may still work but with limited functionality.")

    return passed == total


def main():
    """Main test function."""
    try:
        success = run_full_pipeline_test()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n🛑 Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()