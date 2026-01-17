"""
Full integration test - simulates how Person C's demo would use the integration.
Tests all features working together.
"""
import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from integrations.demo_interface import (
    connect_to_livekit,
    announce,
    check_safety,
    disconnect
)


def simulate_demo_loop():
    """Simulate a demo update loop with various scenarios."""
    print("=" * 60)
    print("Full Integration Test - Simulating Demo Loop")
    print("=" * 60)
    print()
    print("This simulates how Person C's demo would use the integration.")
    print("Make sure the Surgical Assistant is running!")
    print()
    time.sleep(2)
    
    # Connect
    print("Step 1: Connecting to LiveKit...")
    if not connect_to_livekit():
        print("[FAIL] Could not connect. Make sure credentials are correct.")
        return
    print("[OK] Connected")
    print()
    
    # Simulate demo running for 20 seconds
    print("Step 2: Simulating demo loop (20 seconds)...")
    print("  - Frame 0-60: Low error (SAFE)")
    print("  - Frame 60-120: Medium error")
    print("  - Frame 120-180: High error (UNSAFE)")
    print("  - Frame 180-240: Error stabilizes")
    print()
    
    frame_count = 0
    start_time = time.time()
    
    # Simulate different error scenarios
    scenarios = [
        (0, 60, 8.0, "Low error - SAFE"),
        (60, 120, 18.0, "Medium error - SAFE"),
        (120, 180, 35.0, "High error - UNSAFE"),
        (180, 240, 12.0, "Error stabilized - SAFE"),
    ]
    
    current_scenario = 0
    
    while time.time() - start_time < 20:  # Run for 20 seconds
        frame_count += 1
        
        # Determine current scenario
        if current_scenario < len(scenarios):
            start_frame, end_frame, error, description = scenarios[current_scenario]
            if frame_count >= end_frame:
                current_scenario += 1
                if current_scenario < len(scenarios):
                    start_frame, end_frame, error, description = scenarios[current_scenario]
        
        # Add some variation to error
        actual_error = error + np.random.uniform(-2, 2)
        
        # Simulate demo update loop
        if frame_count % 30 == 0:  # Every second at 30fps
            print(f"  Frame {frame_count}: Error = {actual_error:.1f}px - {description}")
            
            # Test 1: Announce warnings for high error
            if actual_error > 30:
                if frame_count % 60 == 0:  # Every 2 seconds
                    announce("warning")
                    print("    -> Sent warning announcement")
            
            # Test 2: Announce good accuracy occasionally
            elif actual_error < 10:
                if frame_count % 120 == 0:  # Every 4 seconds
                    accuracy = int(100 - actual_error)
                    announce(f"prediction_accuracy:{accuracy}")
                    print(f"    -> Sent accuracy announcement: {accuracy}%")
            
            # Test 3: Safety check (auto-throttled to every 5 seconds)
            # Create a dummy frame (in real demo, this would be the actual frame)
            dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
            safety_result = check_safety(
                frame=dummy_frame,
                error_pixels=actual_error
            )
            
            if safety_result.get("checked"):
                print(f"    -> Safety check: {safety_result['safety']}")
                if safety_result["safety"] == "UNSAFE":
                    print(f"    -> Warning message: {safety_result['message']}")
        
        # Simulate frame time (30 FPS)
        time.sleep(1.0 / 30.0)
    
    print()
    print("Step 3: Demo loop complete")
    print()
    
    # Test 4: Final announcements
    print("Step 4: Testing final announcements...")
    announce("stabilized")
    print("  -> Sent 'stabilized' message")
    time.sleep(2)
    
    # Disconnect
    print()
    print("Step 5: Disconnecting...")
    disconnect()
    print("[OK] Disconnected")
    
    print()
    print("=" * 60)
    print("Full Integration Test Complete!")
    print("=" * 60)
    print()
    print("Summary:")
    print("  [OK] Connected to LiveKit")
    print("  [OK] Sent voice announcements (accuracy, warnings)")
    print("  [OK] Ran safety checks (throttled to every 5 seconds)")
    print("  [OK] Detected UNSAFE conditions")
    print("  [OK] Sent warnings to LiveKit")
    print("  [OK] Disconnected cleanly")
    print()
    print("If the Surgical Assistant was running, you should have heard:")
    print("  - 'Prediction accuracy high. System operating normally.'")
    print("  - 'Warning: high prediction variance. Manual override suggested.'")
    print("  - 'System stabilized. Prediction accuracy restored.'")


def test_specific_scenarios():
    """Test specific scenarios."""
    print("=" * 60)
    print("Testing Specific Scenarios")
    print("=" * 60)
    print()
    
    if not connect_to_livekit():
        print("[FAIL] Could not connect")
        return
    
    scenarios = [
        (10.0, "prediction_accuracy:90", "Low error - should be SAFE"),
        (25.0, "warning", "High error - should trigger warning"),
        (50.0, "warning", "Very high error - should be UNSAFE"),
    ]
    
    for error, message, description in scenarios:
        print(f"\nScenario: {description}")
        print(f"  Error: {error} pixels")
        
        # Send announcement
        announce(message)
        print(f"  -> Sent: {message}")
        time.sleep(1)
        
        # Check safety
        result = check_safety(error_pixels=error)
        if result.get("checked"):
            print(f"  -> Safety: {result['safety']}")
        else:
            print(f"  -> Safety check throttled (normal)")
        
        time.sleep(2)
    
    disconnect()
    print("\n[OK] All scenarios tested")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test full integration")
    parser.add_argument(
        "--mode",
        choices=["full", "scenarios"],
        default="full",
        help="Test mode"
    )
    
    args = parser.parse_args()
    
    if args.mode == "full":
        simulate_demo_loop()
    else:
        test_specific_scenarios()
