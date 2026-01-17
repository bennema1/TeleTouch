"""
Test the demo interface to verify it works correctly.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from integrations.demo_interface import (
    connect_to_livekit,
    announce,
    check_safety,
    disconnect
)


def test_interface():
    """Test all interface functions."""
    print("=" * 60)
    print("Testing Demo Interface")
    print("=" * 60)
    print()
    
    # Test 1: Connect
    print("Test 1: Connecting to LiveKit...")
    if connect_to_livekit():
        print("[OK] Connected successfully")
    else:
        print("[FAIL] Connection failed")
        return
    
    print()
    
    # Test 2: Announce
    print("Test 2: Sending announcement...")
    if announce("prediction_accuracy:94"):
        print("[OK] Message sent")
    else:
        print("[FAIL] Failed to send message")
    
    time.sleep(2)
    print()
    
    # Test 3: Safety check
    print("Test 3: Running safety check...")
    result = check_safety(error_pixels=25.0)
    print(f"Result: {result}")
    if result.get("checked"):
        print(f"[OK] Safety check completed: {result['safety']}")
    else:
        print("[INFO] Safety check throttled (this is normal)")
    
    time.sleep(2)
    print()
    
    # Test 4: Another announcement
    print("Test 4: Sending warning...")
    if announce("warning"):
        print("[OK] Warning sent")
    
    time.sleep(2)
    print()
    
    # Test 5: Disconnect
    print("Test 5: Disconnecting...")
    disconnect()
    print("[OK] Disconnected")
    
    print()
    print("=" * 60)
    print("All tests complete!")
    print("=" * 60)
    print()
    print("If the Surgical Assistant is running, you should have heard:")
    print("  - 'Prediction accuracy high. System operating normally.'")
    print("  - 'Warning: high prediction variance. Manual override suggested.'")


if __name__ == "__main__":
    print("Make sure the Surgical Assistant is running:")
    print("  python integrations/surgical_assistant.py connect --room surgery-demo")
    print()
    print("Starting tests in 2 seconds...")
    time.sleep(2)
    test_interface()
