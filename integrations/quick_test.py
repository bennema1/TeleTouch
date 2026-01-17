"""Quick test of all integration functions."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from integrations.demo_interface import (
    connect_to_livekit,
    announce,
    check_safety,
    disconnect
)

print("=== Quick Function Test ===")
print()

print("1. Testing connect...")
result = connect_to_livekit()
print(f"   Result: {result}")
print()

print("2. Testing announce...")
announce("prediction_accuracy:95")
print("   Message sent")
print()

print("3. Testing safety check...")
safety = check_safety(error_pixels=15.0)
print(f"   Safety: {safety['safety']}")
print()

print("4. Disconnecting...")
disconnect()
print("   Disconnected")
print()

print("=== All functions work! ===")
