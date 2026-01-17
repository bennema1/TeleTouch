"""
Test safety check with actual demo screenshots.
Calculates error between white (actual) and green (predicted) dots.
"""
import sys
import math
from pathlib import Path
from typing import Tuple, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from integrations.safety_monitor import check_safety, check_safety_with_warning


def calculate_error_from_positions(
    white_dot_pos: Tuple[float, float],
    green_dot_pos: Tuple[float, float]
) -> float:
    """
    Calculate pixel error between white dot (actual) and green dot (predicted).
    
    Args:
        white_dot_pos: (x, y) position of white dot (actual position)
        green_dot_pos: (x, y) position of green dot (predicted position)
        
    Returns:
        Error distance in pixels
    """
    dx = green_dot_pos[0] - white_dot_pos[0]
    dy = green_dot_pos[1] - white_dot_pos[1]
    return math.sqrt(dx * dx + dy * dy)


def take_screenshot(output_path: str = "demo_screenshot.png") -> Optional[str]:
    """
    Take a screenshot of the demo window.
    Returns path to screenshot file, or None if failed.
    """
    try:
        import pyautogui
        screenshot = pyautogui.screenshot()
        screenshot.save(output_path)
        print(f"[OK] Screenshot saved to: {output_path}")
        return output_path
    except ImportError:
        print("WARNING: pyautogui not installed. Cannot take automatic screenshot.")
        print("Please take a screenshot manually and save it as 'demo_screenshot.png'")
        return None
    except Exception as e:
        print(f"WARNING: Screenshot failed: {e}")
        print("Please take a screenshot manually and save it as 'demo_screenshot.png'")
        return None


def test_safety_with_values():
    """Test safety check with different error values."""
    print("=" * 60)
    print("Testing Safety Check with Different Error Values")
    print("=" * 60)
    print()
    
    # Test cases
    test_cases = [
        (10.0, "SAFE"),   # Should be safe
        (15.0, "SAFE"),   # Borderline safe
        (20.0, "UNSAFE"), # Should be unsafe
        (50.0, "UNSAFE"), # Definitely unsafe
    ]
    
    screenshot_path = "demo_screenshot.png"
    
    # Check if screenshot exists
    if not Path(screenshot_path).exists():
        print(f"WARNING: {screenshot_path} not found.")
        print("Attempting to take screenshot...")
        screenshot_path = take_screenshot(screenshot_path)
        if not screenshot_path or not Path(screenshot_path).exists():
            print("Using dummy path for testing (will use fallback check)")
            screenshot_path = "dummy_screenshot.png"
    
    print(f"Using screenshot: {screenshot_path}")
    print()
    
    for error_pixels, expected in test_cases:
        print(f"Test: {error_pixels} pixels (expected: {expected})")
        result = check_safety(screenshot_path, error_pixels)
        
        status = "[OK]" if result == expected else "[FAIL]"
        print(f"  Result: {result} {status}")
        
        if result != expected:
            print(f"  WARNING: Expected {expected}, got {result}")
        print()


def test_with_manual_positions():
    """
    Test safety check by manually specifying dot positions.
    Useful if you can see the demo and know the coordinates.
    """
    print("=" * 60)
    print("Manual Position Test")
    print("=" * 60)
    print()
    print("If you can see the demo, you can manually enter dot positions.")
    print("Or use this to test with known coordinates.")
    print()
    
    try:
        # Example positions (adjust based on your demo)
        print("Enter white dot (actual) position:")
        white_x = float(input("  X coordinate: ") or "100")
        white_y = float(input("  Y coordinate: ") or "100")
        
        print("Enter green dot (predicted) position:")
        green_x = float(input("  X coordinate: ") or "110")
        green_y = float(input("  Y coordinate: ") or "110")
        
        error = calculate_error_from_positions(
            (white_x, white_y),
            (green_x, green_y)
        )
        
        print(f"\nCalculated error: {error:.2f} pixels")
        
        screenshot_path = "demo_screenshot.png"
        if not Path(screenshot_path).exists():
            screenshot_path = take_screenshot(screenshot_path)
            if not screenshot_path or not Path(screenshot_path).exists():
                screenshot_path = "dummy_screenshot.png"
        
        result = check_safety(screenshot_path, error)
        print(f"Safety check result: {result}")
        
        if result == "UNSAFE":
            print("[WARNING] Unsafe prediction error detected!")
        else:
            print("[OK] Safe prediction error")
            
    except ValueError:
        print("Invalid input. Using default test values.")
        error = calculate_error_from_positions((100, 100), (110, 110))
        print(f"Example error: {error:.2f} pixels")
        result = check_safety("dummy_screenshot.png", error)
        print(f"Safety check result: {result}")


def test_with_demo_integration():
    """
    Test that shows how Person C would integrate this.
    Simulates getting error from demo and checking safety.
    """
    print("=" * 60)
    print("Demo Integration Test")
    print("=" * 60)
    print()
    print("This simulates how Person C's demo would use the safety check.")
    print()
    
    # Simulate getting error from demo
    # In real demo, this would come from position_history.py or predictor.py
    white_pos = (100, 100)  # Actual position
    green_pos = (110, 110)  # Predicted position
    
    error = calculate_error_from_positions(white_pos, green_pos)
    print(f"Demo calculated error: {error:.2f} pixels")
    
    # Take screenshot
    screenshot_path = take_screenshot("demo_screenshot.png")
    if not screenshot_path or not Path(screenshot_path).exists():
        screenshot_path = "dummy_screenshot.png"
        print("Using dummy screenshot for testing")
    
    # Check safety
    result = check_safety(screenshot_path, error)
    print(f"Safety check: {result}")
    
    if result == "UNSAFE":
        print("\n⚠️  UNSAFE - Demo should:")
        print("  1. Display warning overlay")
        print("  2. Send warning to LiveKit")
        print("  3. Agent will announce warning")
    else:
        print("\n✓ SAFE - System operating normally")
    
    return result, error


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test safety check with screenshots")
    parser.add_argument(
        "--mode",
        choices=["values", "manual", "demo", "all"],
        default="all",
        help="Test mode"
    )
    parser.add_argument(
        "--screenshot",
        type=str,
        help="Path to screenshot file (optional)"
    )
    
    args = parser.parse_args()
    
    if args.screenshot:
        # Use provided screenshot
        screenshot_path = args.screenshot
        if not Path(screenshot_path).exists():
            print(f"ERROR: Screenshot not found: {screenshot_path}")
            sys.exit(1)
    
    if args.mode == "values" or args.mode == "all":
        print("\n" + "="*60)
        test_safety_with_values()
    
    if args.mode == "manual" or args.mode == "all":
        print("\n" + "="*60)
        test_with_manual_positions()
    
    if args.mode == "demo" or args.mode == "all":
        print("\n" + "="*60)
        test_with_demo_integration()
    
    print("\n" + "=" * 60)
    print("Testing complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Take a screenshot of Person C's demo")
    print("2. Calculate error from white/green dot positions")
    print("3. Run: python integrations/test_safety_with_screenshot.py")
    print("4. Verify SAFE/UNSAFE responses match expectations")
