"""
Test safety check using actual error values from the demo.
This script can be integrated with Person C's demo to test safety in real-time.
"""
import sys
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from integrations.safety_monitor import check_safety, check_safety_with_warning_async
from livekit.api import AccessToken, VideoGrants
from livekit.rtc import Room

from config import LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET


async def test_safety_with_demo_error(
    error_pixels: float,
    screenshot_path: str = "demo_screenshot.png",
    send_to_livekit: bool = False
):
    """
    Test safety check with a specific error value from the demo.
    
    Args:
        error_pixels: Error value calculated from demo (white dot vs green dot)
        screenshot_path: Path to screenshot (optional)
        send_to_livekit: Whether to send warning to LiveKit if UNSAFE
    """
    print(f"Testing safety check with error: {error_pixels:.2f} pixels")
    
    # Check if screenshot exists
    if not Path(screenshot_path).exists():
        print(f"WARNING: Screenshot not found: {screenshot_path}")
        print("Using fallback rule-based check (no screenshot needed)")
        screenshot_path = "dummy_screenshot.png"
    
    # Check safety
    result = check_safety(screenshot_path, error_pixels)
    
    print(f"Safety result: {result}")
    
    # If UNSAFE and LiveKit is enabled, send warning
    if result == "UNSAFE" and send_to_livekit:
        print("Sending warning to LiveKit...")
        try:
            room_name = "surgery-demo"
            identity = "safety-tester"
            
            token = (
                AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
                .with_identity(identity)
                .with_name(identity)
                .with_grants(VideoGrants(room_join=True, room=room_name, can_publish_data=True))
                .to_jwt()
            )
            
            room = Room()
            await room.connect(LIVEKIT_URL, token)
            
            warning_result = await check_safety_with_warning_async(
                image_path=screenshot_path,
                error_pixels=error_pixels,
                room=room,
                send_to_livekit=True
            )
            
            if warning_result["warning_sent"]:
                print("[OK] Warning sent to LiveKit - agent will announce it")
            
            await room.disconnect()
        except Exception as e:
            print(f"Failed to send to LiveKit: {e}")
    
    return result


async def test_specific_values():
    """Test with the specific values requested: 10px and 50px."""
    print("=" * 60)
    print("Testing Safety Check with Specific Values")
    print("=" * 60)
    print()
    
    test_cases = [
        (10.0, "SAFE"),
        (50.0, "UNSAFE"),
    ]
    
    for error, expected in test_cases:
        print(f"\nTest {error} pixels (expected: {expected}):")
        result = await test_safety_with_demo_error(error, send_to_livekit=False)
        
        if result == expected:
            print(f"[OK] Result matches expectation: {result}")
        else:
            print(f"[FAIL] Expected {expected}, got {result}")
    
    print("\n" + "=" * 60)
    print("Testing complete!")
    print("=" * 60)


def demo_integration_example():
    """
    Example of how Person C would integrate this into their demo.
    This shows the pattern they should follow.
    """
    print("=" * 60)
    print("Demo Integration Example")
    print("=" * 60)
    print()
    print("In Person C's demo, they would do something like this:")
    print()
    print("```python")
    print("from integrations.safety_monitor import check_safety_with_warning_async")
    print("from livekit.rtc import Room")
    print()
    print("# In the demo's update loop:")
    print("error = renderer.calculate_error(white_pos, green_pos)")
    print()
    print("# Check safety")
    print("result = await check_safety_with_warning_async(")
    print("    image_path='screenshot.png',")
    print("    error_pixels=error,")
    print("    room=livekit_room,")
    print("    send_to_livekit=True")
    print(")")
    print()
    print("# Display warning if UNSAFE")
    print("if result['safety'] == 'UNSAFE':")
    print("    renderer.draw_warning(frame, result['message'])")
    print("```")
    print()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test safety check with demo error values"
    )
    parser.add_argument(
        "--error",
        type=float,
        help="Specific error value to test (in pixels)"
    )
    parser.add_argument(
        "--screenshot",
        type=str,
        default="demo_screenshot.png",
        help="Path to screenshot file"
    )
    parser.add_argument(
        "--livekit",
        action="store_true",
        help="Send warnings to LiveKit if UNSAFE"
    )
    parser.add_argument(
        "--test-values",
        action="store_true",
        help="Test with 10px and 50px values"
    )
    parser.add_argument(
        "--example",
        action="store_true",
        help="Show integration example"
    )
    
    args = parser.parse_args()
    
    if args.example:
        demo_integration_example()
    elif args.test_values:
        asyncio.run(test_specific_values())
    elif args.error is not None:
        asyncio.run(test_safety_with_demo_error(
            args.error,
            args.screenshot,
            args.livekit
        ))
    else:
        print("Testing with default values (10px and 50px)...")
        print("Use --help for more options")
        print()
        asyncio.run(test_specific_values())
