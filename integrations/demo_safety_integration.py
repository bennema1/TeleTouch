"""
Example integration for Person C's demo to use safety monitoring.
This shows how to call the safety monitor and handle warnings.
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from livekit.api import AccessToken, VideoGrants
from livekit.rtc import Room
from integrations.safety_monitor import check_safety_with_warning_async

from config import LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET


async def check_and_warn(image_path: str, error_pixels: float, room: Room) -> dict:
    """
    Check safety and send warning if UNSAFE.
    Call this from your demo when you have a new prediction error.
    
    Args:
        image_path: Path to screenshot of current demo state
        error_pixels: Current prediction error in pixels
        room: LiveKit Room object (connected to surgery-demo)
        
    Returns:
        dict with safety result and warning info
    """
    result = await check_safety_with_warning_async(
        image_path=image_path,
        error_pixels=error_pixels,
        room=room,
        send_to_livekit=True
    )
    
    # Return result for demo to display overlay
    return result


async def example_usage():
    """
    Example of how Person C would use this in their demo.
    """
    # Connect to LiveKit room
    room_name = "surgery-demo"
    identity = "demo-client"
    
    token = (
        AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
        .with_identity(identity)
        .with_name(identity)
        .with_grants(VideoGrants(room_join=True, room=room_name))
        .to_jwt()
    )
    
    room = Room()
    await room.connect(LIVEKIT_URL, token)
    print(f"Connected to {room_name}")
    
    # Example: Check safety for a screenshot
    screenshot_path = "demo_screenshot.png"  # Replace with actual screenshot path
    error = 25.5  # Current prediction error in pixels
    
    result = await check_and_warn(screenshot_path, error, room)
    
    print(f"Safety check result: {result['safety']}")
    if result["safety"] == "UNSAFE":
        print(f"WARNING: {result['message']}")
        print("Display warning overlay in demo UI")
        if result["warning_sent"]:
            print("Warning sent to LiveKit - agent will announce it")
    
    # Demo can use result["safety"] to show/hide warning overlay
    # Demo can use result["message"] to display warning text
    
    await room.disconnect()


if __name__ == "__main__":
    print("Example safety monitoring integration")
    print("This shows how Person C's demo would call the safety monitor")
    # Uncomment to test:
    # asyncio.run(example_usage())
