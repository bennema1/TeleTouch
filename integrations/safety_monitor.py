"""
Overshoot Safety Monitor for TeleTouch.
Analyzes surgical simulation screenshots to determine if prediction errors are safe.
Can send warnings to LiveKit room and return results for demo display.
"""
import sys
from pathlib import Path
from typing import Literal, Optional

# Add project root for config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config


def check_safety(image_path: str | Path, error_pixels: float) -> Literal["SAFE", "UNSAFE"]:
    """
    Check if a prediction error is safe using Overshoot vision AI.
    
    Args:
        image_path: Path to screenshot/image of the surgical simulation
        error_pixels: Current prediction error in pixels
        
    Returns:
        "SAFE" or "UNSAFE"
    """
    api_key = getattr(config, "OVERSHOOT_API_KEY", None)
    api_url = getattr(config, "OVERSHOOT_API_URL", "https://api.overshoot.ai/v1/analyze")
    
    if not api_key or api_key == "":
        # Fallback: use rule-based safety check if no API key
        print("WARNING: OVERSHOOT_API_KEY not set. Using fallback rule-based check.")
        return _fallback_safety_check(error_pixels)
    
    try:
        import requests
        
        # Read image file
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        with open(image_path, "rb") as f:
            image_data = f.read()
        
        # Prepare prompt
        prompt = (
            f"This is a surgical simulation display. The white dot shows actual instrument position. "
            f"The green dot shows AI prediction. They are currently {error_pixels} pixels apart.\n\n"
            f"Question: Is this prediction error SAFE or UNSAFE for surgical operations?\n\n"
            f"Context: Errors above 20 pixels could cause tissue damage. Errors below 15 pixels are considered safe.\n\n"
            f"Respond with only one word: SAFE or UNSAFE"
        )
        
        # Send request to Overshoot API
        # Note: Adjust this based on actual Overshoot API format
        response = requests.post(
            api_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "image": image_data.hex(),  # or base64, depending on API
                "prompt": prompt,
                "task": "safety_classification"
            },
            timeout=30
        )
        response.raise_for_status()
        
        result = response.json()
        
        # Extract SAFE/UNSAFE from response
        # Adjust this based on actual API response format
        answer = result.get("answer", "").upper().strip()
        if "SAFE" in answer:
            return "SAFE"
        elif "UNSAFE" in answer:
            return "UNSAFE"
        else:
            # Fallback if response format is unexpected
            print(f"Unexpected API response: {result}. Using fallback check.")
            return _fallback_safety_check(error_pixels)
            
    except ImportError:
        print("ERROR: requests package not installed. Install with: pip install requests")
        return _fallback_safety_check(error_pixels)
    except Exception as e:
        print(f"ERROR: Overshoot API call failed: {e}")
        print("Falling back to rule-based safety check.")
        return _fallback_safety_check(error_pixels)


def _fallback_safety_check(error_pixels: float) -> Literal["SAFE", "UNSAFE"]:
    """
    Fallback rule-based safety check when Overshoot API is unavailable.
    
    Rules:
    - Errors <= 15 pixels: SAFE
    - Errors > 20 pixels: UNSAFE
    - Errors 15-20 pixels: UNSAFE (conservative)
    """
    if error_pixels <= 15:
        return "SAFE"
    else:
        return "UNSAFE"


def check_safety_with_warning(
    image_path: str | Path,
    error_pixels: float,
    room: Optional[object] = None,
    send_to_livekit: bool = True
) -> dict:
    """
    Check safety and send warning to LiveKit if UNSAFE.
    
    Args:
        image_path: Path to screenshot/image of the surgical simulation
        error_pixels: Current prediction error in pixels
        room: Optional LiveKit Room object to send warning message to
        send_to_livekit: Whether to send warning to LiveKit (default: True)
        
    Returns:
        dict with keys:
            - "safety": "SAFE" or "UNSAFE"
            - "warning_sent": bool (whether warning was sent to LiveKit)
            - "message": str (warning message if UNSAFE)
    """
    safety_result = check_safety(image_path, error_pixels)
    
    result = {
        "safety": safety_result,
        "warning_sent": False,
        "message": ""
    }
    
    if safety_result == "UNSAFE":
        warning_msg = f"safety_warning:UNSAFE:error:{error_pixels}"
        result["message"] = f"UNSAFE: Prediction error {error_pixels} pixels exceeds safety threshold"
        
        # Send warning to LiveKit room if provided
        if send_to_livekit and room is not None:
            try:
                # Try to send data message to room
                if hasattr(room, "local_participant") and hasattr(room.local_participant, "publish_data"):
                    import asyncio
                    if asyncio.iscoroutinefunction(room.local_participant.publish_data):
                        # If we're in an async context, create a task
                        try:
                            loop = asyncio.get_running_loop()
                            loop.create_task(room.local_participant.publish_data(warning_msg.encode("utf-8")))
                            result["warning_sent"] = True
                        except RuntimeError:
                            # No running loop, try sync
                            asyncio.run(room.local_participant.publish_data(warning_msg.encode("utf-8")))
                            result["warning_sent"] = True
                    else:
                        # Sync call
                        room.local_participant.publish_data(warning_msg.encode("utf-8"))
                        result["warning_sent"] = True
            except Exception as e:
                print(f"Failed to send warning to LiveKit: {e}")
    
    return result


async def check_safety_with_warning_async(
    image_path: str | Path,
    error_pixels: float,
    room: Optional[object] = None,
    send_to_livekit: bool = True
) -> dict:
    """
    Async version of check_safety_with_warning.
    Use this when calling from async code (like in the demo).
    """
    safety_result = check_safety(image_path, error_pixels)
    
    result = {
        "safety": safety_result,
        "warning_sent": False,
        "message": ""
    }
    
    if safety_result == "UNSAFE":
        warning_msg = f"safety_warning:UNSAFE:error:{error_pixels}"
        result["message"] = f"UNSAFE: Prediction error {error_pixels} pixels exceeds safety threshold"
        
        # Send warning to LiveKit room if provided
        if send_to_livekit and room is not None:
            try:
                if hasattr(room, "local_participant") and hasattr(room.local_participant, "publish_data"):
                    await room.local_participant.publish_data(warning_msg.encode("utf-8"))
                    result["warning_sent"] = True
            except Exception as e:
                print(f"Failed to send warning to LiveKit: {e}")
    
    return result


if __name__ == "__main__":
    # Test the safety monitor
    import tempfile
    
    print("Testing safety monitor...")
    print("(Note: This requires OVERSHOOT_API_KEY in config.py and a valid image)")
    
    # Test with fallback (no API key)
    test_error = 12.5
    result = check_safety("dummy_path.png", test_error)
    print(f"Test error: {test_error} pixels -> {result}")
    
    test_error = 25.0
    result = check_safety("dummy_path.png", test_error)
    print(f"Test error: {test_error} pixels -> {result}")
