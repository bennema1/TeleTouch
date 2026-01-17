"""
Clean integration interface for Person C's demo.
Simple functions to add voice and safety monitoring.
"""
import sys
import asyncio
import time
from pathlib import Path
from typing import Optional, Dict, Any
import threading

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from livekit.api import AccessToken, VideoGrants
from livekit.rtc import Room
from integrations.safety_monitor import check_safety_with_warning_async

from config import LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET


class DemoIntegration:
    """Integration manager for voice and safety features."""
    
    def __init__(self):
        self.room: Optional[Room] = None
        self.connected = False
        self.room_name = "surgery-demo"
        self.identity = "demo-client"
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._last_safety_check = 0.0
        self._safety_check_interval = 5.0  # Check every 5 seconds
        
    def _run_async_loop(self):
        """Run async event loop in background thread."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()
    
    def _run_async(self, coro):
        """Run async function in the background thread."""
        if self._loop is None:
            self._thread = threading.Thread(target=self._run_async_loop, daemon=True)
            self._thread.start()
            # Wait for loop to be ready
            while self._loop is None:
                time.sleep(0.01)
        
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=10.0)
    
    def connect_to_livekit(self) -> bool:
        """
        Connect to LiveKit room.
        Call this once at demo startup.
        
        Returns:
            True if connected successfully, False otherwise
        """
        if self.connected:
            print("[Demo Integration] Already connected to LiveKit")
            return True
        
        try:
            token = (
                AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
                .with_identity(self.identity)
                .with_name(self.identity)
                .with_grants(VideoGrants(
                    room_join=True,
                    room=self.room_name,
                    can_publish_data=True
                ))
                .to_jwt()
            )
            
            self.room = Room()
            self._run_async(self.room.connect(LIVEKIT_URL, token))
            self.connected = True
            print(f"[Demo Integration] Connected to LiveKit room: {self.room_name}")
            return True
            
        except Exception as e:
            print(f"[Demo Integration] Failed to connect to LiveKit: {e}")
            self.connected = False
            return False
    
    def announce(self, message: str) -> bool:
        """
        Send a message to trigger voice narration.
        
        Args:
            message: Message to send (e.g., "prediction_accuracy:94" or "warning")
            
        Returns:
            True if sent successfully, False otherwise
        """
        if not self.connected or self.room is None:
            print("[Demo Integration] Not connected to LiveKit. Call connect_to_livekit() first.")
            return False
        
        try:
            async def send():
                await self.room.local_participant.publish_data(
                    message.encode("utf-8")
                )
            
            self._run_async(send())
            print(f"[Demo Integration] Sent message: {message}")
            return True
            
        except Exception as e:
            print(f"[Demo Integration] Failed to send message: {e}")
            return False
    
    def check_safety(self, 
                    screenshot_path: Optional[str] = None,
                    error_pixels: float = 0.0,
                    frame: Optional[Any] = None) -> Dict[str, Any]:
        """
        Check if prediction error is safe.
        Automatically checks every 5 seconds (throttled).
        
        Args:
            screenshot_path: Path to screenshot (optional, will save frame if provided)
            error_pixels: Current prediction error in pixels
            frame: OpenCV frame (optional, will save if provided)
            
        Returns:
            dict with keys:
                - "safety": "SAFE" or "UNSAFE"
                - "message": Warning message if UNSAFE
                - "warning_sent": Whether warning was sent to LiveKit
                - "checked": Whether check was actually performed (throttled)
        """
        current_time = time.time()
        
        # Throttle: only check every 5 seconds
        if current_time - self._last_safety_check < self._safety_check_interval:
            return {
                "safety": "UNKNOWN",
                "message": "",
                "warning_sent": False,
                "checked": False
            }
        
        self._last_safety_check = current_time
        
        # Save frame if provided
        if frame is not None:
            if screenshot_path is None:
                screenshot_path = "temp_safety_check.png"
            try:
                import cv2
                cv2.imwrite(screenshot_path, frame)
            except Exception as e:
                print(f"[Demo Integration] Failed to save frame: {e}")
                screenshot_path = None
        
        if screenshot_path is None or not Path(screenshot_path).exists():
            screenshot_path = "dummy_screenshot.png"
        
        try:
            result = self._run_async(check_safety_with_warning_async(
                image_path=screenshot_path,
                error_pixels=error_pixels,
                room=self.room if self.connected else None,
                send_to_livekit=self.connected
            ))
            
            result["checked"] = True
            
            if result["safety"] == "UNSAFE":
                print(f"[Demo Integration] UNSAFE detected: {result['message']}")
                # Also send warning message to trigger voice
                self.announce("safety_warning:UNSAFE:error:" + str(error_pixels))
            
            return result
            
        except Exception as e:
            print(f"[Demo Integration] Safety check failed: {e}")
            return {
                "safety": "UNKNOWN",
                "message": f"Check failed: {e}",
                "warning_sent": False,
                "checked": True
            }
    
    def disconnect(self):
        """Disconnect from LiveKit and cleanup."""
        if self.connected and self.room is not None:
            try:
                self._run_async(self.room.disconnect())
                print("[Demo Integration] Disconnected from LiveKit")
            except Exception as e:
                print(f"[Demo Integration] Error disconnecting: {e}")
        
        self.connected = False
        self.room = None
        
        # Note: Event loop cleanup is handled automatically when thread exits


# Global instance for easy import
_integration = DemoIntegration()


# Convenience functions for Person C
def connect_to_livekit() -> bool:
    """Connect to LiveKit room. Call once at demo startup."""
    return _integration.connect_to_livekit()


def announce(message: str) -> bool:
    """
    Send message to trigger voice narration.
    
    Examples:
        announce("prediction_accuracy:94")
        announce("warning")
        announce("stabilized")
    """
    return _integration.announce(message)


def check_safety(screenshot_path: Optional[str] = None,
                error_pixels: float = 0.0,
                frame: Optional[Any] = None) -> Dict[str, Any]:
    """
    Check if prediction error is safe (throttled to every 5 seconds).
    
    Args:
        screenshot_path: Path to screenshot (optional)
        error_pixels: Current prediction error in pixels
        frame: OpenCV frame (optional, will save if provided)
        
    Returns:
        dict with safety result and warning info
    """
    return _integration.check_safety(screenshot_path, error_pixels, frame)


def disconnect():
    """Disconnect from LiveKit. Call at demo shutdown."""
    _integration.disconnect()
