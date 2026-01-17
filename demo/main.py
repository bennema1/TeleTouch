"""
TELE-TOUCH Demo Application
Main entry point for the surgical prediction visualization

This demo shows:
- White cursor: Actual instrument position
- Red cursor: What a lagged robot sees (500ms behind)
- Green cursor: AI prediction (compensating for lag)

Controls:
- SPACE: Pause/Resume
- R: Restart
- Q or ESC: Quit
- S: Screenshot
- 1/2/3: Toggle cursors
- T: Toggle trails
- I: Toggle info panel
"""

import cv2
import numpy as np
import time
from typing import Optional, Tuple
from pathlib import Path

# Import our components
from lag_buffer import LagBuffer
from position_history import PositionHistory
from predictor import create_predictor, PredictorInterface
from overlay_renderer import OverlayRenderer
from synthetic_data import SyntheticDataSource


class TeleTouchDemo:
    """Main demo application class."""
    
    def __init__(self, 
                 width: int = 1280, 
                 height: int = 720,
                 fps: int = 30,
                 latency_ms: int = 500,
                 video_path: Optional[str] = None,
                 model_path: Optional[str] = None):
        """
        Initialize the demo.
        
        Args:
            width: Display width
            height: Display height
            fps: Target frames per second
            latency_ms: Simulated latency in milliseconds
            video_path: Path to surgical video (None = synthetic background)
            model_path: Path to trained AI model (None = dummy predictor)
        """
        self.width = width
        self.height = height
        self.fps = fps
        self.latency_seconds = latency_ms / 1000.0
        self.frame_time = 1.0 / fps
        
        # Video source
        self.video_path = video_path
        self.video_capture = None
        if video_path and Path(video_path).exists():
            self.video_capture = cv2.VideoCapture(video_path)
            # Get actual video dimensions
            self.width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Initialize components
        self.lag_buffer = LagBuffer(delay_seconds=self.latency_seconds)
        self.position_history = PositionHistory(max_length=30)
        self.predictor = create_predictor(model_path)
        self.renderer = OverlayRenderer(self.width, self.height)
        self.data_source = SyntheticDataSource(fps=fps, duration_seconds=120, seed=42)
        
        # State
        self.running = True
        self.paused = False
        self.frame_count = 0
        self.start_time = None
        
        # Toggle states
        self.show_white = True
        self.show_red = True
        self.show_green = True
        self.show_trails = True
        self.show_info = True
        
        # Current positions
        self.white_pos: Optional[Tuple[float, float]] = None
        self.red_pos: Optional[Tuple[float, float]] = None
        self.green_pos: Optional[Tuple[float, float]] = None
        
        # Performance metrics
        self.error_history = []
        self.avg_error = 0.0
        
        print(f"TeleTouchDemo initialized:")
        print(f"  Resolution: {self.width}x{self.height}")
        print(f"  FPS: {self.fps}")
        print(f"  Latency: {latency_ms}ms")
        print(f"  Predictor: {self.predictor.get_name()}")
        print(f"  Video: {'Loaded' if self.video_capture else 'Synthetic'}")
    
    def get_background_frame(self) -> np.ndarray:
        """Get the background frame (video or synthetic)."""
        if self.video_capture is not None:
            ret, frame = self.video_capture.read()
            if not ret:
                # Loop video
                self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.video_capture.read()
            if ret:
                return frame
        
        # Synthetic background (dark with subtle grid)
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        frame[:] = (25, 25, 30)  # Dark blue-gray
        
        # Draw subtle grid
        grid_color = (35, 35, 40)
        grid_spacing = 50
        
        for x in range(0, self.width, grid_spacing):
            cv2.line(frame, (x, 0), (x, self.height), grid_color, 1)
        for y in range(0, self.height, grid_spacing):
            cv2.line(frame, (0, y), (self.width, y), grid_color, 1)
        
        # Add "SYNTHETIC MODE" text
        cv2.putText(frame, "SYNTHETIC DATA MODE", 
                    (self.width - 250, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 60, 70), 1, cv2.LINE_AA)
        
        return frame
    
    def process_frame(self, timestamp: float) -> np.ndarray:
        """
        Process a single frame and return the rendered result.
        
        Args:
            timestamp: Current time in seconds
            
        Returns:
            Rendered frame with all overlays
        """
        # Get background
        frame = self.get_background_frame()
        
        # Get actual position (WHITE cursor)
        self.white_pos = self.data_source.get_current_position()
        
        # Update lag buffer
        self.lag_buffer.push(self.white_pos, timestamp)
        self.lag_buffer.cleanup(timestamp)
        
        # Get lagged position (RED cursor)
        self.red_pos = self.lag_buffer.get_lagged(timestamp)
        
        # Update position history for prediction
        self.position_history.push(self.white_pos)
        
        # Get predicted position (GREEN cursor)
        if self.position_history.is_ready(min_positions=10):
            recent_positions = self.position_history.get_last(10)
            # Predict 15 frames ahead = 500ms at 30fps
            steps_ahead = int(self.latency_seconds * self.fps)
            self.green_pos = self.predictor.predict(recent_positions, steps_ahead)
        else:
            self.green_pos = self.white_pos  # Not enough history yet
        
        # Calculate error
        error = self.renderer.calculate_error(self.white_pos, self.green_pos)
        self.error_history.append(error)
        if len(self.error_history) > 30:
            self.error_history.pop(0)
        self.avg_error = np.mean(self.error_history)
        
        confidence = max(0, min(100, 100 - self.avg_error))
        
        # Update and draw trails - only for visible cursors
        if self.show_trails:
            # Only update trails for cursors that are visible
            self.renderer.update_trails(
                self.white_pos if self.show_white else None,
                self.red_pos if self.show_red else None,
                self.green_pos if self.show_green else None
            )
            self.renderer.draw_trails(frame, 
                                      show_white=self.show_white,
                                      show_red=self.show_red, 
                                      show_green=self.show_green)
        
        # Draw cursors with distinct shapes
        if self.show_white and self.white_pos:
            self.renderer.draw_cursor(frame, self.white_pos, 
                                      self.renderer.WHITE, "ACTUAL",
                                      cursor_type="actual")
        
        if self.show_red and self.red_pos:
            self.renderer.draw_cursor(frame, self.red_pos,
                                      self.renderer.RED, "LAGGED 500ms",
                                      cursor_type="lagged")
        
        if self.show_green and self.green_pos:
            self.renderer.draw_cursor(frame, self.green_pos,
                                      self.renderer.GREEN, "PREDICTED",
                                      cursor_type="predicted")
        
        # Draw info panel
        if self.show_info:
            self.renderer.draw_info_panel(
                frame,
                error_pixels=self.avg_error,
                confidence=confidence,
                predictor_name=self.predictor.get_name(),
                latency_ms=int(self.latency_seconds * 1000)
            )
        
        # Draw legend
        self.renderer.draw_legend(frame)
        
        # Draw warning if error is too high
        if self.avg_error > 40:
            self.renderer.draw_warning(frame, "HIGH PREDICTION ERROR")
        
        # Draw pause indicator
        if self.paused:
            cv2.putText(frame, "PAUSED", 
                        (self.width // 2 - 80, self.height // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)
        
        return frame
    
    def handle_keypress(self, key: int) -> None:
        """Handle keyboard input."""
        if key == ord('q') or key == 27:  # Q or ESC
            self.running = False
        elif key == ord(' '):  # Space - pause
            self.paused = not self.paused
            print(f"{'Paused' if self.paused else 'Resumed'}")
        elif key == ord('r'):  # R - restart
            self.restart()
            print("Restarted")
        elif key == ord('s'):  # S - screenshot
            self.save_screenshot()
        elif key == ord('1'):  # Toggle white cursor
            self.show_white = not self.show_white
            if not self.show_white:
                self.renderer.trail_white.clear()  # Clear trail when hidden
            print(f"White cursor (ACTUAL): {'ON' if self.show_white else 'OFF'}")
        elif key == ord('2'):  # Toggle red cursor
            self.show_red = not self.show_red
            if not self.show_red:
                self.renderer.trail_red.clear()  # Clear trail when hidden
            print(f"Red cursor (LAGGED): {'ON' if self.show_red else 'OFF'}")
        elif key == ord('3'):  # Toggle green cursor
            self.show_green = not self.show_green
            if not self.show_green:
                self.renderer.trail_green.clear()  # Clear trail when hidden
            print(f"Green cursor (PREDICTED): {'ON' if self.show_green else 'OFF'}")
        elif key == ord('t'):  # Toggle trails
            self.show_trails = not self.show_trails
            if not self.show_trails:
                self.renderer.clear_trails()  # Clear all trails when disabled
            print(f"Trails: {'ON' if self.show_trails else 'OFF'}")
        elif key == ord('i'):  # Toggle info panel
            self.show_info = not self.show_info
            print(f"Info panel: {'ON' if self.show_info else 'OFF'}")
    
    def restart(self) -> None:
        """Reset demo to beginning."""
        self.frame_count = 0
        self.start_time = time.time()
        self.lag_buffer.clear()
        self.position_history.clear()
        self.renderer.clear_trails()
        self.data_source.reset()
        self.error_history.clear()
        
        if self.video_capture:
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    def save_screenshot(self) -> None:
        """Save current frame as screenshot."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"tele_touch_screenshot_{timestamp}.png"
        
        # Get current frame
        current_time = time.time() - self.start_time if self.start_time else 0
        frame = self.process_frame(current_time)
        
        cv2.imwrite(filename, frame)
        print(f"Screenshot saved: {filename}")
    
    def run(self) -> None:
        """Main demo loop."""
        print("\n" + "="*50)
        print("TELE-TOUCH Demo Starting")
        print("="*50)
        print("\nControls:")
        print("  SPACE  - Pause/Resume")
        print("  R      - Restart")
        print("  Q/ESC  - Quit")
        print("  S      - Screenshot")
        print("  1/2/3  - Toggle cursors (white/red/green)")
        print("  T      - Toggle trails")
        print("  I      - Toggle info panel")
        print("="*50 + "\n")
        
        # Create window
        window_name = "TELE-TOUCH: Surgical Prediction System"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, self.width, self.height)
        
        self.start_time = time.time()
        
        try:
            while self.running:
                frame_start = time.time()
                
                if not self.paused:
                    # Calculate timestamp
                    current_time = frame_start - self.start_time
                    
                    # Process and render frame
                    frame = self.process_frame(current_time)
                    
                    self.frame_count += 1
                else:
                    # When paused, just show the last frame
                    current_time = self.frame_count / self.fps
                    frame = self.process_frame(current_time)
                
                # Display
                cv2.imshow(window_name, frame)
                
                # Handle input (1ms wait for key)
                key = cv2.waitKey(1) & 0xFF
                if key != 255:  # Key was pressed
                    self.handle_keypress(key)
                
                # Frame rate control
                elapsed = time.time() - frame_start
                sleep_time = self.frame_time - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
                # Print FPS every 100 frames
                if self.frame_count % 100 == 0 and self.frame_count > 0:
                    actual_fps = self.frame_count / (time.time() - self.start_time)
                    print(f"Frame {self.frame_count} | FPS: {actual_fps:.1f} | "
                          f"Avg Error: {self.avg_error:.1f}px")
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            # Cleanup
            if self.video_capture:
                self.video_capture.release()
            cv2.destroyAllWindows()
            
            print("\nDemo ended.")
            print(f"Total frames: {self.frame_count}")
            if self.frame_count > 0:
                total_time = time.time() - self.start_time
                print(f"Average FPS: {self.frame_count / total_time:.1f}")
                print(f"Average prediction error: {self.avg_error:.1f}px")


def main():
    """Entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="TELE-TOUCH Surgical Prediction Demo")
    parser.add_argument("--video", type=str, default=None,
                        help="Path to surgical video file")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to trained AI model (.pth)")
    parser.add_argument("--width", type=int, default=1280,
                        help="Display width (default: 1280)")
    parser.add_argument("--height", type=int, default=720,
                        help="Display height (default: 720)")
    parser.add_argument("--fps", type=int, default=30,
                        help="Target FPS (default: 30)")
    parser.add_argument("--latency", type=int, default=500,
                        help="Simulated latency in ms (default: 500)")
    
    args = parser.parse_args()
    
    demo = TeleTouchDemo(
        width=args.width,
        height=args.height,
        fps=args.fps,
        latency_ms=args.latency,
        video_path=args.video,
        model_path=args.model
    )
    
    demo.run()


if __name__ == "__main__":
    main()