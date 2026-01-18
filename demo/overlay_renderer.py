"""
Overlay Renderer - Draws all visual elements on top of video frames

Elements:
1. Three cursors (white=actual, red=lagged, green=predicted)
2. Motion trails for each cursor
3. Info panel with stats (top-left)
4. Legend (bottom)
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional
from collections import deque


class TrailManager:
    """Manages motion trails for a single cursor."""
    
    def __init__(self, max_length: int = 25, color: Tuple[int, int, int] = (255, 255, 255)):
        """
        Args:
            max_length: How many positions to keep in trail
            color: BGR color for the trail
        """
        self.max_length = max_length
        self.color = color
        self.positions: deque = deque(maxlen=max_length)
    
    def push(self, pos: Tuple[int, int]) -> None:
        """Add a position to the trail (in pixel coordinates)."""
        self.positions.append(pos)
    
    def draw(self, frame: np.ndarray, thickness: int = 2) -> None:
        """Draw the trail with fading opacity."""
        if len(self.positions) < 2:
            return
        
        positions = list(self.positions)
        n = len(positions)
        
        for i in range(n - 1):
            # Calculate opacity (older = more transparent)
            alpha = (i + 1) / n  # 0 to 1
            
            # Create faded color
            faded_color = tuple(int(c * alpha) for c in self.color)
            
            # Draw line segment
            pt1 = positions[i]
            pt2 = positions[i + 1]
            cv2.line(frame, pt1, pt2, faded_color, thickness, cv2.LINE_AA)
    
    def clear(self) -> None:
        """Clear the trail."""
        self.positions.clear()


class OverlayRenderer:
    """Main overlay renderer class."""
    
    # Color definitions (BGR format for OpenCV)
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    RED = (60, 60, 255)       # Brighter red for visibility
    GREEN = (60, 255, 60)     # Brighter green
    CYAN = (255, 255, 0)      # Cyan alternative for predicted
    YELLOW = (80, 255, 255)   # For warnings
    MAGENTA = (255, 0, 255)   # Alternative accent
    
    # Panel colors
    PANEL_BG = (30, 30, 40)      # Dark blue-gray
    PANEL_BORDER = (60, 60, 80)  # Lighter border
    TEXT_COLOR = (220, 220, 220) # Light gray
    
    def __init__(self, frame_width: int = 1280, frame_height: int = 720):
        """
        Initialize renderer.
        
        Args:
            frame_width: Video frame width in pixels
            frame_height: Video frame height in pixels
        """
        self.width = frame_width
        self.height = frame_height
        
        # Create trail managers for each cursor
        self.trail_white = TrailManager(max_length=25, color=self.WHITE)
        self.trail_red = TrailManager(max_length=25, color=self.RED)
        self.trail_green = TrailManager(max_length=25, color=self.GREEN)
        
        # Cursor settings - DIFFERENT SIZES for distinction
        # Made larger for visibility on surgical video backgrounds
        self.cursor_radius_actual = 12     # White: solid dot (robotic arm tip)
        self.cursor_radius_lagged = 18     # Red: medium, hollow circle
        self.cursor_radius_predicted = 22  # Green: largest, crosshair style
        self.cursor_outline = 3            # Thicker outline for visibility
    
    def normalized_to_pixel(self, pos: Tuple[float, float]) -> Tuple[int, int]:
        """Convert normalized (0-1) coordinates to pixel coordinates."""
        x = int(pos[0] * self.width)
        y = int(pos[1] * self.height)
        return (x, y)
    
    def draw_cursor(self, frame: np.ndarray, pos: Tuple[float, float], 
                    color: Tuple[int, int, int], label: str,
                    outline_color: Optional[Tuple[int, int, int]] = None,
                    cursor_type: str = "default") -> None:
        """
        Draw a cursor with label. Different styles for different cursor types.
        
        Args:
            frame: Image to draw on
            pos: (x, y) normalized position
            color: BGR fill color
            label: Text label to show
            outline_color: Outline color (default: contrasting)
            cursor_type: "actual", "lagged", "predicted", or "default"
        """
        if pos is None:
            return
            
        px, py = self.normalized_to_pixel(pos)
        
        # Default outline color
        if outline_color is None:
            outline_color = self.BLACK if color == self.WHITE else self.WHITE
        
        # Draw different shapes based on cursor type
        if cursor_type == "actual":
            # ACTUAL (White): Robotic arm tip - solid filled circle with glow effect
            # Draw outer glow (for visibility on surgical video)
            cv2.circle(frame, (px, py), self.cursor_radius_actual + 6, 
                      (200, 200, 200), 3, cv2.LINE_AA)
            cv2.circle(frame, (px, py), self.cursor_radius_actual + 3, 
                      (150, 150, 150), 2, cv2.LINE_AA)
            # Draw filled circle (robotic arm tip)
            cv2.circle(frame, (px, py), self.cursor_radius_actual, color, -1, cv2.LINE_AA)
            # Draw black outline for contrast
            cv2.circle(frame, (px, py), self.cursor_radius_actual, self.BLACK, 
                       3, cv2.LINE_AA)
            
        elif cursor_type == "lagged":
            # LAGGED (Red): Hollow square/diamond shape - what robot sees (delayed)
            size = self.cursor_radius_lagged
            # Draw outer glow for visibility
            cv2.circle(frame, (px, py), size + 4, (100, 100, 255), 2, cv2.LINE_AA)
            # Draw a rotated square (diamond)
            pts = np.array([
                [px, py - size],      # Top
                [px + size, py],      # Right
                [px, py + size],      # Bottom
                [px - size, py]       # Left
            ], np.int32)
            cv2.polylines(frame, [pts], True, color, 4, cv2.LINE_AA)
            # Inner diamond
            inner_size = size - 5
            pts_inner = np.array([
                [px, py - inner_size],
                [px + inner_size, py],
                [px, py + inner_size],
                [px - inner_size, py]
            ], np.int32)
            cv2.polylines(frame, [pts_inner], True, (150, 150, 255), 2, cv2.LINE_AA)
            
        elif cursor_type == "predicted":
            # PREDICTED (Green): Crosshair/target reticle style - AI prediction
            size = self.cursor_radius_predicted
            # Draw outer glow for visibility on surgical video
            cv2.circle(frame, (px, py), size + 5, (100, 255, 100), 2, cv2.LINE_AA)
            # Outer circle (hollow) - thicker for visibility
            cv2.circle(frame, (px, py), size, color, 3, cv2.LINE_AA)
            # Inner circle (hollow, smaller)
            cv2.circle(frame, (px, py), size // 2, color, 3, cv2.LINE_AA)
            # Crosshair lines - thicker for visibility
            gap = 8  # Gap in center
            line_thickness = 3
            # Horizontal line
            cv2.line(frame, (px - size - 8, py), (px - gap, py), color, line_thickness, cv2.LINE_AA)
            cv2.line(frame, (px + gap, py), (px + size + 8, py), color, line_thickness, cv2.LINE_AA)
            # Vertical line
            cv2.line(frame, (px, py - size - 8), (px, py - gap), color, line_thickness, cv2.LINE_AA)
            cv2.line(frame, (px, py + gap), (px, py + size + 8), color, line_thickness, cv2.LINE_AA)
            
        else:
            # Default: simple filled circle
            cv2.circle(frame, (px, py), 10, color, -1, cv2.LINE_AA)
            cv2.circle(frame, (px, py), 10, outline_color, 2, cv2.LINE_AA)
        
        # Draw label with background - position varies by type to avoid overlap
        if cursor_type == "actual":
            label_pos = (px + 15, py - 10)
        elif cursor_type == "lagged":
            label_pos = (px + 20, py + 5)
        elif cursor_type == "predicted":
            label_pos = (px + 25, py + 15)
        else:
            label_pos = (px + 15, py + 5)
        
        # Get text size for background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        
        # Draw text background
        padding = 3
        cv2.rectangle(frame, 
                      (label_pos[0] - padding, label_pos[1] - text_h - padding),
                      (label_pos[0] + text_w + padding, label_pos[1] + padding),
                      self.PANEL_BG, -1)
        
        # Draw text
        cv2.putText(frame, label, label_pos, font, font_scale, color, thickness, cv2.LINE_AA)
    
    def update_trails(self, white_pos: Optional[Tuple[float, float]],
                      red_pos: Optional[Tuple[float, float]],
                      green_pos: Optional[Tuple[float, float]]) -> None:
        """Update all trails with new positions."""
        if white_pos:
            self.trail_white.push(self.normalized_to_pixel(white_pos))
        if red_pos:
            self.trail_red.push(self.normalized_to_pixel(red_pos))
        if green_pos:
            self.trail_green.push(self.normalized_to_pixel(green_pos))
    
    def draw_trails(self, frame: np.ndarray, 
                    show_white: bool = True,
                    show_red: bool = True,
                    show_green: bool = True) -> None:
        """Draw motion trails only for visible cursors."""
        if show_white:
            self.trail_white.draw(frame, thickness=2)
        if show_red:
            self.trail_red.draw(frame, thickness=2)
        if show_green:
            self.trail_green.draw(frame, thickness=2)
    
    def draw_info_panel(self, frame: np.ndarray, 
                        error_pixels: float,
                        confidence: float,
                        predictor_name: str = "AI Predictor",
                        latency_ms: int = 500) -> None:
        """
        Draw the statistics panel in top-left corner.
        
        Args:
            frame: Image to draw on
            error_pixels: Prediction error in pixels
            confidence: Confidence percentage (0-100)
            predictor_name: Name of the prediction method
            latency_ms: Simulated latency in milliseconds
        """
        # Panel dimensions
        panel_x, panel_y = 20, 20
        panel_w, panel_h = 320, 140
        
        # Draw panel background with transparency
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), 
                      (panel_x + panel_w, panel_y + panel_h),
                      self.PANEL_BG, -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        
        # Draw border
        cv2.rectangle(frame, (panel_x, panel_y), 
                      (panel_x + panel_w, panel_y + panel_h),
                      self.PANEL_BORDER, 2)
        
        # Draw title
        font = cv2.FONT_HERSHEY_SIMPLEX
        title = "TELE-TOUCH PREDICTION SYSTEM"
        cv2.putText(frame, title, (panel_x + 10, panel_y + 28),
                    font, 0.55, self.WHITE, 1, cv2.LINE_AA)
        
        # Draw separator line
        cv2.line(frame, (panel_x + 10, panel_y + 38), 
                 (panel_x + panel_w - 10, panel_y + 38), self.PANEL_BORDER, 1)
        
        # Stats
        y_offset = panel_y + 60
        line_height = 24
        
        stats = [
            f"Latency Compensation: {latency_ms}ms",
            f"Prediction Error: {error_pixels:.1f} px",
            f"Confidence: {confidence:.0f}%",
            f"Method: {predictor_name}"
        ]
        
        for i, stat in enumerate(stats):
            # Color code confidence
            color = self.TEXT_COLOR
            if i == 2:  # Confidence line
                if confidence >= 90:
                    color = self.GREEN
                elif confidence >= 70:
                    color = self.YELLOW
                else:
                    color = self.RED
            
            cv2.putText(frame, stat, (panel_x + 15, y_offset + i * line_height),
                        font, 0.5, color, 1, cv2.LINE_AA)
    
    def draw_legend(self, frame: np.ndarray) -> None:
        """Draw the legend at the bottom of the frame with distinct shapes."""
        # Legend position (bottom center)
        legend_y = self.height - 45
        
        # Background bar
        cv2.rectangle(frame, (0, legend_y - 20), (self.width, self.height),
                      self.PANEL_BG, -1)
        cv2.line(frame, (0, legend_y - 20), (self.width, legend_y - 20),
                 self.PANEL_BORDER, 1)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        x_offset = 60
        spacing = 320
        
        # Item 1: White - solid circle
        x = x_offset
        cv2.circle(frame, (x, legend_y + 8), 6, self.WHITE, -1, cv2.LINE_AA)
        cv2.circle(frame, (x, legend_y + 8), 6, self.BLACK, 2, cv2.LINE_AA)
        cv2.putText(frame, "Actual Position", (x + 15, legend_y + 13),
                    font, 0.5, self.TEXT_COLOR, 1, cv2.LINE_AA)
        
        # Item 2: Red - diamond shape
        x = x_offset + spacing
        size = 8
        pts = np.array([
            [x, legend_y + 8 - size],
            [x + size, legend_y + 8],
            [x, legend_y + 8 + size],
            [x - size, legend_y + 8]
        ], np.int32)
        cv2.polylines(frame, [pts], True, self.RED, 2, cv2.LINE_AA)
        cv2.putText(frame, "Lagged Robot (500ms)", (x + 15, legend_y + 13),
                    font, 0.5, self.TEXT_COLOR, 1, cv2.LINE_AA)
        
        # Item 3: Green - crosshair
        x = x_offset + spacing * 2
        size = 10
        cv2.circle(frame, (x, legend_y + 8), size, self.GREEN, 2, cv2.LINE_AA)
        cv2.line(frame, (x - size - 3, legend_y + 8), (x - 3, legend_y + 8), 
                 self.GREEN, 2, cv2.LINE_AA)
        cv2.line(frame, (x + 3, legend_y + 8), (x + size + 3, legend_y + 8), 
                 self.GREEN, 2, cv2.LINE_AA)
        cv2.line(frame, (x, legend_y + 8 - size - 3), (x, legend_y + 8 - 3), 
                 self.GREEN, 2, cv2.LINE_AA)
        cv2.line(frame, (x, legend_y + 8 + 3), (x, legend_y + 8 + size + 3), 
                 self.GREEN, 2, cv2.LINE_AA)
        cv2.putText(frame, "AI Prediction", (x + 18, legend_y + 13),
                    font, 0.5, self.TEXT_COLOR, 1, cv2.LINE_AA)
    
    def draw_warning(self, frame: np.ndarray, message: str) -> None:
        """Draw a warning banner (when prediction error is too high)."""
        # Warning bar at top
        bar_height = 50
        
        # Flashing yellow background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (self.width, bar_height), (0, 180, 255), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        # Warning icon and text
        font = cv2.FONT_HERSHEY_SIMPLEX
        warning_text = f"⚠ WARNING: {message}"
        cv2.putText(frame, warning_text, (20, 35), font, 0.8, self.BLACK, 2, cv2.LINE_AA)
    
    def calculate_error(self, actual: Tuple[float, float], 
                        predicted: Tuple[float, float]) -> float:
        """
        Calculate prediction error in pixels.
        
        Args:
            actual: (x, y) normalized actual position
            predicted: (x, y) normalized predicted position
            
        Returns:
            Distance in pixels
        """
        if actual is None or predicted is None:
            return 0.0
        
        # Calculate in normalized space, then convert to pixels
        dx = (actual[0] - predicted[0]) * self.width
        dy = (actual[1] - predicted[1]) * self.height
        
        return np.sqrt(dx * dx + dy * dy)
    
    def clear_trails(self) -> None:
        """Clear all trails (use when restarting video)."""
        self.trail_white.clear()
        self.trail_red.clear()
        self.trail_green.clear()


# Quick visual test
if __name__ == "__main__":
    print("Testing OverlayRenderer...")
    
    # Create a test frame
    renderer = OverlayRenderer(1280, 720)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame[:] = (40, 40, 40)  # Dark gray background
    
    # Simulate some positions
    white_pos = (0.5, 0.4)
    red_pos = (0.45, 0.38)
    green_pos = (0.52, 0.42)
    
    # Update trails with a few positions
    for i in range(20):
        t = i / 20.0
        w = (0.3 + t * 0.2, 0.3 + t * 0.1)
        r = (0.28 + t * 0.2, 0.28 + t * 0.1)
        g = (0.32 + t * 0.2, 0.32 + t * 0.1)
        renderer.update_trails(w, r, g)
    
    # Draw everything
    renderer.draw_trails(frame)
    renderer.draw_cursor(frame, white_pos, renderer.WHITE, "ACTUAL")
    renderer.draw_cursor(frame, red_pos, renderer.RED, "LAGGED")
    renderer.draw_cursor(frame, green_pos, renderer.GREEN, "PREDICTED")
    
    error = renderer.calculate_error(white_pos, green_pos)
    confidence = max(0, 100 - error)
    renderer.draw_info_panel(frame, error, confidence, "Quadratic Extrapolation")
    renderer.draw_legend(frame)
    
    # Save test image
    cv2.imwrite("overlay_test.png", frame)
    print("✓ Saved test image: overlay_test.png")
    print(f"  Prediction error: {error:.1f}px")
    print("\n✓ OverlayRenderer test complete!")