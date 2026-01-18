"""
Interactive Parameter Tuning for Color-Based Instrument Detection

This script allows you to tune HSV color ranges and detection parameters
by adjusting sliders and seeing the results in real-time.

Usage:
    python demo/tune_detector.py --video path/to/video.mp4 --frame 100
"""

import cv2
import numpy as np
import argparse
from pathlib import Path


class ColorDetectorTuner:
    """Interactive tool for tuning color detection parameters."""
    
    def __init__(self, frame):
        self.frame = frame.copy()
        self.hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Default parameters
        self.hue_low = 80
        self.hue_high = 130
        self.sat_low = 0
        self.sat_high = 80
        self.val_low = 120
        self.val_high = 255
        self.min_area = 500
        self.min_elongation = 3.0
        
        # Create window and trackbars
        cv2.namedWindow('Tuner', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Tuner', 1200, 400)
        
        # Create trackbars
        cv2.createTrackbar('Hue Low', 'Tuner', self.hue_low, 179, self._on_change)
        cv2.createTrackbar('Hue High', 'Tuner', self.hue_high, 179, self._on_change)
        cv2.createTrackbar('Sat Low', 'Tuner', self.sat_low, 255, self._on_change)
        cv2.createTrackbar('Sat High', 'Tuner', self.sat_high, 255, self._on_change)
        cv2.createTrackbar('Val Low', 'Tuner', self.val_low, 255, self._on_change)
        cv2.createTrackbar('Val High', 'Tuner', self.val_high, 255, self._on_change)
        cv2.createTrackbar('Min Area', 'Tuner', self.min_area, 5000, self._on_change)
        cv2.createTrackbar('Min Elongation', 'Tuner', int(self.min_elongation * 10), 100, self._on_change)
        
        self.update_display()
    
    def _on_change(self, val):
        """Callback when trackbar changes."""
        self.update_parameters()
        self.update_display()
    
    def update_parameters(self):
        """Read current trackbar values."""
        self.hue_low = cv2.getTrackbarPos('Hue Low', 'Tuner')
        self.hue_high = cv2.getTrackbarPos('Hue High', 'Tuner')
        self.sat_low = cv2.getTrackbarPos('Sat Low', 'Tuner')
        self.sat_high = cv2.getTrackbarPos('Sat High', 'Tuner')
        self.val_low = cv2.getTrackbarPos('Val Low', 'Tuner')
        self.val_high = cv2.getTrackbarPos('Val High', 'Tuner')
        self.min_area = cv2.getTrackbarPos('Min Area', 'Tuner')
        self.min_elongation = cv2.getTrackbarPos('Min Elongation', 'Tuner') / 10.0
    
    def create_mask(self):
        """Create HSV mask with current parameters."""
        # Create mask
        mask = cv2.inRange(
            self.hsv_frame,
            np.array([self.hue_low, self.sat_low, self.val_low]),
            np.array([self.hue_high, self.sat_high, self.val_high])
        )
        
        # Clean mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        return mask
    
    def detect_instruments(self, mask):
        """Detect instruments from mask."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        instruments = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area:
                continue
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            elongation = max(w, h) / max(min(w, h), 1)
            
            if elongation < self.min_elongation:
                continue
            
            # Find tip (furthest point from centroid)
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
            
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            # Find point furthest from centroid
            distances = [np.sqrt((p[0][0] - cx)**2 + (p[0][1] - cy)**2) for p in contour]
            max_idx = np.argmax(distances)
            tip_x, tip_y = contour[max_idx][0]
            
            instruments.append({
                'tip': (tip_x, tip_y),
                'centroid': (cx, cy),
                'bbox': (x, y, w, h),
                'area': area
            })
        
        return instruments
    
    def update_display(self):
        """Update the display with current parameters."""
        mask = self.create_mask()
        instruments = self.detect_instruments(mask)
        
        # Create three-panel display
        display = np.zeros((self.frame.shape[0], self.frame.shape[1] * 3, 3), dtype=np.uint8)
        
        # Panel 1: Original frame
        display[:, :self.frame.shape[1]] = self.frame
        
        # Panel 2: Mask
        mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        display[:, self.frame.shape[1]:self.frame.shape[1]*2] = mask_colored
        
        # Panel 3: Detected instruments
        result = self.frame.copy()
        for inst in instruments:
            tip = inst['tip']
            centroid = inst['centroid']
            bbox = inst['bbox']
            
            # Draw bounding box
            cv2.rectangle(result, (bbox[0], bbox[1]), 
                         (bbox[0] + bbox[2], bbox[1] + bbox[3]), 
                         (0, 255, 0), 2)
            
            # Draw centroid
            cv2.circle(result, centroid, 5, (255, 0, 0), -1)
            
            # Draw tip
            cv2.circle(result, tip, 8, (0, 0, 255), -1)
            
            # Draw line from centroid to tip
            cv2.line(result, centroid, tip, (255, 255, 0), 2)
        
        display[:, self.frame.shape[1]*2:] = result
        
        # Add labels
        cv2.putText(display, "Original", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(display, "Mask", (self.frame.shape[1] + 10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(display, f"Detected: {len(instruments)}", 
                   (self.frame.shape[1]*2 + 10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Add parameter text
        param_text = [
            f"H: [{self.hue_low}, {self.hue_high}]",
            f"S: [{self.sat_low}, {self.sat_high}]",
            f"V: [{self.val_low}, {self.val_high}]",
            f"Min Area: {self.min_area}",
            f"Min Elongation: {self.min_elongation:.1f}"
        ]
        y_offset = 60
        for text in param_text:
            cv2.putText(display, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            y_offset += 25
        
        cv2.imshow('Tuner', display)
    
    def get_parameters(self):
        """Get current parameters as dictionary."""
        return {
            'hue': (self.hue_low, self.hue_high),
            'sat': (self.sat_low, self.sat_high),
            'val': (self.val_low, self.val_high),
            'min_area': self.min_area,
            'min_elongation': self.min_elongation
        }
    
    def save_parameters(self, filepath):
        """Save parameters to file."""
        import json
        params = self.get_parameters()
        with open(filepath, 'w') as f:
            json.dump(params, f, indent=2)
        print(f"Parameters saved to {filepath}")
    
    def run(self):
        """Run the tuning interface."""
        print("\n" + "="*60)
        print("COLOR DETECTOR TUNER")
        print("="*60)
        print("\nControls:")
        print("  - Adjust sliders to tune parameters")
        print("  - Press 'S' to save parameters")
        print("  - Press 'Q' or ESC to quit")
        print("\n" + "="*60 + "\n")
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # Q or ESC
                break
            elif key == ord('s'):  # Save
                self.save_parameters('detector_params.json')
        
        cv2.destroyAllWindows()
        return self.get_parameters()


def main():
    parser = argparse.ArgumentParser(description='Tune color detection parameters')
    parser.add_argument('--video', type=str, help='Path to video file')
    parser.add_argument('--frame', type=int, default=100, help='Frame number to use')
    parser.add_argument('--image', type=str, help='Path to image file (alternative to video)')
    
    args = parser.parse_args()
    
    # Load frame
    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"Error: Could not load image {args.image}")
            return
    elif args.video:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            print(f"Error: Could not open video {args.video}")
            return
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print(f"Error: Could not read frame {args.frame}")
            return
    else:
        print("Error: Must provide either --video or --image")
        return
    
    # Run tuner
    tuner = ColorDetectorTuner(frame)
    params = tuner.run()
    
    print("\nFinal parameters:")
    print(params)


if __name__ == '__main__':
    main()
