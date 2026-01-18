#!/usr/bin/env python3
"""
Collect custom demo data for cholecystectomy latency compensation training.
Uses webcam hand tracking to capture surgeon-like movements for personalized model training.
"""

import pygame
import numpy as np
import cv2
import mediapipe as mp
import time
from pathlib import Path
import argparse
import json
from datetime import datetime


class DemoDataCollector:
    """Collect cholecystectomy-specific demo data via webcam."""

    def __init__(self, output_file="demo_data.npy", camera_index=0):
        self.output_file = Path(output_file)
        self.camera_index = camera_index

        # Initialize pygame
        pygame.init()
        self.screen_width = 1200
        self.screen_height = 800
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Cholecystectomy Demo Data Collection")

        # Colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 0, 255)
        self.YELLOW = (255, 255, 0)
        self.CYAN = (0, 255, 255)

        # Initialize hand tracking
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils

        # Camera setup
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Data collection
        self.collected_data = []
        self.is_recording = False
        self.recording_start_time = 0
        self.current_instruction = 0

        # Instructions for cholecystectomy movements
        self.instructions = [
            "DISSECTION: Make precise curved movements following gallbladder anatomy",
            "CLIPPING: Perform quick, accurate positioning for cystic duct clipping",
            "RETRACTION: Simulate steady gallbladder retraction with gentle resistance",
            "NAVIGATION: Navigate around abdominal cavity with controlled movements",
            "PRECISION: Practice fine suturing and tissue manipulation",
            "STEADY: Maintain steady position for critical viewing"
        ]

        # Fonts
        self.font = pygame.font.Font(None, 28)
        self.small_font = pygame.font.Font(None, 20)

    def run_collection(self):
        """Main data collection loop."""
        print("🎥 Cholecystectomy Demo Data Collection")
        print("Instructions:")
        print("  SPACE: Start/Stop recording")
        print("  C: Clear current recording")
        print("  S: Save data and exit")
        print("  ESC: Exit without saving")
        print("  I: Next instruction")
        print("=" * 50)

        clock = pygame.time.Clock()
        running = True

        while running:
            dt = clock.tick(60) / 1000.0

            # Handle events
            running = self._handle_events()

            # Get hand position
            hand_pos, camera_frame = self._get_hand_position()

            # Record data if active
            if self.is_recording and hand_pos is not None:
                timestamp = time.time() - self.recording_start_time
                data_point = [hand_pos[0], hand_pos[1], timestamp]
                self.collected_data.append(data_point)

            # Render
            self._render_frame(hand_pos, camera_frame)

            pygame.display.flip()

        # Cleanup
        self.cap.release()
        pygame.quit()

    def _handle_events(self):
        """Handle pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    if self.collected_data:
                        response = input("Exit without saving? (y/N): ").lower().strip()
                        if response != 'y':
                            return True
                    return False
                elif event.key == pygame.K_SPACE:
                    self._toggle_recording()
                elif event.key == pygame.K_c:
                    self._clear_recording()
                elif event.key == pygame.K_s:
                    self._save_data()
                    return False
                elif event.key == pygame.K_i:
                    self.current_instruction = (self.current_instruction + 1) % len(self.instructions)

        return True

    def _toggle_recording(self):
        """Start or stop recording."""
        if not self.is_recording:
            self.is_recording = True
            self.recording_start_time = time.time()
            print("🎬 Started recording cholecystectomy movements")
        else:
            self.is_recording = False
            duration = time.time() - self.recording_start_time
            print(".1f"
    def _clear_recording(self):
        """Clear current recording."""
        if self.collected_data:
            cleared_count = len(self.collected_data)
            self.collected_data.clear()
            print(f"🗑️ Cleared {cleared_count} data points")
        else:
            print("📝 No data to clear")

    def _save_data(self):
        """Save collected data to file."""
        if not self.collected_data:
            print("⚠️ No data to save")
            return

        # Convert to numpy array
        data_array = np.array(self.collected_data)

        # Save data
        np.save(self.output_file, data_array)

        # Save metadata
        metadata = {
            'collection_date': datetime.now().isoformat(),
            'total_samples': len(self.collected_data),
            'duration_seconds': data_array[-1, 2] - data_array[0, 2] if len(data_array) > 1 else 0,
            'sampling_rate_hz': len(self.collected_data) / (data_array[-1, 2] - data_array[0, 2]) if len(data_array) > 1 else 0,
            'description': 'Cholecystectomy demo data collected via webcam hand tracking',
            'instructions_used': self.instructions,
            'camera_index': self.camera_index
        }

        metadata_file = self.output_file.with_suffix('.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"💾 Saved {len(self.collected_data)} samples to {self.output_file}")
        print(f"📊 Metadata saved to {metadata_file}")
        print(".2f"        print(".1f"
    def _get_hand_position(self):
        """Get normalized hand position from webcam."""
        ret, frame = self.cap.read()
        if not ret:
            return None, None

        # Flip frame
        frame = cv2.flip(frame, 1)

        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            # Use index finger tip
            hand_landmarks = results.multi_hand_landmarks[0]
            index_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]

            # Convert to screen coordinates (normalized 0-1)
            x_norm = index_finger_tip.x
            y_norm = index_finger_tip.y

            return np.array([x_norm, y_norm]), frame

        return None, frame

    def _render_frame(self, hand_pos, camera_frame):
        """Render the collection interface."""
        self.screen.fill(self.BLACK)

        # Draw main collection area
        field_rect = pygame.Rect(50, 50, self.screen_width - 300, self.screen_height - 100)
        pygame.draw.rect(self.screen, (20, 20, 40), field_rect, 2)

        # Draw anatomical reference (simplified gallbladder)
        center_x, center_y = field_rect.centerx, field_rect.centery
        gallbladder_radius = 60

        pygame.draw.circle(self.screen, self.YELLOW, (center_x, center_y), gallbladder_radius, 2)
        # Cystic duct
        duct_end = (center_x + 80, center_y - 30)
        pygame.draw.line(self.screen, self.CYAN, (center_x, center_y), duct_end, 3)
        # Hepatic artery
        artery_end = (center_x + 60, center_y + 40)
        pygame.draw.line(self.screen, self.RED, (center_x, center_y), artery_end, 3)

        # Draw current hand position
        if hand_pos is not None:
            x = int(hand_pos[0] * (field_rect.width - 40) + field_rect.left + 20)
            y = int(hand_pos[1] * (field_rect.height - 40) + field_rect.top + 20)

            color = self.RED if self.is_recording else self.GREEN
            pygame.draw.circle(self.screen, color, (x, y), 8, 3)
            pygame.draw.circle(self.screen, color, (x, y), 12, 1)

        # Draw recorded trail
        if self.collected_data:
            points = []
            for data_point in self.collected_data[-200:]:  # Last 200 points
                x = int(data_point[0] * (field_rect.width - 40) + field_rect.left + 20)
                y = int(data_point[1] * (field_rect.height - 40) + field_rect.top + 20)
                points.append((x, y))

            if len(points) > 1:
                pygame.draw.lines(self.screen, self.BLUE, False, points, 2)

        # Draw UI panels
        self._draw_ui_panels(camera_frame)

    def _draw_ui_panels(self, camera_frame):
        """Draw control panels."""
        panel_x = self.screen_width - 250
        panel_y = 10

        # Main control panel
        pygame.draw.rect(self.screen, (30, 30, 30), (panel_x, panel_y, 240, self.screen_height - 20))
        pygame.draw.rect(self.screen, self.WHITE, (panel_x, panel_y, 240, self.screen_height - 20), 1)

        # Title
        title = self.font.render("Demo Data Collection", True, self.WHITE)
        self.screen.blit(title, (panel_x + 10, panel_y + 10))

        y_offset = 50

        # Recording status
        status_color = self.RED if self.is_recording else self.GREEN
        status_text = "RECORDING" if self.is_recording else "READY"
        text = self.font.render(f"Status: {status_text}", True, status_color)
        self.screen.blit(text, (panel_x + 10, y_offset))
        y_offset += 30

        # Data count
        data_count = len(self.collected_data)
        text = self.font.render(f"Samples: {data_count}", True, self.CYAN)
        self.screen.blit(text, (panel_x + 10, y_offset))
        y_offset += 30

        # Recording time
        if self.is_recording:
            elapsed = time.time() - self.recording_start_time
            time_text = ".1f"
        else:
            time_text = "0.0s"
        text = self.font.render(f"Time: {time_text}", True, self.WHITE)
        self.screen.blit(text, (panel_x + 10, y_offset))
        y_offset += 30

        # Current instruction
        instr_title = self.small_font.render("Current Task:", True, self.YELLOW)
        self.screen.blit(instr_title, (panel_x + 10, y_offset))
        y_offset += 20

        # Instruction text (wrapped)
        instruction = self.instructions[self.current_instruction]
        words = instruction.split()
        lines = []
        current_line = ""
        for word in words:
            test_line = current_line + " " + word if current_line else word
            if self.small_font.size(test_line)[0] < 220:
                current_line = test_line
            else:
                lines.append(current_line)
                current_line = word
        lines.append(current_line)

        for line in lines:
            text = self.small_font.render(line, True, self.WHITE)
            self.screen.blit(text, (panel_x + 10, y_offset))
            y_offset += 18

        y_offset += 20

        # Controls
        controls = [
            "SPACE: Start/Stop Recording",
            "C: Clear Data",
            "I: Next Task",
            "S: Save & Exit",
            "ESC: Exit"
        ]

        for control in controls:
            text = self.small_font.render(control, True, (150, 150, 150))
            self.screen.blit(text, (panel_x + 10, y_offset))
            y_offset += 16

        # Camera feed
        if camera_frame is not None:
            # Resize and position camera feed
            frame_rgb = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (200, 150))
            frame_surface = pygame.surfarray.make_surface(np.transpose(frame_resized, (1, 0, 2)))
            self.screen.blit(frame_surface, (panel_x + 20, self.screen_height - 180))

            # Camera border
            pygame.draw.rect(self.screen, self.GREEN, (panel_x + 20, self.screen_height - 180, 200, 150), 2)


def main():
    """Main collection function."""
    parser = argparse.ArgumentParser(description="Collect cholecystectomy demo data")
    parser.add_argument('--output', type=str, default='demo_data.npy',
                       help='Output file for collected data')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera index to use')

    args = parser.parse_args()

    try:
        collector = DemoDataCollector(args.output, args.camera)
        collector.run_collection()
    except KeyboardInterrupt:
        print("\n👋 Collection interrupted")
    except Exception as e:
        print(f"❌ Collection error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()