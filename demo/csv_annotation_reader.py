"""
CSV Annotation Reader - Loads instrument positions from CSV files.

Supports multiple CSV formats:
1. Frame-based: frame_number, x, y, instrument_type
2. Time-based: timestamp, x, y, instrument_type
3. Multi-instrument: frame, x1, y1, x2, y2, ...
"""

import csv
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np


class CSVAnnotationReader:
    """
    Reads CSV annotation files and provides instrument positions for each frame.
    """
    
    def __init__(self, csv_path: str, 
                 frame_column: str = 'frame',
                 x_column: str = 'x',
                 y_column: str = 'y',
                 instrument_column: Optional[str] = None,
                 has_header: bool = True):
        """
        Initialize CSV annotation reader.
        
        Args:
            csv_path: Path to CSV annotation file
            frame_column: Name of column containing frame numbers (or 'Date' for timestamp-based)
            x_column: Name of column containing x coordinates (or pattern like 'PSM1_position_x')
            y_column: Name of column containing y coordinates (or pattern like 'PSM1_position_y')
            instrument_column: Optional column for instrument type/ID
            has_header: Whether CSV has header row
        """
        self.csv_path = Path(csv_path)
        
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        # Load CSV
        self.df = pd.read_csv(self.csv_path)
        
        print(f"[CSVAnnotationReader] Loaded CSV: {self.csv_path.name}")
        print(f"  Columns: {len(self.df.columns)} total")
        print(f"  Total rows: {len(self.df)}")
        
        # Detect format type
        self.format_type = self._detect_format()
        
        if self.format_type == 'davinci_kinematic':
            # da Vinci robot format: PSM1_position_x, PSM1_position_y, PSM2_position_x, PSM2_position_y
            self._setup_davinci_format()
        else:
            # Standard format: frame, x, y
            self.frame_col = self._find_column(frame_column, ['frame', 'frame_number', 'frame_num', 'f', 'time', 'timestamp', 'Date'])
            self.x_col = self._find_column(x_column, ['x', 'x_pos', 'x_position', 'tip_x', 'pos_x'])
            self.y_col = self._find_column(y_column, ['y', 'y_pos', 'y_position', 'tip_y', 'pos_y'])
            self.instrument_col = self._find_column(instrument_column, ['instrument', 'instrument_type', 'inst', 'id', 'arm'], required=False)
            
            # Check for multiple instruments
            self.multi_instrument = False
            self.instrument_columns = {}
            x_cols = [col for col in self.df.columns if 'x' in col.lower() and col != self.x_col]
            y_cols = [col for col in self.df.columns if 'y' in col.lower() and col != self.y_col]
            
            if len(x_cols) > 0 and len(y_cols) > 0:
                self.multi_instrument = True
                for i, (x_col, y_col) in enumerate(zip(sorted(x_cols), sorted(y_cols))):
                    self.instrument_columns[i] = (x_col, y_col)
                print(f"  Detected {len(self.instrument_columns)} instruments from columns")
        
        # Build frame lookup
        self.frame_data = {}
        self._build_frame_lookup()
        
        print(f"  Processed {len(self.frame_data)} unique frames")
    
    def _detect_format(self) -> str:
        """Detect CSV format type."""
        cols = list(self.df.columns)
        
        # Check for da Vinci format (PSM1_position_x, PSM2_position_x, etc.)
        if any('PSM1_position_x' in col for col in cols) or any('PSM2_position_x' in col for col in cols):
            return 'davinci_kinematic'
        
        return 'standard'
    
    def _setup_davinci_format(self):
        """Setup for da Vinci robot kinematic data format."""
        self.format_type = 'davinci_kinematic'
        
        # Find PSM columns (Patient Side Manipulators - the robotic arms)
        psm1_x = None
        psm1_y = None
        psm2_x = None
        psm2_y = None
        
        for col in self.df.columns:
            if 'PSM1_position_x' in col:
                psm1_x = col
            elif 'PSM1_position_y' in col:
                psm1_y = col
            elif 'PSM2_position_x' in col:
                psm2_x = col
            elif 'PSM2_position_y' in col:
                psm2_y = col
        
        # Store instrument columns
        self.instrument_columns = {}
        if psm1_x and psm1_y:
            self.instrument_columns[0] = (psm1_x, psm1_y)
        if psm2_x and psm2_y:
            self.instrument_columns[1] = (psm2_x, psm2_y)
        
        # Use Date column as frame counter (or row index)
        if 'Date' in self.df.columns:
            self.frame_col = 'Date'
        else:
            self.frame_col = None  # Will use row index
        
        self.multi_instrument = len(self.instrument_columns) > 0
        
        print(f"  Detected da Vinci format with {len(self.instrument_columns)} PSM arms")
        for idx, (x_col, y_col) in self.instrument_columns.items():
            print(f"    PSM{idx+1}: {x_col}, {y_col}")
    
    def _find_column(self, provided: Optional[str], alternatives: List[str], required: bool = True) -> Optional[str]:
        """Find column name in dataframe."""
        if provided and provided in self.df.columns:
            return provided
        
        for alt in alternatives:
            if alt in self.df.columns:
                return alt
        
        if required:
            raise ValueError(f"Could not find required column. Tried: {provided or alternatives}")
        return None
    
    def _build_frame_lookup(self):
        """Build dictionary mapping frame numbers to instrument positions."""
        for idx, row in self.df.iterrows():
            # Get frame number
            if self.format_type == 'davinci_kinematic':
                # Use row index as frame number (or convert Date to frame number)
                if self.frame_col and self.frame_col in self.df.columns:
                    # For da Vinci, we'll use row index as frame number
                    frame_num = idx
                else:
                    frame_num = idx
            else:
                # Standard format
                if self.frame_col:
                    try:
                        frame_num = int(row[self.frame_col])
                    except (ValueError, TypeError):
                        frame_num = idx
                else:
                    frame_num = idx
            
            if self.format_type == 'davinci_kinematic' or self.multi_instrument:
                # Multiple instruments per frame
                instruments = []
                for inst_idx, (x_col, y_col) in self.instrument_columns.items():
                    if pd.notna(row[x_col]) and pd.notna(row[y_col]):
                        x_val = float(row[x_col])
                        y_val = float(row[y_col])
                        
                        # For da Vinci, positions are in meters (3D world coordinates)
                        # We'll use x and y directly (z is ignored for 2D display)
                        instruments.append({
                            'type': f'PSM{inst_idx+1}' if self.format_type == 'davinci_kinematic' else f'instrument_{inst_idx}',
                            'tip_position': [x_val, y_val]
                        })
                self.frame_data[frame_num] = {'instruments': instruments}
            else:
                # Single instrument
                if pd.notna(row[self.x_col]) and pd.notna(row[self.y_col]):
                    inst_type = 'instrument'
                    if self.instrument_col and pd.notna(row[self.instrument_col]):
                        inst_type = str(row[self.instrument_col])
                    
                    if frame_num not in self.frame_data:
                        self.frame_data[frame_num] = {'instruments': []}
                    
                    self.frame_data[frame_num]['instruments'].append({
                        'type': inst_type,
                        'tip_position': [float(row[self.x_col]), float(row[self.y_col])]
                    })
    
    def get_frame(self, frame_number: int) -> List[Dict]:
        """
        Get instrument positions for a specific frame.
        
        Args:
            frame_number: Frame number (0-indexed)
            
        Returns:
            List of instrument dictionaries
        """
        if frame_number in self.frame_data:
            return self.frame_data[frame_number].get('instruments', [])
        else:
            return []
    
    def get_tip_positions(self, frame_number: int) -> List[Tuple[float, float]]:
        """Get just the tip positions for a frame."""
        instruments = self.get_frame(frame_number)
        positions = []
        for inst in instruments:
            tip_pos = inst.get('tip_position')
            if tip_pos and len(tip_pos) >= 2:
                # Keep as float (for da Vinci format, these are in meters)
                positions.append((float(tip_pos[0]), float(tip_pos[1])))
        return positions
    
    def get_all_positions(self, frame_number: int) -> Dict[str, List[Tuple[float, float]]]:
        """Get all positions grouped by instrument type."""
        instruments = self.get_frame(frame_number)
        result = {}
        for inst in instruments:
            inst_type = inst.get('type', 'unknown')
            tip_pos = inst.get('tip_position')
            if tip_pos and len(tip_pos) >= 2:
                if inst_type not in result:
                    result[inst_type] = []
                result[inst_type].append((float(tip_pos[0]), float(tip_pos[1])))
        return result
    
    def has_frame(self, frame_number: int) -> bool:
        """Check if frame has annotations."""
        return frame_number in self.frame_data
    
    def __len__(self) -> int:
        """Total number of annotated frames."""
        return len(self.frame_data)
    
    def get_max_frame(self) -> int:
        """Get maximum frame number."""
        return max(self.frame_data.keys()) if self.frame_data else 0


class CSVAnnotationDataSource:
    """
    Data source that uses CSV annotations to provide instrument positions.
    """
    
    def __init__(self, csv_path: str, video_path: str, instrument_index: int = 0,
                 frame_column: Optional[str] = None,
                 x_column: Optional[str] = None,
                 y_column: Optional[str] = None):
        """
        Initialize CSV annotation data source.
        
        Args:
            csv_path: Path to CSV annotation file
            video_path: Path to corresponding video file
            instrument_index: Which instrument to track (0 = first, 1 = second, etc.)
            frame_column: Optional column name for frame numbers (auto-detected if None)
            x_column: Optional column name for x coordinates (auto-detected if None)
            y_column: Optional column name for y coordinates (auto-detected if None)
        """
        self.csv_path = csv_path
        self.video_path = video_path
        self.instrument_index = instrument_index
        
        # Load CSV annotations
        self.reader = CSVAnnotationReader(
            csv_path,
            frame_column=frame_column or 'frame',
            x_column=x_column or 'x',
            y_column=y_column or 'y'
        )
        
        # Get video frame count
        import cv2
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            self.video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
        else:
            self.video_frame_count = self.reader.get_max_frame() + 1
            self.video_width, self.video_height = 1280, 720  # Default
        
        # For da Vinci format, positions are in meters, need to normalize
        # We'll normalize based on the range of values in the CSV
        if hasattr(self.reader, 'format_type') and self.reader.format_type == 'davinci_kinematic':
            self._normalize_davinci_positions()
        
        # Current frame tracking
        self.current_frame = 0
        
        # Position history for smoothing
        self.position_history = []
        self.last_known_position = (0.5, 0.5)
        
        # Normalization for da Vinci positions (convert meters to normalized 0-1)
        self.x_min, self.x_max = None, None
        self.y_min, self.y_max = None, None
        self._compute_normalization()
        
        print(f"[CSVAnnotationDataSource] Using instrument {instrument_index + 1}")
        print(f"  Video frames: {self.video_frame_count}")
        print(f"  Annotated frames: {len(self.reader)}")
        print(f"  Video resolution: {self.video_width}x{self.video_height}")
        if self.x_min is not None:
            print(f"  Position range: X=[{self.x_min:.3f}, {self.x_max:.3f}], Y=[{self.y_min:.3f}, {self.y_max:.3f}]")
    
    def _compute_normalization(self):
        """Compute normalization ranges for da Vinci positions."""
        if hasattr(self.reader, 'format_type') and self.reader.format_type == 'davinci_kinematic':
            # Get all positions for the selected instrument
            all_x = []
            all_y = []
            
            for frame_num in self.reader.frame_data.keys():
                instruments = self.reader.get_frame(frame_num)
                if self.instrument_index < len(instruments):
                    tip_pos = instruments[self.instrument_index].get('tip_position')
                    if tip_pos:
                        all_x.append(tip_pos[0])
                        all_y.append(tip_pos[1])
            
            if all_x and all_y:
                self.x_min, self.x_max = min(all_x), max(all_x)
                self.y_min, self.y_max = min(all_y), max(all_y)
                # Add small margin
                x_range = self.x_max - self.x_min
                y_range = self.y_max - self.y_min
                if x_range > 0:
                    self.x_min -= x_range * 0.1
                    self.x_max += x_range * 0.1
                if y_range > 0:
                    self.y_min -= y_range * 0.1
                    self.y_max += y_range * 0.1
            else:
                # Default ranges if no data
                self.x_min, self.x_max = -0.5, 0.5
                self.y_min, self.y_max = -0.5, 0.5
    
    def _normalize_davinci_positions(self):
        """Normalize da Vinci positions (deprecated - now done in get_current_position)."""
        pass
    
    def get_current_position(self) -> Tuple[float, float]:
        """Get position for current frame and advance."""
        # Get tip positions for current frame
        tip_positions = self.reader.get_tip_positions(self.current_frame)
        
        # If no annotations for this frame, try previous frames (up to 5 back)
        if not tip_positions:
            for offset in range(1, 6):
                prev_frame = self.current_frame - offset
                if prev_frame >= 0:
                    tip_positions = self.reader.get_tip_positions(prev_frame)
                    if tip_positions:
                        break
        
        # Get the requested instrument position
        if tip_positions and self.instrument_index < len(tip_positions):
            x_val, y_val = tip_positions[self.instrument_index]
            
            # Normalize to 0-1
            # For da Vinci format, positions are in meters, need special normalization
            if hasattr(self.reader, 'format_type') and self.reader.format_type == 'davinci_kinematic':
                # Normalize from world coordinates (meters) to 0-1
                if self.x_min is not None and self.x_max is not None and self.x_max > self.x_min:
                    x_norm = (x_val - self.x_min) / (self.x_max - self.x_min)
                else:
                    x_norm = (x_val + 0.5) / 1.0  # Default: assume -0.5 to 0.5 range
                
                if self.y_min is not None and self.y_max is not None and self.y_max > self.y_min:
                    y_norm = (y_val - self.y_min) / (self.y_max - self.y_min)
                else:
                    y_norm = (y_val + 0.5) / 1.0  # Default: assume -0.5 to 0.5 range
                
                # Clamp to 0-1
                x_norm = max(0.0, min(1.0, x_norm))
                y_norm = max(0.0, min(1.0, y_norm))
            else:
                # Standard format: pixel coordinates
                x_norm = x_val / self.video_width
                y_norm = y_val / self.video_height
            
            # Smooth position (moving average)
            self.position_history.append((x_norm, y_norm))
            if len(self.position_history) > 5:
                self.position_history.pop(0)
            
            # Average last few positions
            if len(self.position_history) > 0:
                avg_x = np.mean([p[0] for p in self.position_history])
                avg_y = np.mean([p[1] for p in self.position_history])
                self.last_known_position = (avg_x, avg_y)
                self.current_frame += 1
                return (avg_x, avg_y)
            else:
                self.last_known_position = (x_norm, y_norm)
                self.current_frame += 1
                return (x_norm, y_norm)
        else:
            # No annotation found - use last known position
            self.current_frame += 1
            return self.last_known_position
    
    def get_position(self, frame_number: int) -> Tuple[float, float]:
        """Get position for a specific frame."""
        tip_positions = self.reader.get_tip_positions(frame_number)
        
        if tip_positions and self.instrument_index < len(tip_positions):
            x_val, y_val = tip_positions[self.instrument_index]
            
            # Normalize (same logic as get_current_position)
            if hasattr(self.reader, 'format_type') and self.reader.format_type == 'davinci_kinematic':
                if self.x_min is not None and self.x_max is not None and self.x_max > self.x_min:
                    x_norm = (x_val - self.x_min) / (self.x_max - self.x_min)
                else:
                    x_norm = (x_val + 0.5) / 1.0
                
                if self.y_min is not None and self.y_max is not None and self.y_max > self.y_min:
                    y_norm = (y_val - self.y_min) / (self.y_max - self.y_min)
                else:
                    y_norm = (y_val + 0.5) / 1.0
                
                x_norm = max(0.0, min(1.0, x_norm))
                y_norm = max(0.0, min(1.0, y_norm))
            else:
                x_norm = x_val / self.video_width
                y_norm = y_val / self.video_height
            
            return (x_norm, y_norm)
        else:
            return self.last_known_position
    
    def reset(self) -> None:
        """Reset to beginning."""
        self.current_frame = 0
        self.position_history = []
    
    def __len__(self) -> int:
        """Total number of frames."""
        return self.video_frame_count
    
    def get_name(self) -> str:
        """Return data source name."""
        return f"CSV Annotations (Instrument {self.instrument_index + 1})"
