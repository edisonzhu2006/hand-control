"""
Screen control system for hand gesture recognition.

Provides comprehensive gesture detection, cursor control, and recording capabilities
with advanced state management, acceleration mapping, and region-aware actions.

Usage:
    controller = ScreenController()
    # Detect and control cursor
    if controller.detect_point_gesture(hand_data):
        controller.move_mouse(hand_data, frame_w, frame_h)
    # Record gestures for training
    controller.start_recording("pinch")
    # ... perform gesture ...
    controller.stop_recording(save=True)
"""

import pyautogui
import numpy as np
import json
import time
from dataclasses import dataclass, asdict, field
from typing import Dict, Tuple, Optional, List, Callable, Any
from enum import Enum
from collections import deque
import os
from datetime import datetime
from functools import wraps


class GestureType(Enum):
    """Enum for supported gesture types."""
    POINT = "point"
    PINCH = "pinch"
    PALM_OPEN = "palm_open"
    SWIPE_LEFT = "swipe_left"
    SWIPE_RIGHT = "swipe_right"
    SWIPE_UP = "swipe_up"
    SWIPE_DOWN = "swipe_down"
    TWO_FINGER_PINCH = "two_finger_pinch"
    IDLE = "idle"


class GestureState(Enum):
    """Enum for gesture state machine states."""
    IDLE = "idle"
    POINTING = "pointing"
    PINCHING = "pinching"
    DRAGGING = "dragging"
    SWIPING = "swiping"
    SCROLLING = "scrolling"
    CONFIRMING = "confirming"


class ScreenRegion(Enum):
    """Enum for screen regions."""
    CENTER = "center"
    TOP = "top"
    BOTTOM = "bottom"
    LEFT = "left"
    RIGHT = "right"
    TOP_LEFT = "top_left"
    TOP_RIGHT = "top_right"
    BOTTOM_LEFT = "bottom_left"
    BOTTOM_RIGHT = "bottom_right"


@dataclass
class GestureRecording:
    """Container for recorded gesture data."""
    gesture_type: str
    timestamp: float
    duration: float
    frames: List[Dict] = field(default_factory=list)
    hand_positions: List[Tuple[float, float]] = field(default_factory=list)
    hand_sizes: List[float] = field(default_factory=list)
    velocities: List[Tuple[float, float]] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


class ScreenController:
    """Control mouse cursor based on hand position with advanced gesture support."""

    def __init__(self, screen_w=None, screen_h=None, smoothing=0.6):
        """Initialize screen controller.

        Args:
            screen_w: Screen width (auto-detect if None)
            screen_h: Screen height (auto-detect if None)
            smoothing: Exponential moving average smoothing factor (0-1)
        """
        if screen_w is None or screen_h is None:
            screen_size = pyautogui.size()
            self.screen_w = screen_w or screen_size[0]
            self.screen_h = screen_h or screen_size[1]
        else:
            self.screen_w = screen_w
            self.screen_h = screen_h

        self.smoothing = smoothing
        self.last_x = None
        self.last_y = None

        # Disable pyautogui safety (corners of screen won't stop movement)
        pyautogui.FAILSAFE = False

        # ===== Gesture State Manager =====
        self.gesture_state = GestureState.IDLE
        self.current_gesture_type = GestureType.IDLE
        self.gesture_start_time = None
        self.gesture_confirmation_time = None
        self.debounce_time = 0.1  # seconds
        self.last_gesture_time = None
        self.confirmation_threshold = 0.3  # seconds
        self.gesture_history = deque(maxlen=20)

        # ===== Cursor Control Parameters =====
        self.dead_zone_radius = 30  # pixels - prevent drift near center
        self.acceleration_factor = 1.5  # multiplier for hand velocity
        self.max_cursor_speed = 500  # pixels/second limit
        self.boundary_mode = "snap"  # "snap" or "circular"
        self.circular_boundary_radius = 100  # pixels from center

        # ===== Screen Regions =====
        self.region_size = 0.15  # 15% of screen width/height for edge regions
        self.region_actions: Dict[ScreenRegion, List[str]] = {
            ScreenRegion.TOP: ["maximize", "minimize"],
            ScreenRegion.BOTTOM: ["close_app"],
            ScreenRegion.LEFT: ["switch_left"],
            ScreenRegion.RIGHT: ["switch_right"],
            ScreenRegion.TOP_LEFT: ["prev_workspace"],
            ScreenRegion.TOP_RIGHT: ["next_workspace"],
        }
        self.current_region = ScreenRegion.CENTER

        # ===== Gesture Detection Parameters =====
        self.pinch_distance_threshold = 30  # pixels for pinch detection
        self.palm_open_threshold = 100  # pixels between fingers for open palm
        self.swipe_min_distance = 100  # pixels minimum for swipe
        self.swipe_max_duration = 0.5  # seconds
        self.scroll_speed_factor = 0.1  # multiplier for scroll velocity

        # ===== Recording Mode =====
        self.recording_enabled = False
        self.current_recording: Optional[GestureRecording] = None
        self.recordings: Dict[str, List[GestureRecording]] = {}
        self.recording_dir = None

        # ===== Gesture Callbacks =====
        self.gesture_callbacks: Dict[GestureType, Callable] = {}

    def hand_to_screen(self, hand_data, frame_w, frame_h, invert_x=False):
        """Convert hand palm center position to screen coordinates.

        Args:
            hand_data: Hand data dict from get_hand_landmarks
            frame_w: Frame width
            frame_h: Frame height
            invert_x: Flip horizontal axis (mirror)

        Returns:
            screen_x, screen_y: Screen coordinates
        """
        palm_x, palm_y = hand_data['palm_center']

        # Normalize to 0-1
        norm_x = palm_x / frame_w
        norm_y = palm_y / frame_h

        if invert_x:
            norm_x = 1 - norm_x

        # Map to screen coordinates
        screen_x = int(norm_x * self.screen_w)
        screen_y = int(norm_y * self.screen_h)

        return screen_x, screen_y

    def smooth_position(self, x, y):
        """Apply exponential moving average smoothing.

        Args:
            x, y: New position

        Returns:
            smoothed_x, smoothed_y: Smoothed position
        """
        if self.last_x is None:
            self.last_x = x
            self.last_y = y
            return x, y

        smoothed_x = int(self.last_x * self.smoothing + x * (1 - self.smoothing))
        smoothed_y = int(self.last_y * self.smoothing + y * (1 - self.smoothing))

        self.last_x = smoothed_x
        self.last_y = smoothed_y

        return smoothed_x, smoothed_y

    def move_mouse(self, hand_data, frame_w, frame_h, smooth=True, invert_x=False):
        """Move mouse cursor to hand position.

        Args:
            hand_data: Hand data dict
            frame_w: Frame width
            frame_h: Frame height
            smooth: Apply smoothing
            invert_x: Flip horizontal axis
        """
        screen_x, screen_y = self.hand_to_screen(hand_data, frame_w, frame_h, invert_x)

        if smooth:
            screen_x, screen_y = self.smooth_position(screen_x, screen_y)

        pyautogui.moveTo(screen_x, screen_y, duration=0)

    def click(self, button='left', duration=0.1):
        """Perform mouse click.

        Args:
            button: 'left', 'right', or 'middle'
            duration: Click duration
        """
        pyautogui.click(button=button, duration=duration)

    def drag(self, start_x, start_y, end_x, end_y, duration=0.5):
        """Drag from one position to another.

        Args:
            start_x, start_y: Starting position
            end_x, end_y: Ending position
            duration: Duration of drag
        """
        pyautogui.moveTo(start_x, start_y, duration=0)
        pyautogui.drag(end_x - start_x, end_y - start_y, duration=duration)

    def reset_smoothing(self):
        """Reset smoothing state."""
        self.last_x = None
        self.last_y = None

    # ===== Gesture Detection Methods =====

    def detect_point_gesture(self, hand_data: Dict, threshold: float = 0.7) -> bool:
        """Detect point gesture (index finger extended, others curled).

        Args:
            hand_data: Hand data dict from detector
            threshold: Confidence threshold (0-1)

        Returns:
            is_pointing: True if point gesture detected
        """
        if not hand_data or 'landmarks' not in hand_data or len(hand_data['landmarks']) < 9:
            return False

        try:
            # Get finger tips and bases
            thumb_tip = np.array(hand_data['landmarks'][4]['pixel'])
            index_tip = np.array(hand_data['landmarks'][8]['pixel'])
            index_pip = np.array(hand_data['landmarks'][6]['pixel'])
            middle_tip = np.array(hand_data['landmarks'][12]['pixel'])
            middle_pip = np.array(hand_data['landmarks'][10]['pixel'])
            ring_tip = np.array(hand_data['landmarks'][16]['pixel'])
            ring_pip = np.array(hand_data['landmarks'][14]['pixel'])

            # Index finger extended (tip far from PIP)
            index_extended = np.linalg.norm(index_tip - index_pip) > 30

            # Other fingers curled (tip close to PIP)
            middle_curled = np.linalg.norm(middle_tip - middle_pip) < 25
            ring_curled = np.linalg.norm(ring_tip - ring_pip) < 25

            return index_extended and middle_curled and ring_curled
        except (IndexError, KeyError):
            return False

    def detect_pinch_gesture(self, hand_data: Dict, threshold: float = 30) -> bool:
        """Detect pinch gesture (thumb and index finger close together).

        Args:
            hand_data: Hand data dict from detector
            threshold: Distance threshold in pixels (lower = more pinched)

        Returns:
            is_pinching: True if pinch detected
        """
        if not hand_data or 'landmarks' not in hand_data or len(hand_data['landmarks']) < 9:
            return False

        try:
            thumb_tip = np.array(hand_data['landmarks'][4]['pixel'])
            index_tip = np.array(hand_data['landmarks'][8]['pixel'])
            distance = np.linalg.norm(thumb_tip - index_tip)
            return distance < threshold
        except (IndexError, KeyError):
            return False

    def detect_palm_open_gesture(self, hand_data: Dict, threshold: float = 100) -> bool:
        """Detect open palm gesture (all fingers spread).

        Args:
            hand_data: Hand data dict from detector
            threshold: Minimum distance between fingers

        Returns:
            is_open: True if palm open detected
        """
        if not hand_data or 'landmarks' not in hand_data or len(hand_data['landmarks']) < 20:
            return False

        try:
            # Get finger tips
            thumb_tip = np.array(hand_data['landmarks'][4]['pixel'])
            index_tip = np.array(hand_data['landmarks'][8]['pixel'])
            middle_tip = np.array(hand_data['landmarks'][12]['pixel'])
            ring_tip = np.array(hand_data['landmarks'][16]['pixel'])
            pinky_tip = np.array(hand_data['landmarks'][20]['pixel'])

            # Check spread between fingers
            thumb_to_index = np.linalg.norm(thumb_tip - index_tip)
            index_to_middle = np.linalg.norm(index_tip - middle_tip)
            middle_to_ring = np.linalg.norm(middle_tip - ring_tip)
            ring_to_pinky = np.linalg.norm(ring_tip - pinky_tip)

            avg_spread = np.mean([thumb_to_index, index_to_middle, middle_to_ring, ring_to_pinky])
            return avg_spread > threshold
        except (IndexError, KeyError):
            return False

    def detect_swipe_gesture(self, hand_trajectory: List[Tuple[float, float]],
                            min_distance: float = 100,
                            max_duration: float = 0.5) -> Optional[str]:
        """Detect swipe gesture direction from hand trajectory.

        Args:
            hand_trajectory: List of (x, y) positions from tracker
            min_distance: Minimum pixels for valid swipe
            max_duration: Maximum duration in seconds

        Returns:
            swipe_direction: "left", "right", "up", "down", or None
        """
        if not hand_trajectory or len(hand_trajectory) < 5:
            return None

        try:
            start_pos = np.array(hand_trajectory[0][:2], dtype=np.float32)
            end_pos = np.array(hand_trajectory[-1][:2], dtype=np.float32)

            delta = end_pos - start_pos
            distance = np.linalg.norm(delta)

            if distance < min_distance:
                return None

            # Determine primary direction
            abs_x = abs(delta[0])
            abs_y = abs(delta[1])

            if abs_x > abs_y:
                return "left" if delta[0] < 0 else "right"
            else:
                return "up" if delta[1] < 0 else "down"
        except (ValueError, IndexError):
            return None

    def detect_two_finger_pinch(self, hands: List[Dict], distance_threshold: float = 50) -> bool:
        """Detect two-finger pinch between two hands.

        Args:
            hands: List of hand data dicts
            distance_threshold: Maximum distance between hands in pixels

        Returns:
            is_pinching: True if hands are close enough
        """
        if len(hands) < 2:
            return False

        try:
            hand1_palm = np.array(hands[0]['palm_center'], dtype=np.float32)
            hand2_palm = np.array(hands[1]['palm_center'], dtype=np.float32)
            distance = np.linalg.norm(hand1_palm - hand2_palm)
            return distance < distance_threshold
        except (KeyError, IndexError):
            return False

    # ===== Gesture State Manager =====

    def update_gesture_state(self, detected_gesture: GestureType,
                            hand_data: Optional[Dict] = None,
                            confidence: float = 1.0) -> GestureState:
        """Update gesture state machine with detected gesture.

        Args:
            detected_gesture: Detected gesture type
            hand_data: Optional hand data for additional context
            confidence: Confidence of detection (0-1)

        Returns:
            new_state: Updated gesture state
        """
        current_time = time.time()

        # Debounce check
        if self.last_gesture_time is not None:
            time_since_last = current_time - self.last_gesture_time
            if time_since_last < self.debounce_time:
                return self.gesture_state

        # State machine transitions
        if self.gesture_state == GestureState.IDLE:
            if detected_gesture == GestureType.POINT:
                self.gesture_state = GestureState.POINTING
                self.current_gesture_type = detected_gesture
            elif detected_gesture == GestureType.PINCH:
                self.gesture_state = GestureState.PINCHING
                self.current_gesture_type = detected_gesture
            elif detected_gesture == GestureType.PALM_OPEN:
                self.gesture_state = GestureState.DRAGGING
                self.current_gesture_type = detected_gesture
            elif detected_gesture in [GestureType.SWIPE_LEFT, GestureType.SWIPE_RIGHT,
                                      GestureType.SWIPE_UP, GestureType.SWIPE_DOWN]:
                self.gesture_state = GestureState.SWIPING
                self.current_gesture_type = detected_gesture
            elif detected_gesture == GestureType.TWO_FINGER_PINCH:
                self.gesture_state = GestureState.SCROLLING
                self.current_gesture_type = detected_gesture

            self.gesture_start_time = current_time

        elif self.gesture_state == GestureState.POINTING:
            if detected_gesture != GestureType.POINT:
                self.gesture_state = GestureState.IDLE
                self.current_gesture_type = GestureType.IDLE

        elif self.gesture_state == GestureState.PINCHING:
            if detected_gesture != GestureType.PINCH:
                self.gesture_state = GestureState.IDLE
                self.current_gesture_type = GestureType.IDLE

        elif self.gesture_state == GestureState.DRAGGING:
            if detected_gesture != GestureType.PALM_OPEN:
                self.gesture_state = GestureState.IDLE
                self.current_gesture_type = GestureType.IDLE

        # Add to history
        self.gesture_history.append({
            'gesture': detected_gesture,
            'state': self.gesture_state,
            'timestamp': current_time,
            'confidence': confidence
        })

        self.last_gesture_time = current_time

        # Invoke callback if registered
        if detected_gesture in self.gesture_callbacks:
            self.gesture_callbacks[detected_gesture](hand_data)

        return self.gesture_state

    def set_debounce_time(self, debounce_ms: float) -> None:
        """Set debounce time to prevent accidental triggers.

        Args:
            debounce_ms: Debounce duration in milliseconds
        """
        self.debounce_time = debounce_ms / 1000.0

    def set_confirmation_threshold(self, threshold_ms: float) -> None:
        """Set confirmation threshold for gesture actions.

        Args:
            threshold_ms: Confirmation duration in milliseconds
        """
        self.confirmation_threshold = threshold_ms / 1000.0

    def register_gesture_callback(self, gesture_type: GestureType,
                                  callback: Callable) -> None:
        """Register a callback function for a gesture type.

        Args:
            gesture_type: Gesture type to trigger callback
            callback: Function to call (receives hand_data dict)
        """
        if not callable(callback):
            raise ValueError("Callback must be callable")
        self.gesture_callbacks[gesture_type] = callback

    # ===== Screen Region Detection =====

    def get_screen_region(self, screen_x: int, screen_y: int) -> ScreenRegion:
        """Determine which screen region a position is in.

        Args:
            screen_x: Screen x coordinate
            screen_y: Screen y coordinate

        Returns:
            region: ScreenRegion enum value
        """
        edge_x = int(self.screen_w * self.region_size)
        edge_y = int(self.screen_h * self.region_size)

        is_left = screen_x < edge_x
        is_right = screen_x > self.screen_w - edge_x
        is_top = screen_y < edge_y
        is_bottom = screen_y > self.screen_h - edge_y

        if is_top and is_left:
            return ScreenRegion.TOP_LEFT
        elif is_top and is_right:
            return ScreenRegion.TOP_RIGHT
        elif is_bottom and is_left:
            return ScreenRegion.BOTTOM_LEFT
        elif is_bottom and is_right:
            return ScreenRegion.BOTTOM_RIGHT
        elif is_top:
            return ScreenRegion.TOP
        elif is_bottom:
            return ScreenRegion.BOTTOM
        elif is_left:
            return ScreenRegion.LEFT
        elif is_right:
            return ScreenRegion.RIGHT
        else:
            return ScreenRegion.CENTER

    def set_region_action(self, region: ScreenRegion, actions: List[str]) -> None:
        """Set actions for a screen region.

        Args:
            region: ScreenRegion to configure
            actions: List of action names
        """
        self.region_actions[region] = actions

    def get_region_actions(self, region: ScreenRegion) -> List[str]:
        """Get actions configured for a screen region.

        Args:
            region: ScreenRegion to query

        Returns:
            actions: List of action names for region
        """
        return self.region_actions.get(region, [])

    # ===== Advanced Cursor Control =====

    def apply_acceleration(self, velocity_x: float, velocity_y: float) -> Tuple[float, float]:
        """Apply acceleration mapping to cursor movement.

        Args:
            velocity_x: Velocity in x direction (pixels/second)
            velocity_y: Velocity in y direction (pixels/second)

        Returns:
            accelerated_velocity: (accel_x, accel_y) tuple
        """
        speed = np.sqrt(velocity_x**2 + velocity_y**2)

        # Apply acceleration factor
        if speed > 0:
            accel_speed = speed * self.acceleration_factor
            # Clamp to max speed
            accel_speed = min(accel_speed, self.max_cursor_speed)

            # Normalize and scale
            direction_x = velocity_x / speed if speed > 0 else 0
            direction_y = velocity_y / speed if speed > 0 else 0

            return (direction_x * accel_speed, direction_y * accel_speed)
        return (0, 0)

    def apply_dead_zone(self, screen_x: int, screen_y: int,
                       center_x: Optional[int] = None,
                       center_y: Optional[int] = None) -> Tuple[int, int]:
        """Apply dead zone to prevent cursor drift near center.

        Args:
            screen_x: Current x position
            screen_y: Current y position
            center_x: Center x (default: screen center)
            center_y: Center y (default: screen center)

        Returns:
            filtered_position: (x, y) tuple with dead zone applied
        """
        if center_x is None:
            center_x = self.screen_w // 2
        if center_y is None:
            center_y = self.screen_h // 2

        dx = screen_x - center_x
        dy = screen_y - center_y
        distance = np.sqrt(dx**2 + dy**2)

        if distance < self.dead_zone_radius:
            return (center_x, center_y)

        return (screen_x, screen_y)

    def apply_boundary_behavior(self, screen_x: int, screen_y: int) -> Tuple[int, int]:
        """Apply boundary behavior to cursor movement.

        Supports "snap" (clamp to screen) and "circular" (bounce at boundary).

        Args:
            screen_x: Current x position
            screen_y: Current y position

        Returns:
            bounded_position: (x, y) tuple with boundaries applied
        """
        if self.boundary_mode == "snap":
            # Clamp to screen bounds
            x = np.clip(screen_x, 0, self.screen_w - 1)
            y = np.clip(screen_y, 0, self.screen_h - 1)
            return (int(x), int(y))

        elif self.boundary_mode == "circular":
            # Create circular boundary around screen center
            center_x = self.screen_w // 2
            center_y = self.screen_h // 2

            dx = screen_x - center_x
            dy = screen_y - center_y
            distance = np.sqrt(dx**2 + dy**2)

            if distance > self.circular_boundary_radius:
                # Scale back to boundary
                direction_x = dx / distance if distance > 0 else 0
                direction_y = dy / distance if distance > 0 else 0

                x = center_x + direction_x * self.circular_boundary_radius
                y = center_y + direction_y * self.circular_boundary_radius
                return (int(x), int(y))

            return (screen_x, screen_y)

        return (screen_x, screen_y)

    def move_cursor_with_acceleration(self, hand_data: Dict, frame_w: int,
                                      frame_h: int, hand_velocity: np.ndarray) -> None:
        """Move cursor with acceleration mapping applied.

        Args:
            hand_data: Hand data dict
            frame_w: Frame width
            frame_h: Frame height
            hand_velocity: Hand velocity vector (pixels/second)
        """
        screen_x, screen_y = self.hand_to_screen(hand_data, frame_w, frame_h)

        # Apply dead zone
        screen_x, screen_y = self.apply_dead_zone(screen_x, screen_y)

        # Apply smoothing
        screen_x, screen_y = self.smooth_position(screen_x, screen_y)

        # Apply boundary behavior
        screen_x, screen_y = self.apply_boundary_behavior(screen_x, screen_y)

        pyautogui.moveTo(screen_x, screen_y, duration=0)

    # ===== Recording Mode =====

    def start_recording(self, gesture_type: str, output_dir: str = "./gesture_recordings") -> None:
        """Start recording a gesture sequence for training.

        Args:
            gesture_type: Name of gesture being recorded
            output_dir: Directory to save recordings

        Returns:
            None
        """
        self.recording_enabled = True
        self.recording_dir = output_dir

        # Create output directory if needed
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Initialize recording
        self.current_recording = GestureRecording(
            gesture_type=gesture_type,
            timestamp=time.time(),
            duration=0.0,
            metadata={'created_at': datetime.now().isoformat()}
        )

        if gesture_type not in self.recordings:
            self.recordings[gesture_type] = []

    def record_gesture_frame(self, hand_data: Dict, hand_velocity: np.ndarray,
                            hand_size: float) -> None:
        """Record a single frame during gesture recording.

        Args:
            hand_data: Hand data dict from detector
            hand_velocity: Hand velocity vector
            hand_size: Hand size metric
        """
        if not self.recording_enabled or self.current_recording is None:
            return

        try:
            palm_x, palm_y = hand_data['palm_center']
            self.current_recording.hand_positions.append((palm_x, palm_y))
            self.current_recording.hand_sizes.append(hand_size)
            self.current_recording.velocities.append((
                float(hand_velocity[0]),
                float(hand_velocity[1])
            ))
            self.current_recording.frames.append({
                'timestamp': time.time(),
                'palm': (palm_x, palm_y),
                'velocity': (float(hand_velocity[0]), float(hand_velocity[1])),
                'size': hand_size
            })
        except (KeyError, IndexError, TypeError):
            pass

    def stop_recording(self, save: bool = True) -> Optional[GestureRecording]:
        """Stop recording gesture and optionally save to file.

        Args:
            save: Whether to save recording to disk

        Returns:
            recording: GestureRecording object or None
        """
        if not self.recording_enabled or self.current_recording is None:
            return None

        self.recording_enabled = False
        recording = self.current_recording
        recording.duration = time.time() - recording.timestamp

        if save:
            self.save_gesture_recording(recording)

        # Store in memory
        if recording.gesture_type in self.recordings:
            self.recordings[recording.gesture_type].append(recording)
        else:
            self.recordings[recording.gesture_type] = [recording]

        self.current_recording = None
        return recording

    def save_gesture_recording(self, recording: GestureRecording,
                              output_dir: Optional[str] = None) -> str:
        """Save a gesture recording to disk as JSON.

        Args:
            recording: GestureRecording to save
            output_dir: Output directory (uses self.recording_dir if None)

        Returns:
            filepath: Path to saved file
        """
        if output_dir is None:
            output_dir = self.recording_dir or "./gesture_recordings"

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Create filename
        timestamp = datetime.fromtimestamp(recording.timestamp).strftime("%Y%m%d_%H%M%S")
        filename = f"{recording.gesture_type}_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)

        # Convert to serializable format
        data = {
            'gesture_type': recording.gesture_type,
            'timestamp': recording.timestamp,
            'duration': recording.duration,
            'hand_positions': recording.hand_positions,
            'hand_sizes': recording.hand_sizes,
            'velocities': recording.velocities,
            'metadata': recording.metadata
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        return filepath

    def load_gesture_recording(self, filepath: str) -> Optional[GestureRecording]:
        """Load a gesture recording from disk.

        Args:
            filepath: Path to gesture recording JSON file

        Returns:
            recording: GestureRecording object or None if file invalid
        """
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            recording = GestureRecording(
                gesture_type=data['gesture_type'],
                timestamp=data['timestamp'],
                duration=data['duration'],
                hand_positions=data['hand_positions'],
                hand_sizes=data['hand_sizes'],
                velocities=data['velocities'],
                metadata=data.get('metadata', {})
            )
            return recording
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            return None

    def load_gesture_recordings_from_dir(self, directory: str) -> Dict[str, List[GestureRecording]]:
        """Load all gesture recordings from a directory.

        Args:
            directory: Directory containing .json recording files

        Returns:
            recordings: Dict mapping gesture_type to list of GestureRecordings
        """
        recordings = {}

        if not os.path.exists(directory):
            return recordings

        for filename in os.listdir(directory):
            if filename.endswith('.json'):
                filepath = os.path.join(directory, filename)
                recording = self.load_gesture_recording(filepath)

                if recording:
                    gesture_type = recording.gesture_type
                    if gesture_type not in recordings:
                        recordings[gesture_type] = []
                    recordings[gesture_type].append(recording)

        self.recordings = recordings
        return recordings

    def get_gesture_statistics(self, gesture_type: str) -> Optional[Dict]:
        """Get statistics from recorded gestures of a type.

        Args:
            gesture_type: Type of gesture to analyze

        Returns:
            stats: Dict with min/max/mean duration, speed, etc. or None
        """
        if gesture_type not in self.recordings or not self.recordings[gesture_type]:
            return None

        recordings = self.recordings[gesture_type]
        durations = [r.duration for r in recordings]
        speeds = []

        for recording in recordings:
            if recording.velocities:
                speeds.extend([
                    np.sqrt(vx**2 + vy**2)
                    for vx, vy in recording.velocities
                ])

        return {
            'count': len(recordings),
            'duration_min': float(np.min(durations)),
            'duration_max': float(np.max(durations)),
            'duration_mean': float(np.mean(durations)),
            'duration_std': float(np.std(durations)),
            'speed_min': float(np.min(speeds)) if speeds else 0,
            'speed_max': float(np.max(speeds)) if speeds else 0,
            'speed_mean': float(np.mean(speeds)) if speeds else 0,
        }

    # ===== Configuration Methods =====

    def set_acceleration_factor(self, factor: float) -> None:
        """Set acceleration factor for cursor movement.

        Args:
            factor: Multiplier for hand velocity (1.0 = no acceleration)
        """
        if factor < 0:
            raise ValueError("Acceleration factor must be non-negative")
        self.acceleration_factor = factor

    def set_dead_zone_radius(self, radius: int) -> None:
        """Set dead zone radius to prevent drift.

        Args:
            radius: Radius in pixels
        """
        if radius < 0:
            raise ValueError("Dead zone radius must be non-negative")
        self.dead_zone_radius = radius

    def set_boundary_mode(self, mode: str) -> None:
        """Set cursor boundary behavior.

        Args:
            mode: "snap" (clamp) or "circular" (bounce)

        Raises:
            ValueError: If mode is invalid
        """
        if mode not in ["snap", "circular"]:
            raise ValueError('Boundary mode must be "snap" or "circular"')
        self.boundary_mode = mode

    def set_gesture_thresholds(self, pinch: float = 30, palm_open: float = 100,
                              swipe_distance: float = 100,
                              swipe_duration: float = 0.5) -> None:
        """Set all gesture detection thresholds.

        Args:
            pinch: Distance threshold for pinch in pixels
            palm_open: Spread threshold for open palm in pixels
            swipe_distance: Minimum distance for swipe in pixels
            swipe_duration: Maximum duration for swipe in seconds
        """
        if any(v < 0 for v in [pinch, palm_open, swipe_distance, swipe_duration]):
            raise ValueError("All thresholds must be non-negative")

        self.pinch_distance_threshold = pinch
        self.palm_open_threshold = palm_open
        self.swipe_min_distance = swipe_distance
        self.swipe_max_duration = swipe_duration

    def get_gesture_history(self, limit: int = 10) -> List[Dict]:
        """Get recent gesture history.

        Args:
            limit: Maximum number of recent gestures to return

        Returns:
            history: List of gesture records
        """
        history = list(self.gesture_history)
        return history[-limit:] if limit > 0 else history

    def clear_gesture_history(self) -> None:
        """Clear gesture history."""
        self.gesture_history.clear()
