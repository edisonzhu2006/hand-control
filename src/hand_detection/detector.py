import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from enum import Enum
import time


class GestureState(Enum):
    """Gesture state machine states."""
    IDLE = "idle"
    PINCHING = "pinching"
    POINTING = "pointing"
    OPEN = "open"
    CLOSED = "closed"
    MOVING = "moving"
    CUSTOM = "custom"


@dataclass
class HandMotionMetrics:
    """Container for hand motion metrics."""
    velocity: np.ndarray  # pixels/second
    acceleration: np.ndarray  # pixels/second^2
    speed: float  # magnitude of velocity
    is_stable: bool  # stability flag
    stability_score: float  # 0-1, higher is more stable
    jitter: float  # measure of tremor


class HandDetector:
    """Real-time hand detection and pose estimation using MediaPipe with advanced tracking."""

    def __init__(self, max_hands=2, confidence=0.5, history_frames=30, fps=30.0):
        """Initialize hand detector.

        Args:
            max_hands: Maximum number of hands to detect (1 or 2)
            confidence: Confidence threshold for detections (0-1)
            history_frames: Number of frames to keep in trajectory history
            fps: Expected frame rate for frame-rate independent calculations
        """
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=confidence,
            min_tracking_confidence=confidence,
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.max_hands = max_hands
        self.history_frames = history_frames
        self.fps = fps
        self.frame_time = 1.0 / fps  # seconds per frame
        self.last_frame_time = time.time()

        # Hand tracking state
        self.hand_trajectories: Dict[int, deque] = {}  # hand_id -> deque of (x, y, z)
        self.hand_confidences: Dict[int, deque] = {}   # hand_id -> deque of confidence scores
        self.hand_sizes: Dict[int, deque] = {}         # hand_id -> deque of hand sizes
        self.hand_velocities: Dict[int, np.ndarray] = {}     # hand_id -> velocity vector
        self.hand_accelerations: Dict[int, np.ndarray] = {}  # hand_id -> acceleration vector
        self.hand_stability_scores: Dict[int, float] = {}    # hand_id -> stability (0-1)
        self.hand_jitter: Dict[int, float] = {}             # hand_id -> jitter magnitude
        self.gesture_states: Dict[int, GestureState] = {}    # hand_id -> current gesture state
        self.hand_timestamps: Dict[int, deque] = {}  # hand_id -> deque of timestamps
        self.hand_reference_size: Dict[int, float] = {}  # hand_id -> reference size for normalization

    def _get_frame_time_delta(self) -> float:
        """Calculate actual frame time delta for frame-rate independent calculations.

        Returns:
            time_delta: Actual time since last frame in seconds
        """
        current_time = time.time()
        time_delta = current_time - self.last_frame_time
        self.last_frame_time = current_time
        return time_delta

    def _calculate_hand_size(self, hand_data: Dict) -> float:
        """Calculate hand size from landmarks.

        Uses palm width (distance from wrist to middle finger MCP) as reference.
        This provides scale-invariant hand measurement.

        Args:
            hand_data: Hand data dict with landmarks

        Returns:
            hand_size: Scalar size metric (pixels)
        """
        if len(hand_data['landmarks']) < 10:
            return 0.0

        # Use wrist to middle finger MCP distance
        wrist = np.array(hand_data['landmarks'][0]['pixel'])
        middle_mcp = np.array(hand_data['landmarks'][9]['pixel'])

        size = float(np.linalg.norm(wrist - middle_mcp))
        return size

    def _calculate_velocity(self, position: np.ndarray, time_delta: float,
                           hand_id: int) -> np.ndarray:
        """Calculate velocity from position trajectory.

        Velocity = change in position / change in time
        Converted to pixels/second for frame-rate independence.

        Args:
            position: Current position (x, y, z)
            time_delta: Time since last frame (seconds)
            hand_id: Hand identifier

        Returns:
            velocity: Velocity vector (pixels/second)
        """
        if hand_id not in self.hand_trajectories or len(self.hand_trajectories[hand_id]) < 2:
            return np.array([0.0, 0.0, 0.0])

        prev_position = np.array(self.hand_trajectories[hand_id][-1])
        delta_pos = position - prev_position

        if time_delta > 0:
            velocity = delta_pos / time_delta
        else:
            velocity = delta_pos / self.frame_time

        return velocity

    def _calculate_acceleration(self, velocity: np.ndarray, time_delta: float,
                               hand_id: int) -> np.ndarray:
        """Calculate acceleration from velocity trajectory.

        Acceleration = change in velocity / change in time
        Converted to pixels/second^2 for frame-rate independence.

        Args:
            velocity: Current velocity vector
            time_delta: Time since last frame (seconds)
            hand_id: Hand identifier

        Returns:
            acceleration: Acceleration vector (pixels/second^2)
        """
        if hand_id not in self.hand_velocities:
            return np.array([0.0, 0.0, 0.0])

        prev_velocity = self.hand_velocities[hand_id]
        delta_vel = velocity - prev_velocity

        if time_delta > 0:
            acceleration = delta_vel / time_delta
        else:
            acceleration = delta_vel / self.frame_time

        return acceleration

    def _calculate_jitter(self, hand_id: int, window_size: int = 5) -> float:
        """Calculate jitter (tremor) from recent velocity changes.

        High jitter indicates shaky/trembling hand. Computed as coefficient of variation
        of velocity magnitude over recent frames.

        Args:
            hand_id: Hand identifier
            window_size: Number of recent velocities to analyze

        Returns:
            jitter: Jitter magnitude (0 = no jitter, higher = more jitter)
        """
        if hand_id not in self.hand_velocities or hand_id not in self.hand_trajectories:
            return 0.0

        trajectory = self.hand_trajectories[hand_id]
        if len(trajectory) < window_size:
            return 0.0

        # Calculate speed magnitudes over recent frames
        speeds = []
        recent = list(trajectory)[-window_size:]

        for i in range(1, len(recent)):
            delta = np.array(recent[i]) - np.array(recent[i-1])
            speed = float(np.linalg.norm(delta))
            speeds.append(speed)

        if len(speeds) < 2:
            return 0.0

        speeds = np.array(speeds)
        mean_speed = np.mean(speeds)

        # Coefficient of variation as jitter metric
        if mean_speed > 0.1:  # Avoid division issues
            jitter = float(np.std(speeds) / mean_speed)
        else:
            jitter = 0.0

        return jitter

    def _calculate_stability_score(self, hand_id: int, detection_confidence: float,
                                   jitter: float) -> float:
        """Calculate overall hand stability score.

        Combines detection confidence and inverse of jitter to produce a 0-1 score.
        Higher score = more stable and confident.

        Args:
            hand_id: Hand identifier
            detection_confidence: Detection confidence (0-1)
            jitter: Jitter magnitude

        Returns:
            stability_score: 0-1 score (1 = very stable)
        """
        # Jitter contributes inversely to stability (e.g., jitter 0.5 -> weight 0.67)
        jitter_stability = 1.0 / (1.0 + jitter)

        # Combine confidence and jitter inverse
        stability = 0.6 * detection_confidence + 0.4 * jitter_stability

        return float(np.clip(stability, 0.0, 1.0))

    def update_tracking(self, hands: List[Dict], detection_confidences: Optional[List[float]] = None) -> None:
        """Update hand tracking state with new detections.

        Maintains trajectory history, calculates velocity/acceleration, updates stability scores.
        Call this after get_hand_landmarks() to track hand motion metrics.

        Args:
            hands: List of hand data dicts from get_hand_landmarks()
            detection_confidences: Optional list of detection confidence scores per hand
        """
        time_delta = self._get_frame_time_delta()

        # Handle case where hands are no longer detected
        if not hands:
            # Could optionally fade out old trajectories here
            return

        for hand_idx, hand_data in enumerate(hands):
            hand_id = hand_idx  # Could also use handedness for more robust ID

            # Get palm center position
            palm_center = hand_data.get('palm_center')
            if palm_center is None:
                continue

            position = np.array([palm_center[0], palm_center[1], 0.0], dtype=np.float32)

            # Initialize tracking structures if needed
            if hand_id not in self.hand_trajectories:
                self.hand_trajectories[hand_id] = deque(maxlen=self.history_frames)
                self.hand_confidences[hand_id] = deque(maxlen=self.history_frames)
                self.hand_sizes[hand_id] = deque(maxlen=self.history_frames)
                self.hand_timestamps[hand_id] = deque(maxlen=self.history_frames)
                self.gesture_states[hand_id] = GestureState.IDLE
                self.hand_velocities[hand_id] = np.array([0.0, 0.0, 0.0])
                self.hand_accelerations[hand_id] = np.array([0.0, 0.0, 0.0])

            # Update trajectory
            self.hand_trajectories[hand_id].append(tuple(position))
            self.hand_timestamps[hand_id].append(time.time())

            # Update confidence
            if detection_confidences and hand_idx < len(detection_confidences):
                confidence = detection_confidences[hand_idx]
            else:
                confidence = 0.9  # Default high confidence if not provided

            self.hand_confidences[hand_id].append(confidence)

            # Calculate hand size
            hand_size = self._calculate_hand_size(hand_data)
            self.hand_sizes[hand_id].append(hand_size)

            # Initialize reference size on first detection
            if hand_id not in self.hand_reference_size:
                self.hand_reference_size[hand_id] = hand_size if hand_size > 0 else 50.0

            # Calculate motion metrics
            velocity = self._calculate_velocity(position, time_delta, hand_id)
            acceleration = self._calculate_acceleration(velocity, time_delta, hand_id)
            jitter = self._calculate_jitter(hand_id)
            stability = self._calculate_stability_score(hand_id, confidence, jitter)

            # Store metrics
            self.hand_velocities[hand_id] = velocity
            self.hand_accelerations[hand_id] = acceleration
            self.hand_jitter[hand_id] = jitter
            self.hand_stability_scores[hand_id] = stability

    def get_hand_confidence(self, hand_id: int) -> float:
        """Get detection confidence for a hand.

        Args:
            hand_id: Hand identifier

        Returns:
            confidence: Average confidence over recent frames (0-1)
        """
        if hand_id not in self.hand_confidences or len(self.hand_confidences[hand_id]) == 0:
            return 0.0

        confidences = list(self.hand_confidences[hand_id])
        return float(np.mean(confidences))

    def get_hand_velocity(self, hand_id: int, frame_independent: bool = True) -> np.ndarray:
        """Get velocity of a hand.

        Args:
            hand_id: Hand identifier
            frame_independent: If True, return in pixels/second; if False, pixels/frame

        Returns:
            velocity: 2D velocity vector (x, y) in pixels/second or pixels/frame
        """
        if hand_id not in self.hand_velocities:
            return np.array([0.0, 0.0])

        vel = self.hand_velocities[hand_id][:2]  # Use only x, y

        if not frame_independent:
            vel = vel * self.frame_time

        return vel

    def get_hand_speed(self, hand_id: int, frame_independent: bool = True) -> float:
        """Get speed (magnitude of velocity) of a hand.

        Args:
            hand_id: Hand identifier
            frame_independent: If True, return in pixels/second; if False, pixels/frame

        Returns:
            speed: Scalar speed magnitude
        """
        vel = self.get_hand_velocity(hand_id, frame_independent=frame_independent)
        return float(np.linalg.norm(vel))

    def get_hand_acceleration(self, hand_id: int, frame_independent: bool = True) -> np.ndarray:
        """Get acceleration of a hand.

        Args:
            hand_id: Hand identifier
            frame_independent: If True, return in pixels/second^2; if False, pixels/frame^2

        Returns:
            acceleration: 2D acceleration vector (x, y)
        """
        if hand_id not in self.hand_accelerations:
            return np.array([0.0, 0.0])

        acc = self.hand_accelerations[hand_id][:2]  # Use only x, y

        if not frame_independent:
            acc = acc * (self.frame_time ** 2)

        return acc

    def get_hand_stability(self, hand_id: int) -> float:
        """Get stability score for a hand (0-1, higher is more stable).

        Combines detection confidence and inverse jitter. A very stable hand
        will have a score close to 1.0.

        Args:
            hand_id: Hand identifier

        Returns:
            stability_score: 0-1 stability score
        """
        if hand_id not in self.hand_stability_scores:
            return 0.0

        return float(self.hand_stability_scores[hand_id])

    def is_hand_stable(self, hand_id: int, threshold: float = 0.7) -> bool:
        """Check if a hand is stable (above stability threshold).

        Args:
            hand_id: Hand identifier
            threshold: Stability threshold (0-1)

        Returns:
            is_stable: True if stability > threshold
        """
        return self.get_hand_stability(hand_id) > threshold

    def get_hand_jitter(self, hand_id: int) -> float:
        """Get jitter (tremor) magnitude for a hand.

        Higher values indicate more shaky/trembling motion.

        Args:
            hand_id: Hand identifier

        Returns:
            jitter: Jitter magnitude (0 = no jitter)
        """
        if hand_id not in self.hand_jitter:
            return 0.0

        return float(self.hand_jitter[hand_id])

    def get_hand_trajectory(self, hand_id: int, last_n: Optional[int] = None) -> List[Tuple[float, float, float]]:
        """Get position trajectory history for a hand.

        Args:
            hand_id: Hand identifier
            last_n: Return only last N positions (None = all)

        Returns:
            trajectory: List of (x, y, z) tuples
        """
        if hand_id not in self.hand_trajectories:
            return []

        trajectory = list(self.hand_trajectories[hand_id])

        if last_n is not None:
            trajectory = trajectory[-last_n:]

        return trajectory

    def normalize_by_hand_size(self, hand_id: int, value: float) -> float:
        """Normalize a value by hand size for scale-invariant metrics.

        Useful for gesture thresholds that should be independent of hand distance.

        Args:
            hand_id: Hand identifier
            value: Value to normalize

        Returns:
            normalized_value: Value normalized by hand size ratio
        """
        if hand_id not in self.hand_reference_size:
            return value

        current_size = self.get_hand_current_size(hand_id)
        reference_size = self.hand_reference_size[hand_id]

        if current_size > 0:
            normalized = value * (reference_size / current_size)
        else:
            normalized = value

        return normalized

    def get_hand_current_size(self, hand_id: int) -> float:
        """Get current hand size.

        Args:
            hand_id: Hand identifier

        Returns:
            hand_size: Current hand size in pixels
        """
        if hand_id not in self.hand_sizes or len(self.hand_sizes[hand_id]) == 0:
            return 0.0

        # Use most recent size
        sizes = list(self.hand_sizes[hand_id])
        return float(sizes[-1])

    def get_hand_motion_metrics(self, hand_id: int) -> Optional[HandMotionMetrics]:
        """Get all motion metrics for a hand in one call.

        Args:
            hand_id: Hand identifier

        Returns:
            metrics: HandMotionMetrics dataclass or None if hand not found
        """
        if hand_id not in self.hand_velocities:
            return None

        velocity = self.get_hand_velocity(hand_id, frame_independent=True)
        acceleration = self.get_hand_acceleration(hand_id, frame_independent=True)
        speed = self.get_hand_speed(hand_id, frame_independent=True)
        is_stable = self.is_hand_stable(hand_id)
        stability = self.get_hand_stability(hand_id)
        jitter = self.get_hand_jitter(hand_id)

        return HandMotionMetrics(
            velocity=velocity,
            acceleration=acceleration,
            speed=speed,
            is_stable=is_stable,
            stability_score=stability,
            jitter=jitter,
        )

    def get_inter_hand_distance(self, hand_data_1: Dict, hand_data_2: Dict) -> float:
        """Calculate distance between two hands (palm centers).

        Args:
            hand_data_1: First hand data dict
            hand_data_2: Second hand data dict

        Returns:
            distance: Euclidean distance in pixels
        """
        palm_1 = np.array(hand_data_1.get('palm_center', [0, 0]))
        palm_2 = np.array(hand_data_2.get('palm_center', [0, 0]))

        distance = float(np.linalg.norm(palm_1 - palm_2))
        return distance

    def get_inter_hand_relative_position(self, hand_data_1: Dict, hand_data_2: Dict) -> Tuple[float, float]:
        """Get relative position of hand_2 w.r.t hand_1 (hand_1 as origin).

        Args:
            hand_data_1: Reference hand (origin)
            hand_data_2: Target hand

        Returns:
            relative_pos: (dx, dy) tuple relative to hand_1
        """
        palm_1 = np.array(hand_data_1.get('palm_center', [0, 0]))
        palm_2 = np.array(hand_data_2.get('palm_center', [0, 0]))

        relative = palm_2 - palm_1
        return (float(relative[0]), float(relative[1]))

    def get_inter_hand_distance_normalized(self, hands: List[Dict],
                                          hand_id_1: int = 0, hand_id_2: int = 1) -> Optional[float]:
        """Get distance between hands normalized by their average size.

        Scale-invariant metric useful for multi-hand gestures.

        Args:
            hands: List of hand data dicts
            hand_id_1: Index of first hand
            hand_id_2: Index of second hand

        Returns:
            normalized_distance: Distance normalized by average hand size, or None
        """
        if len(hands) < 2 or hand_id_1 >= len(hands) or hand_id_2 >= len(hands):
            return None

        distance = self.get_inter_hand_distance(hands[hand_id_1], hands[hand_id_2])

        size_1 = self.get_hand_current_size(hand_id_1) or 50.0
        size_2 = self.get_hand_current_size(hand_id_2) or 50.0
        avg_size = (size_1 + size_2) / 2.0

        if avg_size > 0:
            normalized = distance / avg_size
        else:
            normalized = distance

        return normalized

    def get_gesture_state(self, hand_id: int) -> GestureState:
        """Get current gesture state for a hand.

        Args:
            hand_id: Hand identifier

        Returns:
            gesture_state: Current GestureState enum value
        """
        if hand_id not in self.gesture_states:
            return GestureState.IDLE

        return self.gesture_states[hand_id]

    def set_gesture_state(self, hand_id: int, state: GestureState) -> None:
        """Set gesture state for a hand (for state machine).

        Args:
            hand_id: Hand identifier
            state: New GestureState enum value
        """
        self.gesture_states[hand_id] = state

    def transition_gesture_state(self, hand_id: int, target_state: GestureState,
                                confidence: float = 1.0) -> bool:
        """Attempt a gesture state transition with confidence check.

        Args:
            hand_id: Hand identifier
            target_state: Target GestureState
            confidence: Confidence level for transition (0-1)

        Returns:
            success: True if transition occurred
        """
        current_state = self.get_gesture_state(hand_id)

        # Simple state machine: allow transitions with sufficient confidence
        if confidence >= 0.7:
            self.gesture_states[hand_id] = target_state
            return True

        return False

    def detect(self, frame):
        """Detect hands in frame.

        Args:
            frame: Input image (BGR format from OpenCV)

        Returns:
            results: MediaPipe hand detection results
            frame_h: Frame height
            frame_w: Frame width
            frame_c: Frame channels
        """
        frame_h, frame_w, frame_c = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        return results, frame_h, frame_w, frame_c

    def get_hand_landmarks(self, results, frame_h, frame_w):
        """Extract hand landmarks as normalized and pixel coordinates.

        Args:
            results: MediaPipe detection results
            frame_h: Frame height
            frame_w: Frame width

        Returns:
            hands: List of hand data dicts with landmarks, position, and confidence
        """
        hands = []

        if not results.multi_hand_landmarks:
            return hands

        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Extract detection confidence (handedness classification includes confidence)
            handedness_classification = results.multi_handedness[hand_idx].classification[0]
            handedness_label = handedness_classification.label
            handedness_confidence = float(handedness_classification.score)

            hand_data = {
                'id': hand_idx,
                'landmarks': [],
                'handedness': handedness_label,
                'handedness_confidence': handedness_confidence,
            }

            # Extract all 21 landmarks
            for landmark in hand_landmarks.landmark:
                x_norm = landmark.x
                y_norm = landmark.y
                z_norm = landmark.z

                x_pixel = int(x_norm * frame_w)
                y_pixel = int(y_norm * frame_h)

                hand_data['landmarks'].append({
                    'norm': (x_norm, y_norm, z_norm),
                    'pixel': (x_pixel, y_pixel),
                    'z': z_norm,
                })

            # Calculate palm center (average of wrist and hand center points)
            wrist = hand_data['landmarks'][0]['pixel']
            middle_mcp = hand_data['landmarks'][9]['pixel']
            palm_center = (
                (wrist[0] + middle_mcp[0]) // 2,
                (wrist[1] + middle_mcp[1]) // 2,
            )
            hand_data['palm_center'] = palm_center

            hands.append(hand_data)

        return hands

    def get_pinch_distance(self, hand_data):
        """Calculate distance between thumb and index finger (pinch gesture).

        Args:
            hand_data: Hand data dict from get_hand_landmarks

        Returns:
            distance: Euclidean distance in pixels
        """
        if len(hand_data['landmarks']) < 9:
            return 0

        thumb_tip = hand_data['landmarks'][4]['pixel']
        index_tip = hand_data['landmarks'][8]['pixel']

        distance = np.sqrt(
            (thumb_tip[0] - index_tip[0]) ** 2 +
            (thumb_tip[1] - index_tip[1]) ** 2
        )
        return distance

    def draw_landmarks(self, frame, results):
        """Draw hand landmarks and connections on frame.

        Args:
            frame: Input frame
            results: MediaPipe detection results

        Returns:
            frame: Frame with drawn landmarks
        """
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                )
        return frame

    def reset_hand_tracking(self, hand_id: Optional[int] = None) -> None:
        """Reset tracking state for a hand (useful when hand goes out of frame).

        Args:
            hand_id: Hand identifier to reset, or None to reset all hands
        """
        if hand_id is None:
            # Reset all
            self.hand_trajectories.clear()
            self.hand_confidences.clear()
            self.hand_sizes.clear()
            self.hand_velocities.clear()
            self.hand_accelerations.clear()
            self.hand_stability_scores.clear()
            self.hand_jitter.clear()
            self.hand_timestamps.clear()
            self.gesture_states.clear()
            self.hand_reference_size.clear()
        else:
            # Reset specific hand
            if hand_id in self.hand_trajectories:
                del self.hand_trajectories[hand_id]
            if hand_id in self.hand_confidences:
                del self.hand_confidences[hand_id]
            if hand_id in self.hand_sizes:
                del self.hand_sizes[hand_id]
            if hand_id in self.hand_velocities:
                del self.hand_velocities[hand_id]
            if hand_id in self.hand_accelerations:
                del self.hand_accelerations[hand_id]
            if hand_id in self.hand_stability_scores:
                del self.hand_stability_scores[hand_id]
            if hand_id in self.hand_jitter:
                del self.hand_jitter[hand_id]
            if hand_id in self.hand_timestamps:
                del self.hand_timestamps[hand_id]
            if hand_id in self.gesture_states:
                del self.gesture_states[hand_id]
            if hand_id in self.hand_reference_size:
                del self.hand_reference_size[hand_id]

    def set_fps(self, fps: float) -> None:
        """Update frame rate for frame-rate independent calculations.

        Args:
            fps: New frames per second value
        """
        self.fps = fps
        self.frame_time = 1.0 / fps if fps > 0 else 0.033

    def close(self):
        """Clean up resources."""
        self.reset_hand_tracking()
        self.hands.close()
