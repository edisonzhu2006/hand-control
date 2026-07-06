"""
Gesture recognition engine for hand pose classification and temporal gesture matching.

Supports both static gestures (hand pose) and dynamic gestures (temporal sequences).
Includes template matching, training, and confidence scoring.
"""

import numpy as np
import json
import pickle
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
from pathlib import Path
import os


class GestureType(Enum):
    """Built-in gesture types."""
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    OK_SIGN = "ok_sign"
    PEACE = "peace"
    ROCK = "rock"
    FIST = "fist"
    OPEN_PALM = "open_palm"
    POINTING = "pointing"
    VICTORY = "victory"
    CALL_ME = "call_me"
    WAVE = "wave"
    SWIPE_LEFT = "swipe_left"
    SWIPE_RIGHT = "swipe_right"
    CUSTOM = "custom"


@dataclass
class GestureTemplate:
    """Template for gesture matching."""
    name: str
    gesture_type: str
    landmarks: np.ndarray  # Shape: (21, 3) for static or (frames, 21, 3) for dynamic
    is_dynamic: bool = False
    confidence_threshold: float = 0.7
    metadata: Dict = field(default_factory=dict)

    def to_dict(self):
        return {
            'name': self.name,
            'gesture_type': self.gesture_type,
            'landmarks': self.landmarks.tolist(),
            'is_dynamic': self.is_dynamic,
            'confidence_threshold': self.confidence_threshold,
            'metadata': self.metadata,
        }

    @classmethod
    def from_dict(cls, data):
        data['landmarks'] = np.array(data['landmarks'])
        return cls(**data)


@dataclass
class GestureMatch:
    """Result of gesture recognition."""
    gesture_name: str
    gesture_type: str
    confidence: float
    distance: float
    matched_template_name: str


class GestureRecognizer:
    """Recognize hand gestures from pose data."""

    def __init__(self, data_dir: str = '/Users/edison.zhu/hand-control/data'):
        """Initialize gesture recognizer.

        Args:
            data_dir: Directory for storing gesture models and templates
        """
        self.data_dir = Path(data_dir)
        self.gestures_dir = self.data_dir / 'gestures'
        self.models_dir = self.data_dir / 'models'

        self.gestures_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.templates: Dict[str, GestureTemplate] = {}
        self.gesture_samples: Dict[str, List[np.ndarray]] = {}

        self._initialize_builtin_gestures()

    def _initialize_builtin_gestures(self):
        """Initialize built-in gesture templates."""
        # These are placeholder templates—train with real data for production
        self.templates['thumbs_up'] = GestureTemplate(
            name='thumbs_up',
            gesture_type=GestureType.THUMBS_UP.value,
            landmarks=self._create_placeholder_template('thumbs_up'),
            is_dynamic=False,
        )
        self.templates['thumbs_down'] = GestureTemplate(
            name='thumbs_down',
            gesture_type=GestureType.THUMBS_DOWN.value,
            landmarks=self._create_placeholder_template('thumbs_down'),
            is_dynamic=False,
        )
        self.templates['ok_sign'] = GestureTemplate(
            name='ok_sign',
            gesture_type=GestureType.OK_SIGN.value,
            landmarks=self._create_placeholder_template('ok_sign'),
            is_dynamic=False,
        )
        self.templates['peace'] = GestureTemplate(
            name='peace',
            gesture_type=GestureType.PEACE.value,
            landmarks=self._create_placeholder_template('peace'),
            is_dynamic=False,
        )
        self.templates['fist'] = GestureTemplate(
            name='fist',
            gesture_type=GestureType.FIST.value,
            landmarks=self._create_placeholder_template('fist'),
            is_dynamic=False,
        )
        self.templates['open_palm'] = GestureTemplate(
            name='open_palm',
            gesture_type=GestureType.OPEN_PALM.value,
            landmarks=self._create_placeholder_template('open_palm'),
            is_dynamic=False,
        )

    def _create_placeholder_template(self, gesture_name: str) -> np.ndarray:
        """Create placeholder template for built-in gestures.

        Args:
            gesture_name: Name of gesture

        Returns:
            landmarks: Shape (21, 3) normalized landmark array
        """
        # Return normalized hand landmarks (0-1 range)
        return np.random.randn(21, 3) * 0.1 + 0.5

    def add_gesture(self, name: str, template: np.ndarray, gesture_type: str = 'custom',
                   is_dynamic: bool = False, confidence_threshold: float = 0.7):
        """Add gesture template manually.

        Args:
            name: Unique gesture name
            template: Landmark array (21, 3) for static or (frames, 21, 3) for dynamic
            gesture_type: Type of gesture
            is_dynamic: Whether gesture is temporal
            confidence_threshold: Confidence needed to match
        """
        self.templates[name] = GestureTemplate(
            name=name,
            gesture_type=gesture_type,
            landmarks=template,
            is_dynamic=is_dynamic,
            confidence_threshold=confidence_threshold,
        )

    def train_gesture(self, name: str, samples: List[np.ndarray], gesture_type: str = 'custom',
                     is_dynamic: bool = False):
        """Train gesture model from samples.

        Creates average template from multiple samples.

        Args:
            name: Gesture name
            samples: List of landmark arrays (each shape (21, 3) or (frames, 21, 3))
            gesture_type: Type of gesture
            is_dynamic: Whether gesture is temporal
        """
        if not samples:
            raise ValueError("No samples provided for training")

        self.gesture_samples[name] = samples

        # Average templates to create model
        if is_dynamic:
            # For dynamic gestures, align sequences first
            aligned_samples = self._align_temporal_sequences(samples)
            avg_template = np.mean(aligned_samples, axis=0)
        else:
            # For static gestures, simple averaging
            avg_template = np.mean(samples, axis=0)

        self.templates[name] = GestureTemplate(
            name=name,
            gesture_type=gesture_type,
            landmarks=avg_template,
            is_dynamic=is_dynamic,
            confidence_threshold=0.7,
        )

    def recognize(self, hand_data: Dict, top_k: int = 1) -> List[GestureMatch]:
        """Recognize a static hand gesture using geometric finger-state rules.

        Classifies the pose directly from the 21 MediaPipe landmarks by working
        out which fingers are extended and how the thumb is oriented, then mapping
        that pattern to a built-in gesture. All tests are relative distances, so
        this is robust to hand scale and position (unlike matching against the
        placeholder templates, which were random noise).

        Args:
            hand_data: Hand landmark data from detector
            top_k: Maximum number of matches to return

        Returns:
            matches: List of GestureMatch results, highest confidence first
        """
        if not hand_data or 'landmarks' not in hand_data or len(hand_data['landmarks']) < 21:
            return []

        pts = np.array([lm['pixel'] for lm in hand_data['landmarks']], dtype=np.float64)
        match = self._classify_static(pts)
        return [match][:top_k] if match else []

    def _finger_extended_states(self, pts: np.ndarray) -> Tuple[bool, Dict[str, bool]]:
        """Determine which fingers are extended from landmark geometry.

        A finger is 'extended' when its tip is farther from the wrist than the
        joint below it (PIP for the four fingers, MCP for the thumb). When a
        finger curls, the tip folds back toward the palm and sits closer to the
        wrist than the protruding middle knuckle, so the comparison flips. This
        is orientation- and scale-independent.

        Args:
            pts: (21, 2) array of landmark pixel coordinates

        Returns:
            (thumb_extended, {finger_name: extended})
        """
        def d(i, j):
            return float(np.linalg.norm(pts[i] - pts[j]))

        fingers = {
            'index':  d(8, 0) > d(6, 0),
            'middle': d(12, 0) > d(10, 0),
            'ring':   d(16, 0) > d(14, 0),
            'pinky':  d(20, 0) > d(18, 0),
        }
        thumb = d(4, 0) > d(2, 0)
        return thumb, fingers

    def _classify_static(self, pts: np.ndarray) -> Optional[GestureMatch]:
        """Map finger states and thumb orientation to a built-in gesture.

        Args:
            pts: (21, 2) array of landmark pixel coordinates

        Returns:
            A GestureMatch for the recognized gesture, or None if the pose does
            not confidently match any built-in gesture.
        """
        wrist = pts[0]
        palm_size = float(np.linalg.norm(pts[9] - wrist)) or 1.0
        _, f = self._finger_extended_states(pts)
        n_ext = sum(f.values())

        thumb_index_gap = float(np.linalg.norm(pts[4] - pts[8])) / palm_size
        thumb_dy = (pts[4][1] - wrist[1]) / palm_size  # < 0: thumb tip above wrist

        def make(name: str, conf: float) -> GestureMatch:
            conf = float(np.clip(conf, 0.0, 1.0))
            gtype = self.templates[name].gesture_type if name in self.templates else name
            return GestureMatch(
                gesture_name=name,
                gesture_type=gtype,
                confidence=conf,
                distance=1.0 - conf,
                matched_template_name=name,
            )

        # OK sign: thumb & index tips pinched together, other three fingers up
        if thumb_index_gap < 0.4 and f['middle'] and f['ring'] and f['pinky']:
            return make('ok_sign', 0.95 - thumb_index_gap)

        # Open palm: all four fingers extended
        if n_ext == 4:
            return make('open_palm', 0.92)

        # Peace / victory: index + middle up, ring + pinky curled
        if f['index'] and f['middle'] and not f['ring'] and not f['pinky']:
            return make('peace', 0.9)

        # Closed hand: no fingers extended -> fist / thumbs up / thumbs down by thumb direction
        if n_ext == 0:
            if thumb_dy < -0.6:
                return make('thumbs_up', 0.9)
            if thumb_dy > 0.6:
                return make('thumbs_down', 0.9)
            return make('fist', 0.9)

        return None

    def recognize_temporal(self, hand_trajectory: List[Dict], top_k: int = 1) -> List[GestureMatch]:
        """Recognize dynamic gesture from hand trajectory.

        Args:
            hand_trajectory: List of hand data dicts over time
            top_k: Return top K matches

        Returns:
            matches: List of GestureMatch results
        """
        if not hand_trajectory:
            return []

        # Extract landmark sequence
        landmark_sequence = np.array([
            self._extract_landmarks_array(hand_data)
            for hand_data in hand_trajectory
        ])

        matches = []
        for template_name, template in self.templates.items():
            if not template.is_dynamic:
                continue

            # Align sequences using DTW
            distance = self._dtw_distance(landmark_sequence, template.landmarks)
            confidence = self._distance_to_confidence(distance, max_distance=100)

            if confidence >= template.confidence_threshold:
                matches.append(GestureMatch(
                    gesture_name=template.name,
                    gesture_type=template.gesture_type,
                    confidence=confidence,
                    distance=distance,
                    matched_template_name=template_name,
                ))

        matches.sort(key=lambda m: m.confidence, reverse=True)
        return matches[:top_k]

    def _extract_landmarks_array(self, hand_data: Dict) -> np.ndarray:
        """Extract normalized landmarks from hand data.

        Args:
            hand_data: Hand data dict

        Returns:
            landmarks: Array of shape (21, 3) with normalized coordinates
        """
        landmarks = []
        for lm in hand_data['landmarks']:
            x, y, z = lm['norm']
            landmarks.append([x, y, z])

        return np.array(landmarks)

    def _compute_distance(self, landmarks1: np.ndarray, landmarks2: np.ndarray) -> float:
        """Compute Euclidean distance between landmark arrays.

        Args:
            landmarks1: Shape (21, 3)
            landmarks2: Shape (21, 3)

        Returns:
            distance: Mean Euclidean distance
        """
        diff = landmarks1 - landmarks2
        distances = np.linalg.norm(diff, axis=1)
        return float(np.mean(distances))

    def _dtw_distance(self, sequence1: np.ndarray, sequence2: np.ndarray) -> float:
        """Compute Dynamic Time Warping distance between sequences.

        Args:
            sequence1: Shape (frames1, 21, 3)
            sequence2: Shape (frames2, 21, 3)

        Returns:
            distance: DTW distance
        """
        n, m = len(sequence1), len(sequence2)

        # Frame-wise distances
        frame_distances = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                frame_distances[i, j] = self._compute_distance(sequence1[i], sequence2[j])

        # DTW algorithm
        dtw_matrix = np.full((n + 1, m + 1), np.inf)
        dtw_matrix[0, 0] = 0

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = frame_distances[i - 1, j - 1]
                dtw_matrix[i, j] = cost + min(
                    dtw_matrix[i - 1, j],
                    dtw_matrix[i, j - 1],
                    dtw_matrix[i - 1, j - 1],
                )

        return float(dtw_matrix[n, m] / (n + m))

    def _distance_to_confidence(self, distance: float, max_distance: float = 0.5) -> float:
        """Convert distance to confidence score.

        Args:
            distance: Distance metric
            max_distance: Distance at which confidence = 0

        Returns:
            confidence: Score from 0-1
        """
        confidence = max(0, 1 - distance / max_distance)
        return float(np.clip(confidence, 0, 1))

    def _align_temporal_sequences(self, sequences: List[np.ndarray]) -> np.ndarray:
        """Align variable-length temporal sequences to common length.

        Args:
            sequences: List of arrays with shape (frames, 21, 3)

        Returns:
            aligned: Array with shape (num_sequences, frames, 21, 3)
        """
        # Find common length (median)
        lengths = [len(seq) for seq in sequences]
        common_length = int(np.median(lengths))

        aligned = []
        for seq in sequences:
            if len(seq) == common_length:
                aligned.append(seq)
            else:
                # Resample to common length
                indices = np.linspace(0, len(seq) - 1, common_length)
                resampled = np.array([
                    seq[int(idx)] if int(idx) < len(seq) else seq[-1]
                    for idx in indices
                ])
                aligned.append(resampled)

        return np.array(aligned)

    def save_gesture(self, name: str, filepath: Optional[str] = None):
        """Save trained gesture to disk.

        Args:
            name: Gesture name
            filepath: Save path (default: data/models/{name}.pkl)
        """
        if name not in self.templates:
            raise ValueError(f"Gesture '{name}' not found")

        if filepath is None:
            filepath = self.models_dir / f"{name}.pkl"

        with open(filepath, 'wb') as f:
            pickle.dump(self.templates[name], f)

    def load_gesture(self, name: str, filepath: Optional[str] = None):
        """Load gesture from disk.

        Args:
            name: Gesture name
            filepath: Load path (default: data/models/{name}.pkl)
        """
        if filepath is None:
            filepath = self.models_dir / f"{name}.pkl"

        if not Path(filepath).exists():
            raise FileNotFoundError(f"Gesture file not found: {filepath}")

        with open(filepath, 'rb') as f:
            template = pickle.load(f)

        self.templates[name] = template

    def list_gestures(self) -> List[str]:
        """List all available gesture names.

        Returns:
            gesture_names: List of gesture names
        """
        return list(self.templates.keys())

    def get_gesture_info(self, name: str) -> Dict[str, Any]:
        """Get information about a gesture.

        Args:
            name: Gesture name

        Returns:
            info: Gesture information dict
        """
        if name not in self.templates:
            raise ValueError(f"Gesture '{name}' not found")

        template = self.templates[name]
        return {
            'name': template.name,
            'type': template.gesture_type,
            'is_dynamic': template.is_dynamic,
            'confidence_threshold': template.confidence_threshold,
            'num_samples': len(self.gesture_samples.get(name, [])),
        }
