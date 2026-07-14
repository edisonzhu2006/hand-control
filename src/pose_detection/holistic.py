"""
Holistic detection: body pose + facial expression + finger gestures in one model.

Wraps MediaPipe Holistic, which runs pose, face mesh, and both hands together.
Body extraction is inherited from PoseDetector (Holistic results expose
pose_landmarks with the same 33 indices). On top of that this adds:

- face_metrics(): smile and mouth-open scores (0..1) from face-mesh geometry
- hand_gestures(): per-screen-side finger gesture ('open', 'fist', 'peace',
  'point', 'other', 'none') classified from fingertip extension
"""

import numpy as np
import mediapipe as mp

from .detector import PoseDetector

# Face-mesh landmark indices (canonical MediaPipe face mesh topology).
LIP_UP, LIP_DOWN = 13, 14
MOUTH_L, MOUTH_R = 61, 291
CHEEK_L, CHEEK_R = 234, 454
FOREHEAD, CHIN = 10, 152


def face_metrics_from_pts(pts):
    """Compute expression scores from a dict of face points (x, y arrays).

    Args:
        pts: Dict with keys lip_up, lip_down, mouth_l, mouth_r, cheek_l,
            cheek_r, forehead, chin — each an (x, y) array.

    Returns:
        {'smile': 0..1, 'open': 0..1}
    """
    face_w = float(np.linalg.norm(pts['cheek_r'] - pts['cheek_l'])) or 1.0
    face_h = float(np.linalg.norm(pts['chin'] - pts['forehead'])) or 1.0

    mouth_w = float(np.linalg.norm(pts['mouth_r'] - pts['mouth_l']))
    width_ratio = mouth_w / face_w                      # ~0.35 neutral, 0.45+ grin
    lip_mid_y = (pts['lip_up'][1] + pts['lip_down'][1]) / 2
    corner_lift = (lip_mid_y - (pts['mouth_l'][1] + pts['mouth_r'][1]) / 2) / face_h
    smile = 0.65 * np.clip((width_ratio - 0.37) / 0.09, 0, 1) + \
        0.35 * np.clip(corner_lift / 0.02, 0, 1)

    gap = float(np.linalg.norm(pts['lip_down'] - pts['lip_up']))
    mouth_open = np.clip((gap / face_h - 0.03) / 0.06, 0, 1)
    return {'smile': float(np.clip(smile, 0, 1)), 'open': float(mouth_open)}


def classify_gesture(pts):
    """Classify a hand gesture from 21 landmark (x, y) points.

    A finger counts as extended when its tip is farther from the wrist than
    its PIP joint (scale/orientation independent).

    Args:
        pts: (21, 2) array of hand landmark positions.

    Returns:
        'open' | 'fist' | 'peace' | 'point' | 'other'
    """
    wrist = pts[0]

    def ext(tip, pip):
        return np.linalg.norm(pts[tip] - wrist) > np.linalg.norm(pts[pip] - wrist)

    index, middle = ext(8, 6), ext(12, 10)
    ring, pinky = ext(16, 14), ext(20, 18)
    n = sum([index, middle, ring, pinky])

    if n == 4:
        return 'open'
    if n == 0:
        return 'fist'
    if index and middle and not ring and not pinky:
        return 'peace'
    if index and not middle and not ring and not pinky:
        return 'point'
    return 'other'


class HolisticDetector(PoseDetector):
    """PoseDetector-compatible wrapper around MediaPipe Holistic."""

    def __init__(self, confidence=0.5, model_complexity=1):
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=confidence,
            min_tracking_confidence=confidence,
            model_complexity=model_complexity,
        )
        self.mp_drawing = mp.solutions.drawing_utils
        # note: intentionally NOT calling super().__init__ — no second model.

    def detect(self, frame):
        import cv2
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.holistic.process(frame_rgb)

    def face_metrics(self, results, frame_w, frame_h):
        """Expression scores from the face mesh, or None if no face."""
        if not results.face_landmarks:
            return None
        lms = results.face_landmarks.landmark

        def p(i):
            return np.array([lms[i].x * frame_w, lms[i].y * frame_h])

        return face_metrics_from_pts({
            'lip_up': p(LIP_UP), 'lip_down': p(LIP_DOWN),
            'mouth_l': p(MOUTH_L), 'mouth_r': p(MOUTH_R),
            'cheek_l': p(CHEEK_L), 'cheek_r': p(CHEEK_R),
            'forehead': p(FOREHEAD), 'chin': p(CHIN),
        })

    def hand_gestures(self, results, frame_w, frame_h, mirrored=False):
        """Per-screen-side gestures: {'l': gesture-or-'none', 'r': ...}.

        Holistic labels hands by the image person's anatomy; on a mirrored
        (selfie) frame the anatomical left hand appears on screen-right.
        """
        def pts(hand_lms):
            return np.array([[lm.x * frame_w, lm.y * frame_h]
                             for lm in hand_lms.landmark])

        left = classify_gesture(pts(results.left_hand_landmarks)) \
            if results.left_hand_landmarks else 'none'
        right = classify_gesture(pts(results.right_hand_landmarks)) \
            if results.right_hand_landmarks else 'none'
        if mirrored:
            return {'l': right, 'r': left}
        return {'l': left, 'r': right}

    def draw(self, frame, results):
        """Skeleton + hands on the preview frame."""
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS)
        for hand in (results.left_hand_landmarks, results.right_hand_landmarks):
            if hand:
                self.mp_drawing.draw_landmarks(
                    frame, hand, self.mp_holistic.HAND_CONNECTIONS)
        return frame

    def close(self):
        self.holistic.close()
