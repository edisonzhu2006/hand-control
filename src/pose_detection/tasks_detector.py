"""
High-accuracy face + finger tracking via MediaPipe Tasks models.

Replaces Holistic's pose-cropped face/hands with dedicated learned models:

- FaceLandmarker with blendshapes: real expression scores (mouthSmile,
  jawOpen) instead of geometric lip heuristics
- GestureRecognizer: a trained gesture classifier (Open_Palm, Closed_Fist,
  Victory, Pointing_Up, ...) with the geometric fingertip classifier as a
  fallback for orientations the canned model rejects, and a majority vote
  over recent frames so gestures don't flicker

Body tracking stays on the proven legacy Pose solution (inherited from
PoseDetector), so get_body()/draw() behave exactly as before.
"""

import collections
import os
import urllib.request

import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

from .detector import PoseDetector
from .holistic import classify_gesture

MODEL_DIR = '/Users/edison.zhu/hand-control/data/models'
MODEL_URLS = {
    'face_landmarker.task':
        'https://storage.googleapis.com/mediapipe-models/face_landmarker/'
        'face_landmarker/float16/latest/face_landmarker.task',
    'gesture_recognizer.task':
        'https://storage.googleapis.com/mediapipe-models/gesture_recognizer/'
        'gesture_recognizer/float16/latest/gesture_recognizer.task',
}

GESTURE_MAP = {
    'Open_Palm': 'open',
    'Closed_Fist': 'fist',
    'Victory': 'peace',
    'Pointing_Up': 'point',
    'Thumb_Up': 'other',
    'Thumb_Down': 'other',
    'ILoveYou': 'other',
}

FACE_SMOOTH = 0.45       # EMA factor for expression scores
GESTURE_VOTES = 5        # majority-vote window per hand


def _model_path(name):
    path = os.path.join(MODEL_DIR, name)
    if not os.path.exists(path):
        os.makedirs(MODEL_DIR, exist_ok=True)
        print(f'Downloading {name}...')
        urllib.request.urlretrieve(MODEL_URLS[name], path)
    return path


class TasksDetector(PoseDetector):
    """Legacy Pose body + Tasks-API face blendshapes and hand gestures."""

    def __init__(self, confidence=0.5, model_complexity=1):
        super().__init__(confidence=confidence, model_complexity=model_complexity)

        self.face_lm = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=mp_tasks.BaseOptions(
                    model_asset_path=_model_path('face_landmarker.task')),
                running_mode=vision.RunningMode.VIDEO,
                output_face_blendshapes=True,
                num_faces=1,
            ))
        self.gesture_rec = vision.GestureRecognizer.create_from_options(
            vision.GestureRecognizerOptions(
                base_options=mp_tasks.BaseOptions(
                    model_asset_path=_model_path('gesture_recognizer.task')),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=2,
            ))

        self._ts_ms = 0
        self._face = {'smile': 0.0, 'open': 0.0, 'blinkL': 0.0, 'blinkR': 0.0}
        self._votes = {'Left': collections.deque(maxlen=GESTURE_VOTES),
                       'Right': collections.deque(maxlen=GESTURE_VOTES)}
        self._gestures = {'Left': 'none', 'Right': 'none'}
        self._fingers = {'Left': None, 'Right': None}   # [thumb..pinky] bools
        self._shapes = {'Left': None, 'Right': None}    # 21 wrist-normalized pts

    def detect(self, frame):
        """Run pose + face + gestures; returns pose results (as before)."""
        import cv2
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self._ts_ms += 33

        self._detect_face(mp_img)
        self._detect_gestures(mp_img, frame.shape)

        return self.pose.process(rgb)

    def _detect_face(self, mp_img):
        res = self.face_lm.detect_for_video(mp_img, self._ts_ms)
        if res.face_blendshapes:
            shapes = {c.category_name: c.score for c in res.face_blendshapes[0]}
            smile = (shapes.get('mouthSmileLeft', 0) +
                     shapes.get('mouthSmileRight', 0)) / 2
            open_ = min(1.0, shapes.get('jawOpen', 0) * 1.6)
            a = FACE_SMOOTH
            self._face = {
                'smile': float(a * smile + (1 - a) * self._face['smile']),
                'open': float(a * open_ + (1 - a) * self._face['open']),
                # blendshape sides are the image person's anatomy
                'blinkL': float(a * shapes.get('eyeBlinkLeft', 0) +
                                (1 - a) * self._face['blinkL']),
                'blinkR': float(a * shapes.get('eyeBlinkRight', 0) +
                                (1 - a) * self._face['blinkR']),
            }
        # keep last values briefly when the face drops out (EMA decays anyway)

    def _detect_gestures(self, mp_img, shape):
        res = self.gesture_rec.recognize_for_video(mp_img, self._ts_ms)
        h, w = shape[:2]
        seen = {'Left': 'none', 'Right': 'none'}
        fingers = {'Left': None, 'Right': None}
        shapes = {'Left': None, 'Right': None}
        for i, handedness in enumerate(res.handedness):
            side = handedness[0].category_name          # image-person anatomy
            pts = np.array([[lm.x * w, lm.y * h]
                            for lm in res.hand_landmarks[i]])
            name = res.gestures[i][0].category_name if res.gestures else 'None'
            g = GESTURE_MAP.get(name)
            if g is None:
                # canned model unsure — fall back to fingertip geometry
                g = classify_gesture(pts)
            seen[side] = g
            # per-finger extension: tip farther from wrist than mid joint
            wrist = pts[0]
            def ext(tip, mid):
                return bool(np.linalg.norm(pts[tip] - wrist) >
                            np.linalg.norm(pts[mid] - wrist))
            fingers[side] = [ext(4, 2), ext(8, 6), ext(12, 10),
                             ext(16, 14), ext(20, 18)]
            # wrist-normalized shape (unit = wrist->middle-MCP) for rendering
            size = float(np.linalg.norm(pts[9] - wrist)) or 1.0
            shapes[side] = np.round((pts - wrist) / size, 2).tolist()
        self._fingers = fingers
        self._shapes = shapes
        for side in ('Left', 'Right'):
            self._votes[side].append(seen[side])
            votes = list(self._votes[side])
            self._gestures[side] = max(set(votes), key=votes.count)

    def face_metrics(self, results, frame_w, frame_h, mirrored=False):
        """Smoothed blendshape scores; blink sides keyed by screen side."""
        f = dict(self._face)
        if mirrored:
            f['blinkL'], f['blinkR'] = f['blinkR'], f['blinkL']
        return f

    def hand_gestures(self, results, frame_w, frame_h, mirrored=False):
        """Majority-voted gestures keyed by screen side."""
        left, right = self._gestures['Left'], self._gestures['Right']
        if mirrored:
            return {'l': right, 'r': left}
        return {'l': left, 'r': right}

    def hand_shapes(self, mirrored=False):
        # wrist-normalized 21-point hand outlines keyed by screen side
        left, right = self._shapes['Left'], self._shapes['Right']
        if mirrored:
            return {'l': right, 'r': left}
        return {'l': left, 'r': right}

    def finger_states(self, mirrored=False):
        """Per-finger extension [thumb..pinky] keyed by screen side."""
        left, right = self._fingers['Left'], self._fingers['Right']
        if mirrored:
            return {'l': right, 'r': left}
        return {'l': left, 'r': right}

    def close(self):
        self.face_lm.close()
        self.gesture_rec.close()
        super().close()
