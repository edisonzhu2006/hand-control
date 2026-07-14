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

# Lip rings (canonical face-mesh topology, ordered for closed polygons).
LIP_RING = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
            291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
LIP_INNER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
             308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
FACE_CHEEK_L, FACE_CHEEK_R = 234, 454


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
        self._shapes = {'Left': None, 'Right': None}    # 21 wrist-normalized pts
        self._face_missing = 999                        # frames since a face was seen
        self._mouth = None                              # lip ring, mouth-centered

    def detect(self, frame):
        """Run pose + face + gestures; returns pose results (as before)."""
        import cv2
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self._ts_ms += 33

        # Pose first: it tells us where the head is, so the face model can run
        # on an upscaled head crop — at full-body distance the face is far too
        # small for FaceLandmarker on the raw frame.
        results = self.pose.process(rgb)
        face_img, face_shape = self._head_crop(rgb, results)
        self._detect_face(face_img, face_shape)
        self._detect_gestures(mp_img, frame.shape)
        return results

    @staticmethod
    def _head_crop(rgb, results, out_size=320):
        """Upscaled crop around the head, or the full frame if no pose."""
        import cv2
        h, w = rgb.shape[:2]
        full = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        if not results.pose_landmarks:
            return full, rgb.shape
        lms = results.pose_landmarks.landmark
        nose, lsh, rsh = lms[0], lms[11], lms[12]
        if nose.visibility < 0.3:
            return full, rgb.shape
        cx, cy = nose.x * w, nose.y * h
        shoulder_w = abs(rsh.x - lsh.x) * w
        half = max(60.0, shoulder_w * 0.85)
        x0, x1 = int(max(0, cx - half)), int(min(w, cx + half))
        y0, y1 = int(max(0, cy - half)), int(min(h, cy + half))
        if x1 - x0 < 40 or y1 - y0 < 40:
            return full, rgb.shape
        crop = rgb[y0:y1, x0:x1]
        if crop.shape[0] < out_size:
            crop = cv2.resize(crop, (out_size, out_size),
                              interpolation=cv2.INTER_LINEAR)
        crop = np.ascontiguousarray(crop)
        return mp.Image(image_format=mp.ImageFormat.SRGB, data=crop), crop.shape

    def _detect_face(self, mp_img, frame_shape):
        res = self.face_lm.detect_for_video(mp_img, self._ts_ms)
        if not res.face_blendshapes:
            # decay toward neutral so a vanished face can't hold its last
            # expression (and report stale after ~0.7s via _face_missing)
            self._face_missing += 1
            self._face = {k: v * 0.8 for k, v in self._face.items()}
            self._mouth = None
            return
        self._face_missing = 0
        shapes = {c.category_name: c.score for c in res.face_blendshapes[0]}
        smile = (shapes.get('mouthSmileLeft', 0) +
                 shapes.get('mouthSmileRight', 0)) / 2
        open_ = min(1.0, shapes.get('jawOpen', 0) * 1.6)
        frown = (shapes.get('mouthFrownLeft', 0) +
                 shapes.get('mouthFrownRight', 0)) / 2
        pucker = shapes.get('mouthPucker', 0)
        a = FACE_SMOOTH
        prev = self._face
        self._face = {
            'smile': float(a * smile + (1 - a) * prev['smile']),
            'open': float(a * open_ + (1 - a) * prev['open']),
            'frown': float(a * frown + (1 - a) * prev.get('frown', 0.0)),
            'pucker': float(a * pucker + (1 - a) * prev.get('pucker', 0.0)),
            # blendshape sides are the image person's anatomy
            'blinkL': float(a * shapes.get('eyeBlinkLeft', 0) +
                            (1 - a) * prev['blinkL']),
            'blinkR': float(a * shapes.get('eyeBlinkRight', 0) +
                            (1 - a) * prev['blinkR']),
        }

        # real lip contour, mouth-centered and scaled by face width, so the
        # avatar can draw the player's actual mouth shape
        if res.face_landmarks:
            h, w = frame_shape[:2]
            lms = res.face_landmarks[0]
            outer = np.array([[lms[i].x * w, lms[i].y * h] for i in LIP_RING])
            inner = np.array([[lms[i].x * w, lms[i].y * h] for i in LIP_INNER])
            face_w = float(np.linalg.norm(
                np.array([lms[FACE_CHEEK_R].x * w, lms[FACE_CHEEK_R].y * h]) -
                np.array([lms[FACE_CHEEK_L].x * w, lms[FACE_CHEEK_L].y * h]))) or 1.0
            center = outer.mean(axis=0)
            self._mouth = {
                'o': np.round((outer - center) / face_w, 3).tolist(),
                'i': np.round((inner - center) / face_w, 3).tolist(),
            }

    def _detect_gestures(self, mp_img, shape):
        res = self.gesture_rec.recognize_for_video(mp_img, self._ts_ms)
        h, w = shape[:2]
        seen = {'Left': 'none', 'Right': 'none'}
        shapes = {'Left': None, 'Right': None}
        for i, handedness in enumerate(res.handedness):
            side = handedness[0].category_name          # image-person anatomy
            if seen[side] != 'none':
                # MediaPipe sometimes labels both hands the same side —
                # route the duplicate into the free slot instead of losing it
                side = 'Left' if side == 'Right' else 'Right'
            pts = np.array([[lm.x * w, lm.y * h]
                            for lm in res.hand_landmarks[i]])
            name = res.gestures[i][0].category_name if res.gestures else 'None'
            g = GESTURE_MAP.get(name)
            if g is None:
                # canned model unsure — fall back to fingertip geometry
                g = classify_gesture(pts)
            seen[side] = g
            # wrist-normalized shape (unit = wrist->middle-MCP) for rendering.
            # A palm aimed at the camera foreshortens: the palm unit collapses
            # and fingertips cluster near the wrist, so the normalized shape
            # turns to amplified garbage — detect that and send no shape (the
            # client falls back to a clean mitt).
            wrist = pts[0]
            size = float(np.linalg.norm(pts[9] - wrist))
            tips = [4, 8, 12, 16, 20]
            span = max(float(np.linalg.norm(pts[t] - wrist)) for t in tips)
            if size < 15.0 or span < 1.35 * size:
                shapes[side] = None
            else:
                shape = np.clip((pts - wrist) / size, -2.8, 2.8)
                prev = self._shapes.get(side)
                if prev is not None and len(prev) == 21:
                    shape = 0.45 * shape + 0.55 * np.array(prev)  # steady hands
                shapes[side] = np.round(shape, 2).tolist()
        self._shapes = shapes
        for side in ('Left', 'Right'):
            self._votes[side].append(seen[side])
            votes = list(self._votes[side])
            self._gestures[side] = max(set(votes), key=votes.count)

    def face_metrics(self, results, frame_w, frame_h, mirrored=False):
        """Smoothed blendshape scores; None once the face has been gone ~0.7s."""
        if self._face_missing > 20:
            return None
        f = dict(self._face)
        f['mouth'] = self._mouth
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

    def close(self):
        self.face_lm.close()
        self.gesture_rec.close()
        super().close()
