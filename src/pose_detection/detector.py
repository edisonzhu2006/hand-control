"""
Upper-body pose detection using MediaPipe Pose.

Exposes the shoulder / elbow / wrist / index / thumb landmarks for one arm so a
robot arm can be puppeted directly from the operator's arm: the elbow and wrist
become pivots and the hand's pointing direction aims the gripper.
"""

import cv2
import mediapipe as mp
import numpy as np


class PoseDetector:
    """Detect a single arm's key joints from a webcam frame."""

    # MediaPipe Pose landmark indices per side.
    LEFT = {'shoulder': 11, 'elbow': 13, 'wrist': 15, 'pinky': 17, 'index': 19, 'thumb': 21}
    RIGHT = {'shoulder': 12, 'elbow': 14, 'wrist': 16, 'pinky': 18, 'index': 20, 'thumb': 22}

    def __init__(self, confidence=0.5, model_complexity=1):
        """Initialize the pose detector.

        Args:
            confidence: Detection/tracking confidence threshold (0-1).
            model_complexity: MediaPipe Pose model complexity (0, 1, or 2).
        """
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=confidence,
            min_tracking_confidence=confidence,
            model_complexity=model_complexity,
        )
        self.mp_drawing = mp.solutions.drawing_utils

    def detect(self, frame):
        """Run pose estimation on a BGR frame and return MediaPipe results."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.pose.process(frame_rgb)

    def get_arm(self, results, frame_w, frame_h, prefer='auto'):
        """Extract one arm's joints as pixel coordinates.

        Args:
            results: MediaPipe pose results.
            frame_w: Frame width.
            frame_h: Frame height.
            prefer: 'left', 'right', or 'auto' (pick the more visible arm).

        Returns:
            A dict with 'shoulder', 'elbow', 'wrist', 'index', 'thumb', 'pinky'
            as (x, y) pixel arrays, plus 'side' and 'visibility'; or None if no
            pose is detected.
        """
        if not results.pose_landmarks:
            return None

        lms = results.pose_landmarks.landmark

        def pack(mapping):
            arm = {}
            core_vis = 0.0
            for name, idx in mapping.items():
                lm = lms[idx]
                arm[name] = np.array([lm.x * frame_w, lm.y * frame_h], dtype=float)
                arm[name + '_vis'] = float(lm.visibility)
            core_vis = arm['shoulder_vis'] + arm['elbow_vis'] + arm['wrist_vis']
            return arm, core_vis

        left, left_vis = pack(self.LEFT)
        right, right_vis = pack(self.RIGHT)

        if prefer == 'left':
            arm, side, vis = left, 'left', left_vis
        elif prefer == 'right':
            arm, side, vis = right, 'right', right_vis
        else:
            if right_vis >= left_vis:
                arm, side, vis = right, 'right', right_vis
            else:
                arm, side, vis = left, 'left', left_vis

        arm['side'] = side
        arm['visibility'] = vis
        return arm

    # Landmarks used by get_body: head, both arms, and both legs.
    BODY = {
        'nose': 0,
        'l_shoulder': 11, 'r_shoulder': 12,
        'l_elbow': 13, 'r_elbow': 14,
        'l_wrist': 15, 'r_wrist': 16,
        'l_hip': 23, 'r_hip': 24,
        'l_knee': 25, 'r_knee': 26,
        'l_ankle': 27, 'r_ankle': 28,
    }

    def get_body(self, results, frame_w, frame_h, mirrored=False):
        """Extract head, arms and legs as pixel coordinates.

        Args:
            results: MediaPipe pose results.
            frame_w: Frame width.
            frame_h: Frame height.
            mirrored: Set True when detection ran on a horizontally flipped
                (selfie-mirror) frame. MediaPipe labels sides by the image
                person's anatomy, which is opposite to screen side in a
                mirrored frame — this swaps l_/r_ so keys mean screen side.

        Returns:
            A dict mapping each BODY key to an (x, y) pixel array plus a
            '<key>_vis' visibility score, or None if no pose is detected.
        """
        if not results.pose_landmarks:
            return None

        lms = results.pose_landmarks.landmark
        body = {}
        for name, idx in self.BODY.items():
            if mirrored:
                if name.startswith('l_'):
                    name = 'r_' + name[2:]
                elif name.startswith('r_'):
                    name = 'l_' + name[2:]
            lm = lms[idx]
            body[name] = np.array([lm.x * frame_w, lm.y * frame_h], dtype=float)
            body[name + '_vis'] = float(lm.visibility)
        return body

    def get_body3d(self, results, mirrored=False):
        """Extract the BODY joints as 3D world coordinates in meters.

        MediaPipe world landmarks: origin at the hip midpoint, x right,
        y down, z increasing away from the camera. With mirrored=True the
        sides are swapped and x negated so keys mean screen side, matching
        get_body(mirrored=True).

        Returns:
            Dict of joint -> (x, y, z) float arrays, or None if no pose.
        """
        if not results.pose_world_landmarks:
            return None
        lms = results.pose_world_landmarks.landmark
        body = {}
        for name, idx in self.BODY.items():
            if mirrored:
                if name.startswith('l_'):
                    name = 'r_' + name[2:]
                elif name.startswith('r_'):
                    name = 'l_' + name[2:]
            lm = lms[idx]
            x = -lm.x if mirrored else lm.x
            body[name] = np.array([x, lm.y, lm.z], dtype=float)
        return body

    def draw(self, frame, results):
        """Draw the full pose skeleton on the frame."""
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
        return frame

    def close(self):
        """Release resources."""
        self.pose.close()
