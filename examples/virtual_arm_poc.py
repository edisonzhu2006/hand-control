#!/usr/bin/env python3
"""
Proof of concept: control a virtual 2D robot arm with your hand.

Your palm position drives the arm's end effector via resolved-rate Jacobian
servoing (the arm smoothly chases wherever your hand is), and pinching your
thumb and index finger opens/closes the gripper. The window shows the live
webcam with the hand skeleton on the left and the simulated arm on the right.

Controls:
    move hand  - move the arm end effector
    pinch      - close the gripper (spread fingers to open)
    r          - reset arm to home pose
    q          - quit
"""

import sys

import cv2
import numpy as np

sys.path.insert(0, '/Users/edison.zhu/hand-control')
from src.hand_detection.detector import HandDetector
from src.arm.kinematics import Kinematics
from src.arm.visualizer import ArmVisualizer2D
from src.gestures.recognizer import GestureRecognizer

ARM_CONFIG = '/Users/edison.zhu/hand-control/data/arm_configs/3dof_arm.json'
PANEL_H = 480
SIM_W = 480
HOME_POSE = np.array([0.6, -0.9, 0.4])


def map_palm_to_target(palm, frame_w, frame_h, reach):
    """Map a palm pixel position to an (x, y) target in the arm's workspace (mm).

    The target is clamped to an annulus inside the reachable circle so it stays
    reachable and away from the base singularity.
    """
    nx = palm[0] / frame_w
    ny = palm[1] / frame_h
    tx = (nx - 0.5) * 2.0 * 0.78 * reach
    ty = (0.5 - ny) * 2.0 * 0.78 * reach  # screen up -> arm up

    v = np.array([tx, ty], dtype=float)
    r = np.linalg.norm(v)
    r_min, r_max = 0.18 * reach, 0.96 * reach
    if r > r_max:
        v = v / r * r_max
    elif 0.0 < r < r_min:
        v = v / r * r_min
    return v


def pinch_openness(hand):
    """Return gripper openness 0 (pinched/closed) .. 1 (spread/open) from the hand."""
    lm = hand['landmarks']
    thumb = np.array(lm[4]['pixel'], dtype=float)
    index = np.array(lm[8]['pixel'], dtype=float)
    wrist = np.array(lm[0]['pixel'], dtype=float)
    middle_mcp = np.array(lm[9]['pixel'], dtype=float)

    palm_size = np.linalg.norm(wrist - middle_mcp) or 1.0
    ratio = np.linalg.norm(thumb - index) / palm_size
    return float(np.clip((ratio - 0.2) / (1.1 - 0.2), 0.0, 1.0))


def main():
    detector = HandDetector(max_hands=1, confidence=0.6)
    recognizer = GestureRecognizer()
    kin = Kinematics.from_config(ARM_CONFIG)
    viz = ArmVisualizer2D(kin, panel_size=(PANEL_H, SIM_W))

    joints = HOME_POSE[:kin.num_joints].copy()
    gripper = 1.0

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('Error: could not open webcam')
        return

    print('Virtual Arm POC - move your hand to drive the arm; pinch to grip. (q=quit, r=reset)')

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)  # mirror so motion feels natural
            frame_h, frame_w = frame.shape[:2]

            results, _, _, _ = detector.detect(frame)
            hands = detector.get_hand_landmarks(results, frame_h, frame_w)
            frame = detector.draw_landmarks(frame, results)

            target = None
            gesture_name = None
            tracking = bool(hands)

            if hands:
                hand = hands[0]
                palm = hand['palm_center']
                cv2.circle(frame, palm, 9, (0, 255, 0), -1)

                target = map_palm_to_target(palm, frame_w, frame_h, viz.reach)

                # Solve IK for the target (seeded with the current pose so the
                # solution stays close), then step the joints toward it with a
                # per-frame cap. This follows smoothly and, unlike pure Cartesian
                # servoing, never stalls at a singularity because the motion is
                # interpolated in joint space toward a known-good solution.
                target_pose = np.eye(4)
                target_pose[:2, 3] = target
                target_joints, _ = kin.inverse_kinematics(
                    target_pose, initial_guess=joints, max_iterations=100)
                step = np.clip(target_joints - joints, -0.2, 0.2)
                joints = kin._clamp_joints(joints + step)

                gripper = pinch_openness(hand)
                matches = recognizer.recognize(hand, top_k=1)
                if matches:
                    gesture_name = matches[0].gesture_name

            sim = viz.render(joints, target_world=target, gripper=gripper,
                             gesture=gesture_name, tracking=tracking)

            # Left panel: webcam scaled to the sim panel height.
            cam_w = int(frame_w * PANEL_H / frame_h)
            cam = cv2.resize(frame, (cam_w, PANEL_H))
            cv2.putText(cam, 'move hand = arm  |  pinch = gripper  |  q quit  r reset',
                        (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            if not hands:
                cv2.putText(cam, 'Show your hand to the camera', (16, PANEL_H // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2, cv2.LINE_AA)

            cv2.imshow('Virtual Arm POC', np.hstack([cam, sim]))

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                joints = HOME_POSE[:kin.num_joints].copy()

    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()
        print('Demo complete')


if __name__ == '__main__':
    main()
