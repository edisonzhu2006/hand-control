#!/usr/bin/env python3
"""
Proof of concept: puppet a virtual robot arm with your own arm.

Uses MediaPipe Pose to track your shoulder, elbow, wrist and hand, then maps the
angles of your arm segments directly onto the robot's joints:

    robot joint 1 (base)  <- your upper-arm direction (shoulder -> elbow)
    robot joint 2 (elbow) <- your forearm bend       (elbow  -> wrist)
    robot joint 3 (wrist) <- where your fingers point (wrist  -> index)  => claw aim
    gripper               <- your thumb/index spread (pinch to close)

Stand back so the camera can see your shoulder, elbow, wrist and hand. The left
panel shows the webcam with your pose skeleton; the right panel is the robot arm.

Controls:
    move your arm  - the robot arm mimics your elbow and wrist
    point          - the claw points where your fingers point
    pinch          - close the gripper
    q              - quit
"""

import sys

import cv2
import numpy as np

sys.path.insert(0, '/Users/edison.zhu/hand-control')
from src.pose_detection.detector import PoseDetector
from src.arm.kinematics import Kinematics
from src.arm.visualizer import ArmVisualizer2D
from src.utils.filters import OneEuroFilter

ARM_CONFIG = '/Users/edison.zhu/hand-control/data/arm_configs/3dof_arm.json'
PANEL_H = 480
SIM_W = 480
VIS_THRESHOLD = 0.3    # min landmark visibility to trust a joint
GRIP_SMOOTHING = 0.4   # EMA factor for the gripper only

# One Euro filter tuning for pixel-space landmarks: smooth hard when still,
# open up quickly during fast motion.
FILTER_MIN_CUTOFF = 0.8   # Hz
FILTER_BETA = 0.008
FILTERED_POINTS = ('shoulder', 'elbow', 'wrist', 'index', 'thumb')


def _angle(v):
    """Angle of a vector in the standard math plane (image y points down, so flip it)."""
    return np.arctan2(-v[1], v[0])


def _wrap(a):
    """Wrap an angle to [-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def arm_to_joints(arm, prev_joints):
    """Map a detected human arm to robot joint angles (relative DH angles).

    Args:
        arm: Arm dict from PoseDetector.get_arm().
        prev_joints: Previous joint angles, used to hold a joint when its
            driving landmark is not visible enough.

    Returns:
        (joints, valid): joint angles array and whether the core arm was tracked.
    """
    joints = prev_joints.copy()

    core_ok = (arm['shoulder_vis'] > VIS_THRESHOLD and
               arm['elbow_vis'] > VIS_THRESHOLD and
               arm['wrist_vis'] > VIS_THRESHOLD)
    if not core_ok:
        return joints, False

    upper = arm['elbow'] - arm['shoulder']   # upper arm
    fore = arm['wrist'] - arm['elbow']        # forearm
    a_upper = _angle(upper)
    a_fore = _angle(fore)

    joints[0] = a_upper                        # base <- upper-arm direction
    joints[1] = _wrap(a_fore - a_upper)        # elbow <- forearm relative to upper arm

    # Wrist / claw aim: only update when the hand landmarks are reliable,
    # otherwise keep the previous wrist angle to avoid jitter.
    if arm['index_vis'] > VIS_THRESHOLD:
        hand = arm['index'] - arm['wrist']
        a_hand = _angle(hand)
        if len(joints) > 2:
            joints[2] = _wrap(a_hand - a_fore)

    return joints, True


def grip_openness(arm):
    """Estimate gripper openness 0 (pinched) .. 1 (open) from thumb/index spread."""
    if arm['index_vis'] < VIS_THRESHOLD or arm['thumb_vis'] < VIS_THRESHOLD:
        return None
    forearm_len = np.linalg.norm(arm['wrist'] - arm['elbow']) or 1.0
    spread = np.linalg.norm(arm['index'] - arm['thumb']) / forearm_len
    return float(np.clip((spread - 0.12) / (0.45 - 0.12), 0.0, 1.0))


def main():
    pose = PoseDetector(confidence=0.5, model_complexity=1)
    kin = Kinematics.from_config(ARM_CONFIG)
    viz = ArmVisualizer2D(kin, panel_size=(PANEL_H, SIM_W))

    joints = np.zeros(kin.num_joints)
    gripper = 1.0
    filters = {name: OneEuroFilter(FILTER_MIN_CUTOFF, FILTER_BETA)
               for name in FILTERED_POINTS}
    last_side = None

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('Error: could not open webcam')
        return

    print('Arm Puppet POC - move your arm; the robot mimics your elbow & wrist, claw follows your fingers. (q=quit)')

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)  # mirror so it feels like a reflection
            frame_h, frame_w = frame.shape[:2]

            results = pose.detect(frame)
            frame = pose.draw(frame, results)
            arm = pose.get_arm(results, frame_w, frame_h, prefer='auto')

            tracking = False
            label = None
            if arm is not None:
                # Reset filters if we switched arms (avoid blending two arms).
                if arm['side'] != last_side:
                    for f in filters.values():
                        f.reset()
                    last_side = arm['side']

                # One Euro smoothing on the raw landmarks: jitter-free when the
                # arm is still, near-zero lag when it moves fast.
                for name in FILTERED_POINTS:
                    arm[name] = filters[name].apply(arm[name])

                target_joints, tracking = arm_to_joints(arm, joints)
                if tracking:
                    joints = kin._clamp_joints(target_joints)
                    g = grip_openness(arm)
                    if g is not None:
                        gripper = GRIP_SMOOTHING * g + (1 - GRIP_SMOOTHING) * gripper
                    label = f"{arm['side']} arm"
            else:
                last_side = None
                for f in filters.values():
                    f.reset()

            sim = viz.render(joints, target_world=None, gripper=gripper,
                             gesture=label, tracking=tracking)

            cam_w = int(frame_w * PANEL_H / frame_h)
            cam = cv2.resize(frame, (cam_w, PANEL_H))
            cv2.putText(cam, 'move arm = robot  |  point = claw aim  |  pinch = grip  |  q quit',
                        (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            if not tracking:
                cv2.putText(cam, 'Step back so your shoulder, elbow & wrist are visible',
                            (16, PANEL_H // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 220, 255), 2, cv2.LINE_AA)

            cv2.imshow('Arm Puppet POC', np.hstack([cam, sim]))

            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        pose.close()
        print('Demo complete')


if __name__ == '__main__':
    main()
