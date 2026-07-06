"""
Map a tracked human arm onto planar robot joint angles.

The operator's arm puppets the robot directly:

    robot joint 1 (base)  <- upper-arm direction (shoulder -> elbow)
    robot joint 2 (elbow) <- forearm bend        (elbow -> wrist)
    robot joint 3 (wrist) <- hand pointing       (wrist -> index), amplified by
                             wrist_gain so a comfortable wrist bend sweeps the
                             claw through a much larger arc
    gripper               <- thumb/index spread
"""

import numpy as np

VIS_THRESHOLD = 0.3  # minimum landmark visibility to trust a joint


def angle_of(v):
    """Angle of a vector in the standard math plane (image y points down, so flip it)."""
    return np.arctan2(-v[1], v[0])


def wrap(a):
    """Wrap an angle to [-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def arm_to_joints(arm, prev_joints, wrist_gain=2.0, vis_threshold=VIS_THRESHOLD):
    """Map a detected human arm to robot joint angles (relative DH angles).

    Args:
        arm: Arm dict from PoseDetector.get_arm().
        prev_joints: Previous joint angles, used to hold a joint when its
            driving landmark is not visible enough.
        wrist_gain: Amplification applied to the wrist/claw angle. The human
            wrist only bends ~±60°, so a gain of 2 lets that comfortable range
            drive the claw through ±120°. Clipped (not wrapped) at ±pi so a
            hard bend saturates instead of flipping sign.
        vis_threshold: Minimum landmark visibility to trust a joint.

    Returns:
        (joints, valid): joint angles array and whether the core arm was tracked.
    """
    joints = prev_joints.copy()

    core_ok = all(arm[k + '_vis'] > vis_threshold
                  for k in ('shoulder', 'elbow', 'wrist'))
    if not core_ok:
        return joints, False

    upper = arm['elbow'] - arm['shoulder']
    fore = arm['wrist'] - arm['elbow']
    a_upper = angle_of(upper)
    a_fore = angle_of(fore)

    joints[0] = a_upper                  # base <- upper-arm direction
    joints[1] = wrap(a_fore - a_upper)   # elbow <- forearm relative to upper arm

    # Wrist / claw aim: only update when the hand landmark is reliable,
    # otherwise keep the previous wrist angle to avoid jitter.
    if len(joints) > 2 and arm['index_vis'] > vis_threshold:
        rel = wrap(angle_of(arm['index'] - arm['wrist']) - a_fore)
        joints[2] = float(np.clip(wrist_gain * rel, -np.pi, np.pi))

    return joints, True


def grip_openness(arm, vis_threshold=VIS_THRESHOLD):
    """Estimate gripper openness 0 (pinched) .. 1 (open) from thumb/index spread.

    Returns None when the hand landmarks are not visible enough to judge.
    """
    if arm['index_vis'] < vis_threshold or arm['thumb_vis'] < vis_threshold:
        return None
    forearm_len = np.linalg.norm(arm['wrist'] - arm['elbow']) or 1.0
    spread = np.linalg.norm(arm['index'] - arm['thumb']) / forearm_len
    return float(np.clip((spread - 0.12) / (0.45 - 0.12), 0.0, 1.0))
