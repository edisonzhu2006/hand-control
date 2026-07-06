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


def arm_to_joints(arm, prev_joints, wrist_gain=2.0, vis_threshold=VIS_THRESHOLD,
                  hand=None):
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
        hand: Optional MediaPipe Hands data dict (from HandDetector) matched to
            this arm's wrist. When provided, its index finger drives the claw
            aim — far more precise than Pose's coarse hand landmarks.

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

    # Wrist / claw aim: prefer the Hands model's index finger when available,
    # fall back to Pose's coarse index landmark; hold the previous angle when
    # neither is reliable, to avoid jitter.
    if len(joints) > 2:
        a_hand = None
        if hand is not None:
            a_hand = hand_point_angle(hand)
        elif arm['index_vis'] > vis_threshold:
            a_hand = angle_of(arm['index'] - arm['wrist'])
        if a_hand is not None:
            rel = wrap(a_hand - a_fore)
            joints[2] = float(np.clip(wrist_gain * rel, -np.pi, np.pi))

    return joints, True


def hand_grip_openness(hand):
    """Gripper openness 0 (pinched) .. 1 (open) from MediaPipe Hands landmarks.

    Uses the thumb-tip to index-tip distance normalized by palm size — far more
    reliable than the Pose model's coarse thumb/index points.

    Args:
        hand: Hand data dict from HandDetector.get_hand_landmarks().

    Returns:
        Openness 0..1, or None if the hand has too few landmarks.
    """
    lm = hand.get('landmarks', [])
    if len(lm) < 21:
        return None
    thumb_tip = np.array(lm[4]['pixel'], dtype=float)
    index_tip = np.array(lm[8]['pixel'], dtype=float)
    wrist = np.array(lm[0]['pixel'], dtype=float)
    middle_mcp = np.array(lm[9]['pixel'], dtype=float)

    palm_size = np.linalg.norm(wrist - middle_mcp) or 1.0
    ratio = np.linalg.norm(thumb_tip - index_tip) / palm_size
    return float(np.clip((ratio - 0.25) / (1.0 - 0.25), 0.0, 1.0))


def hand_point_angle(hand):
    """Angle the index finger points at (MCP -> tip), math convention.

    Args:
        hand: Hand data dict from HandDetector.get_hand_landmarks().

    Returns:
        Pointing angle in radians, or None if the hand has too few landmarks.
    """
    lm = hand.get('landmarks', [])
    if len(lm) < 21:
        return None
    mcp = np.array(lm[5]['pixel'], dtype=float)
    tip = np.array(lm[8]['pixel'], dtype=float)
    return angle_of(tip - mcp)


def match_hand_to_wrist(hands, wrist_px, max_dist):
    """Pick the detected hand whose wrist is closest to the pose arm's wrist.

    Args:
        hands: List of hand data dicts from HandDetector.
        wrist_px: (x, y) pixel position of the pose model's wrist.
        max_dist: Maximum wrist-to-wrist pixel distance to accept a match.

    Returns:
        The matching hand dict, or None.
    """
    best, best_d = None, float(max_dist)
    for h in hands:
        lm = h.get('landmarks', [])
        if not lm:
            continue
        d = float(np.linalg.norm(np.array(lm[0]['pixel'], dtype=float) - wrist_px))
        if d < best_d:
            best, best_d = h, d
    return best


def grip_openness(arm, vis_threshold=VIS_THRESHOLD):
    """Estimate gripper openness 0 (pinched) .. 1 (open) from thumb/index spread.

    Returns None when the hand landmarks are not visible enough to judge.
    """
    if arm['index_vis'] < vis_threshold or arm['thumb_vis'] < vis_threshold:
        return None
    forearm_len = np.linalg.norm(arm['wrist'] - arm['elbow']) or 1.0
    spread = np.linalg.norm(arm['index'] - arm['thumb']) / forearm_len
    return float(np.clip((spread - 0.12) / (0.45 - 0.12), 0.0, 1.0))
