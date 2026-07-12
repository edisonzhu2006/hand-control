#!/usr/bin/env python3
"""
Hole in the Wall: strike the pose before the wall arrives.

A ghost pose is overlaid on your body (anchored to your own shoulders, scaled
to your own limb lengths). A wall closes in on a countdown — match the ghost
with your arms before it arrives. Segments turn green as you match them. If
your overall match is good enough when the wall hits, you pass through and
score; otherwise you crash. Each pass shaves time off the next round.

Controls:
    strike the pose - match all four ghost arm segments
    s               - skip to a new pose
    q               - quit
"""

import sys
import time

import cv2
import numpy as np

sys.path.insert(0, '/Users/edison.zhu/hand-control')
from src.pose_detection.detector import PoseDetector
from src.utils.filters import OneEuroFilter

VIS_THRESHOLD = 0.3
MATCH_TOLERANCE = 25.0    # deg: a segment within this of the target counts as matched
PASS_THRESHOLD = 0.70     # overall match needed to go through the wall
ROUND_TIME = 6.0          # seconds per wall, shrinks with each pass
ROUND_TIME_MIN = 3.0
ROUND_TIME_STEP = 0.25
FILTER_MIN_CUTOFF = 0.8
FILTER_BETA = 0.008

# Arm segments: (start joint, end joint, target-angle key).
SEGMENTS = [
    ('l_shoulder', 'l_elbow', 'lu'),
    ('l_elbow', 'l_wrist', 'lf'),
    ('r_shoulder', 'r_elbow', 'ru'),
    ('r_elbow', 'r_wrist', 'rf'),
]

# Target poses as segment angles in degrees (math convention: 0 = screen
# right, 90 = up). 'l_*' is the arm on the left side of the mirrored view.
POSES = [
    {'name': 'T-POSE', 'angles': {'lu': 180, 'lf': 180, 'ru': 0, 'rf': 0}},
    {'name': 'GOALPOST', 'angles': {'lu': 180, 'lf': 90, 'ru': 0, 'rf': 90}},
    {'name': 'ARMS UP', 'angles': {'lu': 135, 'lf': 135, 'ru': 45, 'rf': 45}},
    {'name': 'LEFT UP RIGHT OUT', 'angles': {'lu': 135, 'lf': 90, 'ru': 0, 'rf': 0}},
    {'name': 'RIGHT UP LEFT OUT', 'angles': {'lu': 180, 'lf': 180, 'ru': 45, 'rf': 90}},
    {'name': 'ARMS DOWN', 'angles': {'lu': 225, 'lf': 225, 'ru': -45, 'rf': -45}},
]


def seg_angle(body, a, b):
    """Angle of the segment a->b in degrees (math convention), or None if not visible."""
    if body[a + '_vis'] < VIS_THRESHOLD or body[b + '_vis'] < VIS_THRESHOLD:
        return None
    v = body[b] - body[a]
    return float(np.degrees(np.arctan2(-v[1], v[0])))


def ang_diff(a, b):
    """Absolute angular difference in degrees, wrapped to [0, 180]."""
    return abs((a - b + 180.0) % 360.0 - 180.0)


def match_pose(body, angles):
    """Score how well the body matches the target angles.

    Args:
        body: Body dict from PoseDetector.get_body().
        angles: Target angles dict keyed like POSES[n]['angles'].

    Returns:
        (match, seg_ok): overall match 0..1 (None if too few segments are
        visible) and a {angle_key: matched} dict for visible segments.
    """
    scores = []
    seg_ok = {}
    for a, b, key in SEGMENTS:
        cur = seg_angle(body, a, b)
        if cur is None:
            continue
        err = ang_diff(cur, angles[key])
        seg_ok[key] = err < MATCH_TOLERANCE
        scores.append(np.clip(1.0 - err / 60.0, 0.0, 1.0))
    if len(scores) < 2:
        return None, seg_ok
    return float(np.mean(scores)), seg_ok


def ghost_points(body, angles):
    """Compute ghost joint pixel positions for the target pose.

    Anchored at the player's own shoulders and scaled to their own limb
    lengths, so the ghost sits on their body regardless of size or distance.
    """
    def direction(deg):
        r = np.radians(deg)
        return np.array([np.cos(r), -np.sin(r)])  # screen y grows downward

    ghost = {}
    for side in ('l', 'r'):
        sh = body[side + '_shoulder']
        upper_len = np.linalg.norm(body[side + '_elbow'] - sh)
        fore_len = np.linalg.norm(body[side + '_wrist'] - body[side + '_elbow'])
        # Fall back to plausible lengths if a joint is barely visible.
        upper_len = upper_len if upper_len > 10 else 90.0
        fore_len = fore_len if fore_len > 10 else 80.0

        elbow = sh + direction(angles[side + 'u']) * upper_len
        wrist = elbow + direction(angles[side + 'f']) * fore_len
        ghost[side + '_shoulder'] = sh
        ghost[side + '_elbow'] = elbow
        ghost[side + '_wrist'] = wrist
    return ghost


class HoleInWallGame:
    """Round state machine: countdown wall, evaluation, scoring, difficulty."""

    def __init__(self, now):
        self.score = 0
        self.streak = 0
        self.round_time = ROUND_TIME
        self.result = None          # (text, color, until)
        self._order = np.random.permutation(len(POSES)).tolist()
        self.pose = None
        self.new_round(now)

    def new_round(self, now):
        if not self._order:
            self._order = np.random.permutation(len(POSES)).tolist()
        # Avoid repeating the pose we just played.
        if self.pose is not None and POSES[self._order[0]] is self.pose and len(self._order) > 1:
            self._order.append(self._order.pop(0))
        self.pose = POSES[self._order.pop(0)]
        self.deadline = now + self.round_time

    def time_left(self, now):
        return max(0.0, self.deadline - now)

    def update(self, match, now):
        """Advance the round; evaluate when the wall arrives.

        Args:
            match: Current overall match 0..1, or None if the body isn't visible.
            now: Current time (seconds).

        Returns:
            'pass' | 'crash' | None for this frame.
        """
        if self.time_left(now) > 0:
            return None

        if match is not None and match >= PASS_THRESHOLD:
            self.score += 1
            self.streak += 1
            self.round_time = max(ROUND_TIME_MIN, self.round_time - ROUND_TIME_STEP)
            self.result = ('THROUGH!', (80, 230, 120), now + 1.2)
            outcome = 'pass'
        else:
            self.streak = 0
            self.result = ('CRASHED!', (60, 60, 255), now + 1.2)
            outcome = 'crash'
        self.new_round(now)
        return outcome


def draw_ghost(frame, ghost, seg_ok):
    """Draw the target pose skeleton; matched segments green, unmatched orange."""
    overlay = frame.copy()
    for a, b, key in SEGMENTS:
        color = (80, 230, 120) if seg_ok.get(key) else (60, 140, 255)
        pa = tuple(np.round(ghost[a]).astype(int))
        pb = tuple(np.round(ghost[b]).astype(int))
        cv2.line(overlay, pa, pb, color, 14, cv2.LINE_AA)
        cv2.circle(overlay, pb, 10, color, -1, cv2.LINE_AA)
    # Shoulder bar to visually tie the ghost together.
    cv2.line(overlay,
             tuple(np.round(ghost['l_shoulder']).astype(int)),
             tuple(np.round(ghost['r_shoulder']).astype(int)),
             (200, 200, 200), 6, cv2.LINE_AA)
    return cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)


def draw_wall(frame, progress):
    """Draw the approaching wall: a border that thickens as time runs out."""
    h, w = frame.shape[:2]
    # Border grows from the edges toward an inner window as progress -> 1.
    max_inset = int(0.13 * min(h, w))
    inset = int(max_inset * progress)
    color = (140, 120, 60) if progress < 0.8 else (60, 60, 255)
    if inset > 0:
        cv2.rectangle(frame, (0, 0), (w, inset), color, -1)              # top
        cv2.rectangle(frame, (0, h - inset), (w, h), color, -1)          # bottom
        cv2.rectangle(frame, (0, 0), (inset, h), color, -1)              # left
        cv2.rectangle(frame, (w - inset, 0), (w, h), color, -1)          # right
    cv2.rectangle(frame, (inset, inset), (w - inset, h - inset), color, 3)
    return frame


def _outlined(frame, text, org, scale, color, thickness):
    """putText with a black outline so it stays readable on any background."""
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, thickness, cv2.LINE_AA)


def draw_hud(frame, game, match, now):
    """Score, streak, pose name, countdown and match meter."""
    h, w = frame.shape[:2]
    left = game.time_left(now)

    (tw, _), _ = cv2.getTextSize(game.pose['name'], cv2.FONT_HERSHEY_SIMPLEX, 1.1, 3)
    _outlined(frame, game.pose['name'], ((w - tw) // 2, 46), 1.1, (255, 255, 255), 3)
    _outlined(frame, f'{left:.1f}s', (w // 2 - 34, 86), 0.9,
              (255, 255, 255) if left > 2 else (80, 80, 255), 2)
    _outlined(frame, f'SCORE {game.score}   STREAK {game.streak}', (16, 34),
              0.7, (80, 230, 120), 2)

    # Match meter along the bottom.
    if match is not None:
        bar_w = int((w - 200) * match)
        color = (80, 230, 120) if match >= PASS_THRESHOLD else (60, 140, 255)
        cv2.rectangle(frame, (100, h - 42), (100 + bar_w, h - 22), color, -1)
        cv2.rectangle(frame, (100, h - 42), (w - 100, h - 22), (200, 200, 200), 2)
        pass_x = 100 + int((w - 200) * PASS_THRESHOLD)
        cv2.line(frame, (pass_x, h - 48), (pass_x, h - 16), (255, 255, 255), 2)
        cv2.putText(frame, f'{match * 100:.0f}%', (16, h - 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, 'Step back so both arms are visible', (100, h - 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2, cv2.LINE_AA)

    if game.result and now < game.result[2]:
        text, color, _ = game.result
        _outlined(frame, text, (w // 2 - 160, h // 2), 2.0, color, 5)
    return frame


def main():
    pose_det = PoseDetector(confidence=0.5, model_complexity=1)
    filters = {name: OneEuroFilter(FILTER_MIN_CUTOFF, FILTER_BETA)
               for name in PoseDetector.BODY}

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('Error: could not open webcam')
        return

    game = HoleInWallGame(time.time())
    print('Hole in the Wall - match the ghost pose before the wall arrives! (s=skip, q=quit)')

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            frame_h, frame_w = frame.shape[:2]
            now = time.time()

            results = pose_det.detect(frame)
            body = pose_det.get_body(results, frame_w, frame_h)

            match, seg_ok = None, {}
            if body is not None:
                for name in PoseDetector.BODY:
                    body[name] = filters[name].apply(body[name])
                match, seg_ok = match_pose(body, game.pose['angles'])
                shoulders_ok = (body['l_shoulder_vis'] > VIS_THRESHOLD and
                                body['r_shoulder_vis'] > VIS_THRESHOLD)
                if shoulders_ok:
                    ghost = ghost_points(body, game.pose['angles'])
                    frame = draw_ghost(frame, ghost, seg_ok)
            else:
                for f in filters.values():
                    f.reset()

            progress = 1.0 - game.time_left(now) / game.round_time
            frame = draw_wall(frame, min(max(progress, 0.0), 1.0))
            outcome = game.update(match, now)
            if outcome:
                print(f"{game.pose['name']:<20} <- next | last round: {outcome.upper()} "
                      f"| score {game.score} streak {game.streak}")
            frame = draw_hud(frame, game, match, now)

            cv2.imshow('Hole in the Wall', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                game.new_round(now)

    finally:
        cap.release()
        cv2.destroyAllWindows()
        pose_det.close()
        print(f'Final score: {game.score}')


if __name__ == '__main__':
    main()
