#!/usr/bin/env python3
"""
Hole in the Wall: your body puppets a stickman — fit him through the hole.

The webcam is the controller, not the view: your pose drives a chunky red
stickman standing in an arena while a brick wall rushes at him from the
horizon with a stickman-shaped hole cut out. Strike the pose so he fits —
his arms glow green as they match. Fit when the wall arrives and he flies
through it; miss and it flattens him (X-eyes, screen shake, flying bricks).
Three hearts, streak multipliers, PERFECT bonuses, persistent high score.
A small camera preview sits in the corner so you can frame yourself.

Controls:
    strike the pose - fit through the hole
    s               - skip this wall (no penalty)
    SPACE           - play again (on game over)
    q               - quit
"""

import json
import os
import subprocess
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, '/Users/edison.zhu/hand-control')
from src.pose_detection.detector import PoseDetector
from src.render.stickman import Stickman, pose_from_body, pose_from_angles
from src.utils.filters import OneEuroFilter

VIS_THRESHOLD = 0.3
MATCH_TOLERANCE = 30.0     # deg: a segment within this of the target counts as matched
FALLOFF = 50.0             # deg: score decays to 0 this far beyond tolerance
PASS_THRESHOLD = 0.70      # overall match needed to fit through the hole
GRACE_WINDOW = 0.5         # s: verdict uses the best match in this final window
ROUND_TIME = 6.0           # seconds per wall, shrinks with each pass
ROUND_TIME_MIN = 3.5
ROUND_TIME_STEP = 0.2
PERFECT_MATCH = 0.90
START_LIVES = 3
COUNTDOWN_SECS = 3
RESULT_SECS = 1.0
FILTER_MIN_CUTOFF = 0.8
FILTER_BETA = 0.008
FRAME_W, FRAME_H = 960, 540
AVATAR_ANCHOR = (FRAME_W // 2, 195)   # avatar shoulder center in the world
WALL_BEHIND_UNTIL = 0.90              # wall draws behind the avatar until this scale
HIGHSCORE_PATH = '/Users/edison.zhu/hand-control/data/hole_in_wall_highscore.json'

SEGMENTS = [
    ('l_shoulder', 'l_elbow', 'lu'),
    ('l_elbow', 'l_wrist', 'lf'),
    ('r_shoulder', 'r_elbow', 'ru'),
    ('r_elbow', 'r_wrist', 'rf'),
]

# Target poses as segment angles in degrees (math convention: 0 = screen
# right, 90 = up). 'l_*' is the arm on the left side of the mirrored view.
# The first six are the easier pool used at level 1.
POSES = [
    {'name': 'T-POSE', 'angles': {'lu': 180, 'lf': 180, 'ru': 0, 'rf': 0}},
    {'name': 'GOALPOST', 'angles': {'lu': 180, 'lf': 90, 'ru': 0, 'rf': 90}},
    {'name': 'ARMS UP', 'angles': {'lu': 135, 'lf': 135, 'ru': 45, 'rf': 45}},
    {'name': 'ARMS DOWN', 'angles': {'lu': 225, 'lf': 225, 'ru': -45, 'rf': -45}},
    {'name': 'LEFT HAND UP', 'angles': {'lu': 100, 'lf': 100, 'ru': -70, 'rf': -70}},
    {'name': 'RIGHT HAND UP', 'angles': {'lu': -110, 'lf': -110, 'ru': 80, 'rf': 80}},
    {'name': 'LEFT UP RIGHT OUT', 'angles': {'lu': 135, 'lf': 90, 'ru': 0, 'rf': 0}},
    {'name': 'RIGHT UP LEFT OUT', 'angles': {'lu': 180, 'lf': 180, 'ru': 45, 'rf': 90}},
    {'name': 'HANDS UP', 'angles': {'lu': 105, 'lf': 105, 'ru': 75, 'rf': 75}},
    {'name': 'MUSCLE FLEX', 'angles': {'lu': 180, 'lf': 55, 'ru': 0, 'rf': 125}},
    {'name': 'HANDS ON HIPS', 'angles': {'lu': 235, 'lf': -55, 'ru': -55, 'rf': 235}},
    {'name': 'AIRPLANE', 'angles': {'lu': 150, 'lf': 150, 'ru': -30, 'rf': -30}},
]
EASY_POOL = 6

STICK = Stickman(height=300)
MINI = Stickman(height=72)


# ---------------------------------------------------------------- pose math

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
    """Overall match 0..1 (None if <2 segments visible) and per-segment flags.

    A segment scores full credit anywhere inside MATCH_TOLERANCE and decays
    linearly beyond it, so the meter agrees with the green segment feedback:
    all segments green always means a passing score.
    """
    scores = []
    seg_ok = {}
    for a, b, key in SEGMENTS:
        cur = seg_angle(body, a, b)
        if cur is None:
            continue
        err = ang_diff(cur, angles[key])
        seg_ok[key] = err < MATCH_TOLERANCE
        scores.append(np.clip(1.0 - max(err - MATCH_TOLERANCE, 0.0) / FALLOFF, 0.0, 1.0))
    if len(scores) < 2:
        return None, seg_ok
    return float(np.mean(scores)), seg_ok


# ------------------------------------------------------------ world visuals

def build_world(w, h):
    """Prebuilt arena backdrop: sky gradient, floor with perspective lines."""
    img = np.zeros((h, w, 3), np.uint8)
    horizon = int(h * 0.52)
    top = np.array([70, 48, 34], float)       # dark blue-gray sky (BGR)
    bottom = np.array([120, 86, 58], float)
    for y in range(horizon):
        a = y / max(horizon - 1, 1)
        img[y, :] = (top * (1 - a) + bottom * a).astype(np.uint8)
    floor_top = np.array([64, 54, 46], float)
    floor_bottom = np.array([34, 30, 26], float)
    for y in range(horizon, h):
        a = (y - horizon) / max(h - horizon - 1, 1)
        img[y, :] = (floor_top * (1 - a) + floor_bottom * a).astype(np.uint8)
    # perspective lines converging on the horizon center
    vp = (w // 2, horizon)
    for x in range(-w, 2 * w, w // 6):
        cv2.line(img, (x, h), vp, (52, 44, 38), 2, cv2.LINE_AA)
    for i in range(1, 6):
        y = horizon + int((h - horizon) * (i / 6) ** 1.7)
        cv2.line(img, (0, y), (w, y), (52, 44, 38), 1)
    cv2.line(img, (0, horizon), (w, horizon), (86, 66, 50), 2)
    # avatar spotlight
    cv2.ellipse(img, (w // 2, int(h * 0.92)), (240, 46), 0, 0, 360, (74, 64, 54), -1)
    return img


def brick_texture(w, h):
    """Foam-brick wall texture, prebuilt once."""
    img = np.full((h, w, 3), (190, 150, 60), np.uint8)
    step = max(36, h // 12)
    for y in range(0, h, step):
        cv2.line(img, (0, y), (w, y), (160, 122, 42), 2)
        off = step if (y // step) % 2 else 0
        for x in range(off, w, step * 2):
            cv2.line(img, (x, y), (x, y + step), (160, 122, 42), 2)
    return img


def composite_wall(frame, wall, mask, scale, rim_color):
    """Draw the wall over the frame at the given approach scale (about center)."""
    h, w = frame.shape[:2]
    if scale < 1.0:
        sw, sh = max(2, int(w * scale)), max(2, int(h * scale))
        wall_s = cv2.resize(wall, (sw, sh), interpolation=cv2.INTER_AREA)
        mask_s = cv2.resize(mask, (sw, sh), interpolation=cv2.INTER_NEAREST)
        x0, y0 = (w - sw) // 2, (h - sh) // 2
        region = frame[y0:y0 + sh, x0:x0 + sw]
    else:
        zw, zh = int(w * scale), int(h * scale)
        wall_z = cv2.resize(wall, (zw, zh), interpolation=cv2.INTER_LINEAR)
        mask_z = cv2.resize(mask, (zw, zh), interpolation=cv2.INTER_NEAREST)
        cx0, cy0 = (zw - w) // 2, (zh - h) // 2
        wall_s = wall_z[cy0:cy0 + h, cx0:cx0 + w]
        mask_s = mask_z[cy0:cy0 + h, cx0:cx0 + w]
        region = frame

    solid = mask_s == 0
    region[solid] = wall_s[solid]
    contours, _ = cv2.findContours(mask_s, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(region, contours, -1, rim_color, 3, cv2.LINE_AA)
    return frame


# ------------------------------------------------------------------- juice

class SoundPlayer:
    """Fire-and-forget system sounds via afplay (silently disabled if absent)."""

    NAMES = {'tick': 'Tink', 'go': 'Ping', 'pass': 'Glass', 'perfect': 'Hero',
             'crash': 'Basso', 'gameover': 'Submarine'}

    def __init__(self):
        self.paths = {}
        for key, name in self.NAMES.items():
            p = f'/System/Library/Sounds/{name}.aiff'
            if os.path.exists(p):
                self.paths[key] = p

    def play(self, key):
        p = self.paths.get(key)
        if p:
            subprocess.Popen(['afplay', p], stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)


class Particles:
    """Brick chunks that burst out on a crash."""

    def __init__(self):
        self.parts = []

    def burst(self, center, n=30):
        for _ in range(n):
            ang = np.random.uniform(0, 2 * np.pi)
            speed = np.random.uniform(150, 650)
            self.parts.append({
                'p': np.array(center, float),
                'v': np.array([np.cos(ang), -abs(np.sin(ang))]) * speed,
                's': np.random.randint(5, 14),
                'c': (int(np.random.uniform(150, 200)),
                      int(np.random.uniform(110, 160)),
                      int(np.random.uniform(40, 80))),
                't': np.random.uniform(0.5, 1.0),
            })

    def step_draw(self, frame, dt):
        h, w = frame.shape[:2]
        alive = []
        for part in self.parts:
            part['t'] -= dt
            if part['t'] <= 0:
                continue
            part['v'][1] += 1400 * dt
            part['p'] += part['v'] * dt
            x, y = int(part['p'][0]), int(part['p'][1])
            if -20 < x < w + 20 and -20 < y < h + 20:
                s = part['s']
                cv2.rectangle(frame, (x - s // 2, y - s // 2),
                              (x + s // 2, y + s // 2), part['c'], -1)
            alive.append(part)
        self.parts = alive


def _outlined(frame, text, org, scale, color, thickness):
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, thickness, cv2.LINE_AA)


def _centered(frame, text, y, scale, color, thickness):
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    _outlined(frame, text, ((frame.shape[1] - tw) // 2, y), scale, color, thickness)


def draw_heart(img, c, s, color, filled=True):
    pts = np.array([[c[0] - s, c[1] - s // 4], [c[0] + s, c[1] - s // 4],
                    [c[0], c[1] + s]])
    if filled:
        cv2.circle(img, (c[0] - s // 2, c[1] - s // 4), s // 2 + 1, color, -1, cv2.LINE_AA)
        cv2.circle(img, (c[0] + s // 2, c[1] - s // 4), s // 2 + 1, color, -1, cv2.LINE_AA)
        cv2.fillPoly(img, [pts], color, cv2.LINE_AA)
    else:
        cv2.circle(img, (c[0] - s // 2, c[1] - s // 4), s // 2 + 1, color, 2, cv2.LINE_AA)
        cv2.circle(img, (c[0] + s // 2, c[1] - s // 4), s // 2 + 1, color, 2, cv2.LINE_AA)
        cv2.polylines(img, [pts], True, color, 2, cv2.LINE_AA)


# -------------------------------------------------------------------- game

class HoleInWallGame:
    """State machine: COUNTDOWN -> WALL -> RESULT -> (WALL | GAME_OVER)."""

    def __init__(self, now):
        self.high_score = self._load_high_score()
        self.sounds = SoundPlayer()
        self.particles = Particles()
        self.popups = []
        self.reset(now)

    def _load_high_score(self):
        try:
            with open(HIGHSCORE_PATH) as f:
                return int(json.load(f).get('high_score', 0))
        except (OSError, ValueError):
            return 0

    def _save_high_score(self):
        try:
            os.makedirs(os.path.dirname(HIGHSCORE_PATH), exist_ok=True)
            with open(HIGHSCORE_PATH, 'w') as f:
                json.dump({'high_score': self.high_score}, f)
        except OSError:
            pass

    def reset(self, now):
        self.state = 'COUNTDOWN'
        self.state_t0 = now
        self.lives = START_LIVES
        self.score = 0
        self.walls_passed = 0
        self.streak = 0
        self.round_time = ROUND_TIME
        self.pose = None
        self.target_pose = None   # stickman pose dict for the current wall
        self.hole_mask = None     # cached full-frame hole for the current wall
        self.deadline = None
        self.result = None
        self.new_record = False
        self._order = []
        self._last_tick = None
        self._recent = []

    @property
    def level(self):
        return self.walls_passed // 3 + 1

    @property
    def multiplier(self):
        return min(1 + self.streak // 2, 5)

    def _pool(self):
        return POSES[:EASY_POOL] if self.level == 1 else POSES

    def new_wall(self, now):
        pool = self._pool()
        if not self._order:
            self._order = np.random.permutation(len(pool)).tolist()
        idx = self._order.pop(0)
        if self.pose is not None and pool[idx % len(pool)] is self.pose and self._order:
            self._order.append(idx)
            idx = self._order.pop(0)
        self.pose = pool[idx % len(pool)]
        self.target_pose = pose_from_angles(self.pose['angles'])
        self.hole_mask = STICK.hole_mask((FRAME_H, FRAME_W), AVATAR_ANCHOR,
                                         self.target_pose)
        self.deadline = now + self.round_time
        self.state = 'WALL'
        self.state_t0 = now
        self._recent = []

    def time_left(self, now):
        return max(0.0, (self.deadline or now) - now)

    def progress(self, now):
        return min(max(1.0 - self.time_left(now) / self.round_time, 0.0), 1.0)

    def update(self, match, now):
        """Advance the state machine. Returns the frame's outcome event or None."""
        if self.state == 'COUNTDOWN':
            elapsed = now - self.state_t0
            remaining = COUNTDOWN_SECS - int(elapsed)
            if remaining != self._last_tick:
                self._last_tick = remaining
                self.sounds.play('go' if remaining <= 0 else 'tick')
            if elapsed >= COUNTDOWN_SECS + 0.6:
                self.new_wall(now)
            return None

        if self.state == 'WALL':
            if match is not None:
                self._recent.append((now, match))
            self._recent = [(t, m) for t, m in self._recent if now - t <= GRACE_WINDOW]
            if self.time_left(now) > 0:
                return None
            if self._recent:
                match = max(m for _, m in self._recent)
            if match is not None and match >= PASS_THRESHOLD:
                self.walls_passed += 1
                self.streak += 1
                points = 100 * self.multiplier
                perfect = match >= PERFECT_MATCH
                if perfect:
                    points += 50
                self.score += points
                self.round_time = max(ROUND_TIME_MIN, self.round_time - ROUND_TIME_STEP)
                self.result = ('pass', match)
                self.sounds.play('perfect' if perfect else 'pass')
                self.popups.append(
                    (f'+{points}' + (' PERFECT!' if perfect else ''),
                     AVATAR_ANCHOR[0] - 70, AVATAR_ANCHOR[1] - 90,
                     (80, 230, 120), now))
                outcome = 'pass'
            else:
                self.lives -= 1
                self.streak = 0
                self.result = ('crash', match or 0.0)
                self.sounds.play('crash')
                self.particles.burst(np.array(AVATAR_ANCHOR, float) + [0, 60])
                outcome = 'crash'
            self.state = 'RESULT'
            self.state_t0 = now
            return outcome

        if self.state == 'RESULT':
            if now - self.state_t0 >= RESULT_SECS:
                if self.lives <= 0:
                    self.state = 'GAME_OVER'
                    self.state_t0 = now
                    self.new_record = self.score > self.high_score
                    if self.new_record:
                        self.high_score = self.score
                        self._save_high_score()
                    self.sounds.play('gameover')
                else:
                    self.new_wall(now)
            return None

        return None


# --------------------------------------------------------------- rendering

def render(world_bg, game, avatar_pose, match, seg_ok, wall_texture, cam_pip, now):
    """Compose the full game frame for the current state."""
    frame = world_bg.copy()
    h, w = frame.shape[:2]

    face = 'idle'
    if game.state == 'RESULT':
        face = 'win' if game.result[0] == 'pass' else 'hit'

    wall_on_top = None
    if game.state == 'WALL':
        p = game.progress(now)
        scale = 0.22 + 0.78 * p ** 2.2
        rim = (80, 230, 120) if (match or 0) >= PASS_THRESHOLD else (245, 245, 245)
        if scale < WALL_BEHIND_UNTIL:
            frame = composite_wall(frame, wall_texture, game.hole_mask, scale, rim)
        else:
            wall_on_top = (scale, rim)
    elif game.state == 'RESULT' and game.result[0] == 'pass':
        t = (now - game.state_t0) / RESULT_SECS
        wall_on_top = (1.0 + 2.2 * t ** 1.5, (80, 230, 120))

    STICK.draw(frame, AVATAR_ANCHOR, avatar_pose, face=face,
               seg_ok=seg_ok if game.state == 'WALL' else None)

    if wall_on_top is not None:
        scale, rim = wall_on_top
        frame = composite_wall(frame, wall_texture, game.hole_mask, scale, rim)

    # crash effects: shake + red flash + bricks
    if game.state == 'RESULT' and game.result[0] == 'crash':
        t = (now - game.state_t0) / RESULT_SECS
        amp = max(0.0, 1.0 - t * 2) * 14
        if amp > 0.5:
            dx, dy = np.random.uniform(-amp, amp, 2)
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            frame = cv2.warpAffine(frame, M, (w, h))
        flash = max(0.0, 0.55 - t)
        if flash > 0:
            red = np.full_like(frame, (40, 40, 220))
            frame = cv2.addWeighted(red, flash, frame, 1 - flash, 0)
        _centered(frame, 'CRASHED!', h // 2 - 40, 2.2, (60, 60, 255), 5)
    game.particles.step_draw(frame, 1 / 30)

    if game.state == 'RESULT' and game.result[0] == 'pass':
        _centered(frame, 'THROUGH!', h // 2 - 40, 2.2, (80, 230, 120), 5)

    # floating score popups
    keep = []
    for text, x, y, color, t0 in game.popups:
        age = now - t0
        if age < 1.1:
            _outlined(frame, text, (x, int(y - 45 * age)), 0.9, color, 2)
            keep.append((text, x, y, color, t0))
    game.popups = keep

    if game.state == 'COUNTDOWN':
        n = COUNTDOWN_SECS - int(now - game.state_t0)
        text = str(n) if n > 0 else 'GO!'
        pulse = 1.0 - (now - game.state_t0) % 1.0
        _centered(frame, text, h // 2 - 60, 2.5 + 1.5 * pulse, (255, 255, 255), 6)
        _centered(frame, 'Mirror the stickman with your body - get ready...',
                  h - 60, 0.8, (220, 220, 220), 2)

    elif game.state == 'GAME_OVER':
        frame = (frame * 0.35).astype(np.uint8)
        _centered(frame, 'GAME OVER', h // 2 - 70, 2.2, (60, 60, 255), 5)
        _centered(frame, f'SCORE  {game.score}', h // 2, 1.3, (255, 255, 255), 3)
        hs_color = (80, 230, 120) if game.new_record else (200, 200, 200)
        hs_text = f'NEW HIGH SCORE!  {game.high_score}' if game.new_record \
            else f'HIGH SCORE  {game.high_score}'
        _centered(frame, hs_text, h // 2 + 45, 1.0, hs_color, 2)
        _centered(frame, 'press SPACE to play again', h // 2 + 100, 0.8, (200, 200, 200), 2)

    # HUD
    if game.state in ('WALL', 'RESULT', 'COUNTDOWN'):
        for i in range(START_LIVES):
            draw_heart(frame, (30 + i * 42, 34), 14, (70, 70, 235), filled=i < game.lives)
        _outlined(frame, f'SCORE {game.score}', (16, 80), 0.75, (255, 255, 255), 2)
        _outlined(frame, f'LVL {game.level}  x{game.multiplier}', (16, 110), 0.65,
                  (60, 200, 255), 2)

    if game.state == 'WALL':
        _centered(frame, game.pose['name'], 40, 1.0, (255, 255, 255), 3)
        left = game.time_left(now)
        _centered(frame, f'{left:.1f}s', 76, 0.85,
                  (255, 255, 255) if left > 2 else (80, 80, 255), 2)
        # target pictogram chip
        chip_x, chip_y = w // 2 - 62, 88
        sub = frame[chip_y:chip_y + 130, chip_x:chip_x + 124]
        frame[chip_y:chip_y + 130, chip_x:chip_x + 124] = (sub * 0.45).astype(np.uint8)
        cv2.rectangle(frame, (chip_x, chip_y), (chip_x + 124, chip_y + 130),
                      (200, 200, 200), 1)
        MINI.draw_outline(frame, (chip_x + 62, chip_y + 34), game.target_pose)

        if match is not None:
            bar_w = int((w - 240) * match)
            color = (80, 230, 120) if match >= PASS_THRESHOLD else (60, 140, 255)
            cv2.rectangle(frame, (120, h - 38), (120 + bar_w, h - 20), color, -1)
            cv2.rectangle(frame, (120, h - 38), (w - 120, h - 20), (220, 220, 220), 2)
            px = 120 + int((w - 240) * PASS_THRESHOLD)
            cv2.line(frame, (px, h - 44), (px, h - 14), (255, 255, 255), 2)
            _outlined(frame, f'{match * 100:.0f}%', (26, h - 22), 0.7, color, 2)
        else:
            _centered(frame, 'Step back so the camera sees both your arms',
                      h - 24, 0.7, (0, 220, 255), 2)

    # camera preview (the "controller")
    if cam_pip is not None:
        ph, pw = cam_pip.shape[:2]
        x0, y0 = w - pw - 12, 12
        frame[y0:y0 + ph, x0:x0 + pw] = cam_pip
        cv2.rectangle(frame, (x0, y0), (x0 + pw, y0 + ph), (200, 200, 200), 2)
        cv2.putText(frame, 'camera', (x0 + 4, y0 + ph - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    return frame


def main():
    pose_det = PoseDetector(confidence=0.5, model_complexity=1)
    filters = {name: OneEuroFilter(FILTER_MIN_CUTOFF, FILTER_BETA)
               for name in PoseDetector.BODY}
    wall_texture = brick_texture(FRAME_W, FRAME_H)
    world_bg = build_world(FRAME_W, FRAME_H)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('Error: could not open webcam')
        return

    game = HoleInWallGame(time.time())
    print('Hole in the Wall - your body drives the stickman. Fit through the hole!')
    print('(s=skip, SPACE=restart, q=quit)')

    try:
        while True:
            ret, cam = cap.read()
            if not ret:
                break
            cam = cv2.resize(cv2.flip(cam, 1), (FRAME_W, FRAME_H))
            now = time.time()

            results = pose_det.detect(cam)
            # Frame is selfie-mirrored, so swap l/r to screen-side semantics —
            # the arm you see on the left drives the avatar's left arm.
            body = pose_det.get_body(results, FRAME_W, FRAME_H, mirrored=True)

            match, seg_ok = None, {}
            if body is not None:
                for name in PoseDetector.BODY:
                    body[name] = filters[name].apply(body[name])
                if game.pose is not None:
                    match, seg_ok = match_pose(body, game.pose['angles'])
            else:
                for f in filters.values():
                    f.reset()

            avatar_pose = pose_from_body(body)

            outcome = game.update(match, now)
            if outcome:
                print(f'{outcome.upper():>6}: score {game.score}  lives {game.lives}  '
                      f'level {game.level}')

            cam_small = cv2.resize(pose_det.draw(cam, results), (208, 117))
            frame = render(world_bg, game, avatar_pose, match, seg_ok,
                           wall_texture, cam_small, now)
            cv2.imshow('Hole in the Wall', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s') and game.state == 'WALL':
                game.new_wall(now)
            elif key == ord(' ') and game.state == 'GAME_OVER':
                game.reset(now)

    finally:
        cap.release()
        cv2.destroyAllWindows()
        pose_det.close()
        print(f'Final score: {game.score}  (high score: {game.high_score})')


if __name__ == '__main__':
    main()
