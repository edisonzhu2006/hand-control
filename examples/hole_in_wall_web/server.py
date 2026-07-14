#!/usr/bin/env python3
"""
Hole in the Wall — WebGL edition server.

Python does everything it already did well — camera capture, MediaPipe pose
tracking, and the full game state machine from examples/hole_in_wall_game.py —
and streams game state (plus a small camera preview) over a websocket. The
browser (static/game.js, PixiJS) is a pure renderer: neon arena, glowing
avatar, bloom, WebAudio sound.

Run:
    python3 examples/hole_in_wall_web/server.py
Then play at http://localhost:8765 (opens automatically).
"""

import asyncio
import base64
import os
import json
import sys
import threading
import time
import webbrowser
from pathlib import Path

import cv2
import numpy as np
from aiohttp import web, WSMsgType

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.pose_detection.detector import PoseDetector
from src.pose_detection.tasks_detector import TasksDetector
from src.render.stickman import pose_from_body
from src.utils.filters import OneEuroFilter
from examples.hole_in_wall_game import (
    HoleInWallGame, match_pose, seg_angle, ang_diff, POSES, EASY_POOL,
    MATCH_TOLERANCE, FALLOFF, PASS_THRESHOLD, GRACE_WINDOW,
    ROUND_TIME, ROUND_TIME_MIN, ROUND_TIME_STEP, PERFECT_MATCH, START_LIVES,
    COUNTDOWN_SECS, RESULT_SECS, HIGHSCORE_PATH,
    FILTER_MIN_CUTOFF, FILTER_BETA, FRAME_W, FRAME_H)

PORT = 8765
STATIC = Path(__file__).parent / 'static'

# Poses beyond the base set: torso leans (work seated) and a squat (needs the
# player's legs in frame — gated on leg visibility). 'lean' is the target
# angle of the shoulders->hips segment (-90 = upright); 'legs' are target
# thigh/shin angles keyed like the stickman pose.
LEAN_TOLERANCE = 14.0
WEB_POSES = POSES + [
    {'name': 'LEAN LEFT', 'angles': {'lu': 200, 'lf': 200, 'ru': 20, 'rf': 20},
     'lean': -72},
    {'name': 'LEAN RIGHT', 'angles': {'lu': 160, 'lf': 160, 'ru': -20, 'rf': -20},
     'lean': -108},
    {'name': 'SQUAT', 'angles': {'lu': 180, 'lf': 180, 'ru': 0, 'rf': 0},
     'legs': {'lt': -128, 'ls': -85, 'rt': -52, 'rs': -95}, 'needs_legs': True},
    {'name': 'SQUAT GOALPOST', 'angles': {'lu': 180, 'lf': 90, 'ru': 0, 'rf': 90},
     'legs': {'lt': -128, 'ls': -85, 'rt': -52, 'rs': -95}, 'needs_legs': True},
    {'name': 'STAR JUMP', 'angles': {'lu': 135, 'lf': 135, 'ru': 45, 'rf': 45},
     'legs': {'lt': -122, 'ls': -122, 'rt': -58, 'rs': -58}, 'needs_legs': True},
    {'name': 'LUNGE LEFT', 'angles': {'lu': 180, 'lf': 180, 'ru': 0, 'rf': 0},
     'legs': {'lt': -145, 'ls': -100, 'rt': -70, 'rs': -80}, 'needs_legs': True},
    {'name': 'LUNGE RIGHT', 'angles': {'lu': 180, 'lf': 180, 'ru': 0, 'rf': 0},
     'legs': {'lt': -110, 'ls': -100, 'rt': -35, 'rs': -80}, 'needs_legs': True},
    {'name': 'KICK LEFT', 'angles': {'lu': 160, 'lf': 160, 'ru': -20, 'rf': -20},
     'legs': {'lt': -150, 'ls': -150, 'rt': -80, 'rs': -85}, 'needs_legs': True},
    {'name': 'KICK RIGHT', 'angles': {'lu': 200, 'lf': 200, 'ru': 20, 'rf': 20},
     'legs': {'lt': -100, 'ls': -95, 'rt': -30, 'rs': -30}, 'needs_legs': True},
]
WEB_POSES = WEB_POSES + [
    {'name': 'T-POSE + SMILE', 'angles': {'lu': 180, 'lf': 180, 'ru': 0, 'rf': 0},
     'face': 'smile'},
    {'name': 'ARMS UP + WOW', 'angles': {'lu': 135, 'lf': 135, 'ru': 45, 'rf': 45},
     'face': 'wow'},
    {'name': 'PEACE SIGNS', 'angles': {'lu': 180, 'lf': 90, 'ru': 0, 'rf': 90},
     'hands': 'peace'},
    {'name': 'FISTS UP', 'angles': {'lu': 180, 'lf': 90, 'ru': 0, 'rf': 90},
     'hands': 'fist'},
    {'name': 'HIGH FIVES', 'angles': {'lu': 155, 'lf': 100, 'ru': 25, 'rf': 80},
     'hands': 'open'},
    {'name': 'POINT + SMILE', 'angles': {'lu': 180, 'lf': 180, 'ru': 0, 'rf': 0},
     'hands': 'point', 'face': 'smile'},
]
LEG_POOL = [p for p in WEB_POSES if p.get('needs_legs')]
DAILY_POOL = [p for p in WEB_POSES if not p.get('needs_legs')]
FACE_HAND_POOL = [p for p in WEB_POSES if p.get('face') or p.get('hands')]


def torso_angle(body):
    """Angle of the shoulders-center -> hips-center segment, or None."""
    for k in ('l_shoulder', 'r_shoulder', 'l_hip', 'r_hip'):
        if body.get(k + '_vis', 0) < 0.3:
            return None
    shc = (body['l_shoulder'] + body['r_shoulder']) / 2
    hipc = (body['l_hip'] + body['r_hip']) / 2
    v = hipc - shc
    return float(np.degrees(np.arctan2(-v[1], v[0])))


def match_pose_ex(body, pose):
    """Arms matcher extended with optional torso lean and thigh targets."""
    match, seg_ok = match_pose(body, pose['angles'])
    if match is None:
        return None, seg_ok
    scores = [match] * 4  # arms, weighted as before

    if 'lean' in pose:
        cur = torso_angle(body)
        if cur is not None:
            err = ang_diff(cur, pose['lean'])
            seg_ok['lean'] = err < LEAN_TOLERANCE
            scores.append(float(np.clip(
                1.0 - max(err - LEAN_TOLERANCE, 0.0) / 25.0, 0.0, 1.0)))

    if 'legs' in pose:
        # Tight tolerance and double weight per thigh: with four arm segments
        # in the mean, generous leg scoring would let a standing player pass
        # a squat wall.
        for a, b, key in (('l_hip', 'l_knee', 'lt'), ('r_hip', 'r_knee', 'rt')):
            cur = seg_angle(body, a, b)
            if cur is None:
                continue
            err = ang_diff(cur, pose['legs'][key])
            seg_ok[key] = err < 15.0
            leg_score = float(np.clip(1.0 - max(err - 15.0, 0.0) / 30.0, 0.0, 1.0))
            scores.extend([leg_score, leg_score])

    return float(np.mean(scores)), seg_ok


def legs_visible(body):
    return body is not None and all(
        body.get(k + '_vis', 0) > 0.35
        for k in ('l_knee', 'r_knee', 'l_ankle', 'r_ankle'))


class WebGame(HoleInWallGame):
    """Adds a menu, endless/daily modes, extended poses, and a hold bonus."""

    HANDOFF_SECS = 3.0

    def __init__(self, now):
        self.mode = 'endless'
        self.daily_date = None
        self.daily_scores = {}
        self.legs_ok_until = 0.0
        self.two_p = False
        self.leg_mode = False
        self.players = []
        self.active_p = 0
        self.winner = None
        super().__init__(now)
        self.state = 'MENU'
        self._load_daily()

    # -- persistence for the daily mode (same file as the high score)
    def _load_daily(self):
        try:
            with open(HIGHSCORE_PATH) as f:
                self.daily_scores = json.load(f).get('daily', {})
        except (OSError, ValueError):
            self.daily_scores = {}

    def _save_high_score(self):
        try:
            os.makedirs(os.path.dirname(HIGHSCORE_PATH), exist_ok=True)
            with open(HIGHSCORE_PATH, 'w') as f:
                json.dump({'high_score': self.high_score,
                           'daily': self.daily_scores}, f)
        except OSError:
            pass

    def _blank_player(self):
        return {'score': 0, 'lives': START_LIVES, 'streak': 0,
                'round_time': ROUND_TIME, 'walls_passed': 0}

    def _stash_player(self):
        self.players[self.active_p] = {
            'score': self.score, 'lives': self.lives, 'streak': self.streak,
            'round_time': self.round_time, 'walls_passed': self.walls_passed}

    def _load_player(self, i):
        self.active_p = i
        p = self.players[i]
        self.score = p['score']
        self.lives = p['lives']
        self.streak = p['streak']
        self.round_time = p['round_time']
        self.walls_passed = p['walls_passed']

    def start(self, now, mode):
        self.mode = mode
        self.reset(now)
        self.players = [self._blank_player()
                        for _ in range(2 if self.two_p else 1)]
        self.active_p = 0
        self.winner = None
        self._hold = 0.0
        self._last_t = now
        if mode == 'daily':
            self.daily_date = time.strftime('%Y-%m-%d')
            self._rng = np.random.RandomState(int(time.strftime('%Y%m%d')))
        else:
            self._rng = np.random.RandomState()

    def reset(self, now):
        super().reset(now)
        self._hold = 0.0
        self._last_t = now
        self.last_gain = None
        self.wall_dx = 0.0
        self.slide_amp = 0.0
        self.tight = False
        self.fake_pose = None
        self._swapped = False
        self._last_px = 0.0
        self.px0 = 0.0
        self._last_sw = 180.0
        self.sw0 = 180.0
        self.depth_req = 0
        self._face = None
        self._hands = None

    def note_legs(self, body, now):
        if legs_visible(body):
            self.legs_ok_until = now + 3.0

    DAILY_POOL = None  # bound after module init; fixed so seeded RNGs agree

    def _pool(self, now=None):
        now = time.time() if now is None else now
        if self.leg_mode:
            return LEG_POOL
        if self.mode == 'daily':
            return WebGame.DAILY_POOL
        pool = WEB_POSES[:EASY_POOL] if self.level == 1 else WEB_POSES
        if now > self.legs_ok_until:
            pool = [p for p in pool if not p.get('needs_legs')]
        return pool

    SLIDE_PERIOD = 3.6      # s per sway cycle for sliding holes
    POS_FREE = 50.0         # px of free positional slack
    POS_FALLOFF = 90.0      # px beyond slack where position score hits 0

    def new_wall(self, now):
        pool = self._pool(now)
        if not self._order or getattr(self, '_order_n', None) != len(pool):
            self._order = self._rng.permutation(len(pool)).tolist()
            self._order_n = len(pool)
        idx = self._order.pop(0) % len(pool)
        if self.pose is not None and pool[idx] is self.pose and self._order:
            self._order.append(idx)     # defer, don't drop, the colliding pose
            idx = self._order.pop(0) % len(pool)
        self.pose = pool[idx]
        self.deadline = now + self.round_time
        self.state = 'WALL'
        self.state_t0 = now
        self._recent = []
        self._hold = 0.0

        # wall modifiers (rolled from the seeded rng so daily stays shared)
        lvl = self.level
        self.wall_dx = 0.0
        self.slide_amp = 0.0
        self.tight = False
        self.fake_pose = None
        self._swapped = False
        self.px0 = self._last_px    # where the player stands now = neutral
        self.sw0 = self._last_sw     # shoulder width at wall spawn (depth ref)
        self.depth_req = 0
        if lvl >= 2 and self._rng.rand() < 0.40:
            self.wall_dx = float(self._rng.choice([-1, 1]) *
                                 self._rng.uniform(80, 150))
        if lvl >= 3 and self._rng.rand() < 0.30:
            self.slide_amp = float(self._rng.uniform(50, 100))
        if lvl >= 2 and self._rng.rand() < 0.15:
            self.tight = True
        if lvl >= 3 and self._rng.rand() < 0.25:
            others = [p for p in pool if p is not self.pose]
            self.fake_pose = others[int(self._rng.rand() * len(others))]
        if (lvl >= 2 and not self.wall_dx and not self.slide_amp and
                self._rng.rand() < 0.30):
            self.depth_req = int(self._rng.choice([-1, 1]))   # -1 back, +1 closer

    @property
    def pass_threshold(self):
        return 0.85 if self.tight else PASS_THRESHOLD

    def hole_dx(self, now):
        # current horizontal hole offset (sliding holes sway over time)
        dx = getattr(self, 'wall_dx', 0.0)
        if getattr(self, 'slide_amp', 0.0):
            dx += self.slide_amp * np.sin(
                2 * np.pi * (now - self.state_t0) / self.SLIDE_PERIOD)
        return float(dx)

    def note_face_hands(self, face, hands):
        self._face = face          # {'smile','open'} or None
        self._hands = hands        # {'l','r'} gestures or None

    def face_hand_scores(self):
        # (face_score, hands_score) for the current pose's requirements,
        # or None per slot when the pose does not require it
        fs = hs = None
        req_face = self.pose.get('face') if self.pose else None
        req_hands = self.pose.get('hands') if self.pose else None
        if req_face:
            f = self._face or {'smile': 0.0, 'open': 0.0}
            fs = f['smile'] if req_face == 'smile' else f['open']
        if req_hands:
            g = self._hands or {'l': 'none', 'r': 'none'}
            shown = [side for side in ('l', 'r') if g[side] != 'none']
            if not shown:
                hs = 0.0
            else:
                hs = sum(1.0 for s in shown if g[s] == req_hands) / len(shown)
                if len(shown) == 1:
                    hs *= 0.75   # one hidden hand can't score full marks
        return fs, hs

    def note_px(self, body):
        # track horizontal shoulder-center position (-1..1) and shoulder
        # pixel width (the depth signal: closer = wider)
        if (body is not None and body.get('l_shoulder_vis', 0) > 0.3 and
                body.get('r_shoulder_vis', 0) > 0.3):
            shc = (body['l_shoulder'][0] + body['r_shoulder'][0]) / 2
            self._last_px = float(shc / FRAME_W * 2 - 1)
            sw = float(np.linalg.norm(body['r_shoulder'] - body['l_shoulder']))
            if sw > 20:
                self._last_sw = sw
        return self._last_px

    def depth_delta(self):
        # log size ratio vs wall spawn: + = stepped closer, - = stepped back
        return float(np.clip(np.log(self._last_sw / max(self.sw0, 1.0)),
                             -0.6, 0.6))

    def avatar_x(self):
        # arena offset from stepping sideways, relative to the wall-start spot
        return float(np.clip((self._last_px - self.px0) * 260, -210, 210))

    def full_match(self, body, now):
        # One weighted sum over every scored channel. Required channels whose
        # landmarks are missing score 0 (not skipped) — otherwise a standing
        # player with knees out of frame passes squat walls on arms alone, and
        # nested averaging used to dilute leg weight below the pass bar.
        arms, seg_ok = match_pose(body, self.pose['angles'])
        if arms is None:
            return None, seg_ok
        terms = [(arms, 4.0)]

        if 'lean' in self.pose:
            cur = torso_angle(body)
            if cur is None:
                lean_score = 0.0
            else:
                err = ang_diff(cur, self.pose['lean'])
                lean_score = float(np.clip(
                    1.0 - max(err - LEAN_TOLERANCE, 0.0) / 25.0, 0.0, 1.0))
            seg_ok['lean'] = lean_score >= 0.6
            terms.append((lean_score, 2.0))

        if 'legs' in self.pose:
            for a, b, key in (('l_hip', 'l_knee', 'lt'), ('r_hip', 'r_knee', 'rt')):
                cur = seg_angle(body, a, b)
                if cur is None:
                    leg_score = 0.0
                else:
                    err = ang_diff(cur, self.pose['legs'][key])
                    leg_score = float(np.clip(
                        1.0 - max(err - 15.0, 0.0) / 30.0, 0.0, 1.0))
                seg_ok[key] = leg_score >= 0.99
                terms.append((leg_score, 3.0))

        if self.wall_dx or self.slide_amp:
            err = abs(self.avatar_x() - self.hole_dx(now))
            pos = float(np.clip(
                1.0 - max(err - self.POS_FREE, 0.0) / self.POS_FALLOFF, 0.0, 1.0))
            seg_ok['pos'] = err < self.POS_FREE + 20
            terms.append((pos, 3.0))

        if self.depth_req:
            delta = self.depth_delta()
            err = max(0.0, 0.18 - self.depth_req * delta)
            depth_score = float(np.clip(1.0 - err / 0.15, 0.0, 1.0))
            seg_ok['depth'] = err < 0.04
            terms.append((depth_score, 3.0))

        fs, hs = self.face_hand_scores()
        if fs is not None:
            seg_ok['face'] = fs >= 0.6
            terms.append((fs, 2.0))
        if hs is not None:
            seg_ok['hands'] = hs >= 0.9
            terms.append((hs, 2.0))

        total_w = sum(w for _, w in terms)
        return float(sum(s * w for s, w in terms) / total_w), seg_ok

    def target_payload(self):
        if self.pose is None:
            return None
        t = dict(self.pose['angles'])
        if 'lean' in self.pose:
            t['lean'] = self.pose['lean']
        if 'legs' in self.pose:
            t['legs'] = self.pose['legs']
        return t

    def compute_points(self, match, perfect):
        bonus = int(min(self._hold, 2.0) * 30)   # up to +60 for holding
        points = 100 * self.multiplier + (50 if perfect else 0) + bonus
        if self.tight:
            points *= 2
        self.last_gain = {'points': points, 'perfect': perfect, 'bonus': bonus}
        return points

    def _on_pass(self, match, now):
        outcome = super()._on_pass(match, now)
        self.popups.clear()          # the browser draws its own popups
        if self.two_p:
            self._stash_player()
        return outcome

    def _on_crash(self, match, now):
        outcome = super()._on_crash(match, now)
        if self.two_p:
            self._stash_player()
        return outcome

    def _advance_after_result(self, now):
        if self.two_p:
            return self._advance_two_p(now)
        return super()._advance_after_result(now)

    def update(self, match, now):
        dt = min(now - self._last_t, 0.2)
        self._last_t = now

        if self.state == 'MENU':
            return None

        if self.state == 'HANDOFF':
            if now - self.state_t0 >= self.HANDOFF_SECS:
                self.new_wall(now)
            return None

        if self.state == 'WALL':
            # fake-out: the hole swaps to a different pose at half-way
            if (self.fake_pose is not None and not self._swapped and
                    self.progress(now) >= 0.5):
                self.pose = self.fake_pose
                self._swapped = True
                self._recent = []
                self._hold = 0.0
                return 'fakeout'
            if match is not None and match >= self.pass_threshold:
                self._hold += dt   # reward locking the pose early

        return super().update(match, now)

    def _advance_two_p(self, now):
        self._stash_player()
        other = 1 - self.active_p
        if self.players[other]['lives'] > 0:
            self._load_player(other)
            self.state = 'HANDOFF'
            self.state_t0 = now
            return 'handoff'
        if self.lives > 0:
            self.new_wall(now)
            return None
        # both players out
        s0, s1 = self.players[0]['score'], self.players[1]['score']
        self.winner = -1 if s0 == s1 else (0 if s0 > s1 else 1)
        self.state = 'GAME_OVER'
        self.state_t0 = now
        self.new_record = False
        self.record_score(max(s0, s1))
        return None

    def record_score(self, score):
        # persist any run's score (2P winners, abandoned runs) against the
        # high score / daily best without changing the game-over display
        changed = False
        if score > self.high_score:
            self.high_score = score
            changed = True
        if self.mode == 'daily' and self.daily_date:
            prev = self.daily_scores.get(self.daily_date, 0)
            if score > prev:
                self.daily_scores[self.daily_date] = score
                changed = True
        if changed:
            self._save_high_score()

    def _finish_game(self, now):
        if self.mode == 'daily' and self.daily_date:
            prev = self.daily_scores.get(self.daily_date, 0)
            self.daily_scores[self.daily_date] = max(prev, self.score)
        super()._finish_game(now)


LOCK = threading.Lock()
CLIENTS = set()        # connected websockets (shared with the capture thread)
LATEST = None          # most recent payload dict (with accumulated events)
FLAGS = set()          # input flags from the browser: 'restart', 'skip'
RUNNING = True


def payload_pose(pose):
    """Stickman pose dict -> JSON-safe structure."""
    out = {}
    for k, v in pose.items():
        out[k] = [float(v[0]), float(v[1])] if isinstance(v, np.ndarray) else float(v)
    return out


def capture_loop():
    """Camera + detection + game logic thread; publishes LATEST payloads.

    Wrapped in a reopen-on-failure supervisor: a macOS camera can hang
    cap.read() forever (sleep, device contention), so a watchdog thread
    releases the capture when frames stop flowing, which unblocks the read
    and lets this loop reopen the device.
    """
    global RUNNING
    pose_det = TasksDetector(confidence=0.5, model_complexity=1)
    game = WebGame(time.time())
    game.sounds.paths = {}          # browser plays the sounds instead

    while RUNNING:
        try:
            _capture_session(pose_det, game)
        except Exception:
            import traceback
            traceback.print_exc()
        if RUNNING:
            print('Camera session ended; reopening in 1s...')
            time.sleep(1.0)
    pose_det.close()
    print(f'Final score: {game.score} (high score: {game.high_score})')


def _capture_session(pose_det, game):
    """One camera session: runs until the camera fails or stalls."""
    global LATEST
    filters = {name: OneEuroFilter(FILTER_MIN_CUTOFF, FILTER_BETA)
               for name in PoseDetector.BODY}

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('Error: could not open webcam')
        time.sleep(2.0)
        return

    # Watchdog: if no frame lands for 3s, release the capture from outside,
    # which unblocks a hung cap.read().
    last_frame = [time.time()]
    session_alive = [True]

    def watchdog():
        while session_alive[0] and RUNNING:
            if time.time() - last_frame[0] > 3.0:
                print('Watchdog: camera stalled, releasing capture')
                try:
                    cap.release()
                except Exception:
                    pass
                return
            time.sleep(0.5)

    threading.Thread(target=watchdog, daemon=True).start()

    frame_i = 0
    pip_b64 = None
    print('Tracking started.')

    try:
        while RUNNING:
            ret, cam = cap.read()
            if not ret:
                break
            last_frame[0] = time.time()
            now = time.time()
            frame_i += 1
            if not CLIENTS:
                # keep frames flowing for the watchdog, skip all inference
                time.sleep(0.05)
                continue
            cam = cv2.resize(cv2.flip(cam, 1), (FRAME_W, FRAME_H))

            results = pose_det.detect(cam)
            body = pose_det.get_body(results, FRAME_W, FRAME_H, mirrored=True)
            body3d = pose_det.get_body3d(results, mirrored=True)
            if body3d and body:
                # drop joints the 2D pass can't see (world coords for
                # occluded legs are hallucinated and make the avatar flail)
                body3d = {k: v for k, v in body3d.items()
                          if body.get(k + '_vis', 0) > 0.3}

            match, seg_ok = None, {}
            if body is not None:
                for name in PoseDetector.BODY:
                    body[name] = filters[name].apply(body[name])
                game.note_legs(body, now)
                game.note_px(body)
                game.note_face_hands(
                    pose_det.face_metrics(results, FRAME_W, FRAME_H, mirrored=True),
                    pose_det.hand_gestures(results, FRAME_W, FRAME_H, mirrored=True))
                if game.pose is not None and game.state == 'WALL':
                    match, seg_ok = game.full_match(body, now)
            else:
                for f in filters.values():
                    f.reset()

            events = []
            with LOCK:
                flags = set(FLAGS)
                FLAGS.clear()
            if 'restart' in flags:
                if game.state == 'MENU':
                    game.start(now, 'endless')
                elif game.state == 'GAME_OVER':
                    game.start(now, game.mode)
            if 'daily' in flags and game.state in ('MENU', 'GAME_OVER'):
                game.start(now, 'daily')
            if 'menu' in flags and game.state != 'MENU':
                game.record_score(game.score)   # ESC must not eat a record run
                game.state = 'MENU'
                game.pose = None
            if 'toggle2p' in flags and game.state == 'MENU':
                game.two_p = not game.two_p
            if 'togglelegs' in flags and game.state == 'MENU':
                game.leg_mode = not game.leg_mode
            if 'skip' in flags and game.state == 'WALL':
                game.new_wall(now)

            prev_state = game.state
            prev_tick = game._last_tick
            outcome = game.update(match, now)
            if outcome:
                events.append(outcome)
            if game.state == 'COUNTDOWN' and game._last_tick != prev_tick:
                events.append('go' if game._last_tick <= 0 else 'tick')
            if game.state == 'GAME_OVER' and prev_state != 'GAME_OVER':
                events.append('gameover')

            # small camera preview at ~half the frame rate; sent only on
            # frames where it was actually re-encoded (client keeps the last)
            pip_b64 = None
            if frame_i % 2 == 0:
                small = cv2.resize(pose_det.draw(cam, results), (480, 270))
                ok, buf = cv2.imencode('.jpg', small, [cv2.IMWRITE_JPEG_QUALITY, 72])
                if ok:
                    pip_b64 = base64.b64encode(buf).decode('ascii')

            data = {
                'seq': frame_i,
                'state': game.state,
                'pose': payload_pose(pose_from_body(body)),
                'targetAngles': game.target_payload(),
                'poseName': game.pose['name'] if game.pose else '',
                'match': match,
                'segOk': seg_ok,
                'tracked': body is not None,
                'progress': game.progress(now) if game.state == 'WALL' else 0.0,
                'timeLeft': game.time_left(now) if game.state == 'WALL' else 0.0,
                'resultT': ((now - game.state_t0) / RESULT_SECS
                            if game.state == 'RESULT' else 0.0),
                'outcome': game.result[0] if game.result else None,
                'countdown': (COUNTDOWN_SECS - int(now - game.state_t0)
                              if game.state == 'COUNTDOWN' else None),
                'score': game.score,
                'lives': game.lives,
                'level': game.level,
                'mult': game.multiplier,
                'highScore': game.high_score,
                'newRecord': game.new_record,
                'mode': game.mode,
                'dailyDate': game.daily_date,
                'dailyBest': game.daily_scores.get(game.daily_date, 0)
                             if game.daily_date else 0,
                'lastGain': game.last_gain,
                'holdT': round(getattr(game, '_hold', 0.0), 2),
                'twoP': game.two_p,
                'legMode': game.leg_mode,
                'activeP': game.active_p,
                'players': ([{'score': p['score'], 'lives': p['lives']}
                             for p in game.players]
                            if game.two_p and len(game.players) == 2 else None),
                'winner': game.winner,
                'faceReq': game.pose.get('face') if game.pose else None,
                'handsReq': game.pose.get('hands') if game.pose else None,
                'faceLive': game._face,
                'handShapes': pose_det.hand_shapes(mirrored=True),
                'pose3d': ({k: [round(float(v[0]), 3), round(float(v[1]), 3),
                                round(float(v[2]), 3)] for k, v in body3d.items()}
                           if body3d else None),
                'depthReq': game.depth_req,
                'depthDelta': round(game.depth_delta(), 3),
                'ax': round(game.avatar_x(), 1),
                'holeDx': round(game.hole_dx(now), 1) if game.pose else 0.0,
                'passThreshold': game.pass_threshold,
                'tight': game.tight,
                'pip': pip_b64,
            }
            with LOCK:
                if LATEST is not None and LATEST.get('events'):
                    events = LATEST['events'] + events   # keep undelivered events
                data['events'] = events
                LATEST = data
    finally:
        session_alive[0] = False
        try:
            cap.release()
        except Exception:
            pass


async def ws_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    request.app['clients'].add(ws)
    CLIENTS.add(ws)
    try:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                try:
                    key = json.loads(msg.data).get('key')
                except (ValueError, AttributeError, TypeError):
                    continue
                if key in ('restart', 'skip', 'daily', 'menu', 'toggle2p', 'togglelegs'):
                    with LOCK:
                        FLAGS.add(key)
    finally:
        request.app['clients'].discard(ws)
        CLIENTS.discard(ws)
    return ws


async def index(_request):
    return web.FileResponse(STATIC / 'index.html')


async def broadcaster(app):
    global LATEST
    last_seq = None
    try:
        while True:
            await asyncio.sleep(1 / 30)
            with LOCK:
                data = LATEST
                if data is not None and app['clients']:
                    LATEST = dict(data, events=[])   # events delivered once
            if data is None or not app['clients']:
                continue
            if data.get('seq') == last_seq and not data.get('events'):
                continue                             # nothing new to send
            last_seq = data.get('seq')
            text = json.dumps(data)
            for ws in list(app['clients']):
                try:
                    await ws.send_str(text)
                except ConnectionError:
                    app['clients'].discard(ws)
    except asyncio.CancelledError:
        pass


async def on_startup(app):
    app['clients'] = set()
    app['bg'] = asyncio.create_task(broadcaster(app))
    threading.Thread(target=capture_loop, daemon=True).start()


async def on_cleanup(app):
    global RUNNING
    RUNNING = False
    app['bg'].cancel()


def main():
    app = web.Application()
    app.router.add_get('/', index)
    app.router.add_get('/ws', ws_handler)
    app.router.add_static('/static/', STATIC)
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)

    url = f'http://localhost:{PORT}'
    print(f'Hole in the Wall (WebGL) at {url}')
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    web.run_app(app, port=PORT, print=None)


if __name__ == '__main__':
    main()


WebGame.DAILY_POOL = DAILY_POOL
