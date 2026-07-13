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
import json
import sys
import threading
import time
import webbrowser
from pathlib import Path

import cv2
import numpy as np
from aiohttp import web, WSMsgType

sys.path.insert(0, '/Users/edison.zhu/hand-control')
from src.pose_detection.detector import PoseDetector
from src.render.stickman import pose_from_body
from src.utils.filters import OneEuroFilter
from examples.hole_in_wall_game import (
    HoleInWallGame, match_pose, COUNTDOWN_SECS, RESULT_SECS,
    FILTER_MIN_CUTOFF, FILTER_BETA, FRAME_W, FRAME_H)

PORT = 8765
STATIC = Path(__file__).parent / 'static'

LOCK = threading.Lock()
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
    """Camera + detection + game logic thread; publishes LATEST payloads."""
    global LATEST, RUNNING
    pose_det = PoseDetector(confidence=0.5, model_complexity=1)
    filters = {name: OneEuroFilter(FILTER_MIN_CUTOFF, FILTER_BETA)
               for name in PoseDetector.BODY}

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('Error: could not open webcam')
        RUNNING = False
        return

    game = HoleInWallGame(time.time())
    game.sounds.paths = {}          # browser plays the sounds instead
    frame_i = 0
    pip_b64 = None
    print('Tracking started.')

    try:
        while RUNNING:
            ret, cam = cap.read()
            if not ret:
                break
            cam = cv2.resize(cv2.flip(cam, 1), (FRAME_W, FRAME_H))
            now = time.time()
            frame_i += 1

            results = pose_det.detect(cam)
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

            events = []
            with LOCK:
                flags = set(FLAGS)
                FLAGS.clear()
            if 'restart' in flags and game.state == 'GAME_OVER':
                game.reset(now)
                events.append('go')
            if 'skip' in flags and game.state == 'WALL':
                game.new_wall(now)

            prev_state = game.state
            prev_tick = game._last_tick
            outcome = game.update(match, now)
            if outcome:
                perfect = outcome == 'pass' and game.result[1] >= 0.90
                events.append('perfect' if perfect else outcome)
            if game.state == 'COUNTDOWN' and game._last_tick != prev_tick:
                events.append('go' if game._last_tick <= 0 else 'tick')
            if game.state == 'GAME_OVER' and prev_state != 'GAME_OVER':
                events.append('gameover')

            # small camera preview at ~half the frame rate
            if frame_i % 2 == 0:
                small = cv2.resize(pose_det.draw(cam, results), (208, 117))
                ok, buf = cv2.imencode('.jpg', small, [cv2.IMWRITE_JPEG_QUALITY, 60])
                if ok:
                    pip_b64 = base64.b64encode(buf).decode('ascii')

            data = {
                'state': game.state,
                'pose': payload_pose(pose_from_body(body)),
                'targetAngles': game.pose['angles'] if game.pose else None,
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
                'pip': pip_b64,
            }
            with LOCK:
                if LATEST is not None and LATEST.get('events'):
                    events = LATEST['events'] + events   # keep undelivered events
                data['events'] = events
                LATEST = data
    finally:
        cap.release()
        pose_det.close()
        print(f'Final score: {game.score} (high score: {game.high_score})')


async def ws_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    request.app['clients'].add(ws)
    try:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                try:
                    key = json.loads(msg.data).get('key')
                except ValueError:
                    continue
                if key in ('restart', 'skip'):
                    with LOCK:
                        FLAGS.add(key)
    finally:
        request.app['clients'].discard(ws)
    return ws


async def index(_request):
    return web.FileResponse(STATIC / 'index.html')


async def broadcaster(app):
    global LATEST
    try:
        while True:
            await asyncio.sleep(1 / 30)
            with LOCK:
                data = LATEST
                if data is not None:
                    LATEST = dict(data, events=[])   # events delivered once
            if data is None or not app['clients']:
                continue
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
