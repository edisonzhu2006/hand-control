#!/usr/bin/env python3
"""
Pick-and-place game: puppet the robot arm with your own arm and move blocks
into the bin.

Your arm drives the robot (elbow and wrist as pivots, claw aims where your
fingers point). Steer the claw over a block, pinch to grab it, carry it to the
bin on the right, and spread your fingers to drop it in. Each block landed in
the bin scores a point and a new block spawns.

Controls:
    move your arm  - drive the robot arm
    pinch          - close the claw (grab a nearby block)
    spread fingers - open the claw (release / drop)
    n              - spawn an extra block
    q              - quit
"""

import sys
import time

import cv2
import numpy as np

sys.path.insert(0, '/Users/edison.zhu/hand-control')
from src.pose_detection.detector import PoseDetector
from src.pose_detection.arm_mapping import arm_to_joints, grip_openness
from src.arm.kinematics import Kinematics
from src.arm.visualizer import ArmVisualizer2D
from src.utils.filters import OneEuroFilter

ARM_CONFIG = '/Users/edison.zhu/hand-control/data/arm_configs/3dof_arm.json'
PANEL_H = 480
SIM_W = 480
WRIST_GAIN = 2.0
GRIP_SMOOTHING = 0.4
FILTER_MIN_CUTOFF = 0.8
FILTER_BETA = 0.008
FILTERED_POINTS = ('shoulder', 'elbow', 'wrist', 'index', 'thumb')

BLOCK_COLORS = [(80, 170, 255), (255, 170, 90), (170, 120, 255), (120, 220, 160)]


class PickPlaceGame:
    """Game state: blocks to grab, a bin to drop them in, and a score."""

    GRAB_RADIUS = 65.0   # mm: how close the claw must be to grab a block
    GRIP_CLOSE = 0.35    # gripper below this = closed (grab)
    GRIP_OPEN = 0.55     # gripper above this = open (release); gap = hysteresis
    GRAVITY = 2500.0     # mm/s^2 for dropped blocks
    BLOCK = 55.0         # block side length (mm)

    def __init__(self, reach, num_blocks=3):
        self.reach = reach
        self.floor_y = -0.64 * reach
        self.bin_x = (0.42 * reach, 0.88 * reach)
        self.spawn_x = (-0.88 * reach, -0.25 * reach)
        self.blocks = []
        self.held = None
        self.score = 0
        self.flash_until = 0.0
        self._spawned = 0
        for _ in range(num_blocks):
            self.spawn_block()

    def spawn_block(self):
        """Spawn a block on the floor in the spawn zone, avoiding overlaps."""
        for _ in range(20):
            x = float(np.random.uniform(*self.spawn_x))
            if all(abs(x - b['pos'][0]) > self.BLOCK * 1.5 for b in self.blocks
                   if b['state'] == 'idle'):
                break
        block = {
            'pos': np.array([x, self.floor_y + self.BLOCK / 2]),
            'color': BLOCK_COLORS[self._spawned % len(BLOCK_COLORS)],
            'state': 'idle',  # idle | held | falling
            'vy': 0.0,
        }
        self._spawned += 1
        self.blocks.append(block)
        return block

    def nearest_grabbable(self, ee):
        """Return the closest idle block within grab range of the claw, or None."""
        best, best_d = None, self.GRAB_RADIUS
        for b in self.blocks:
            if b['state'] != 'idle':
                continue
            d = float(np.linalg.norm(b['pos'] - ee))
            if d < best_d:
                best, best_d = b, d
        return best

    def update(self, ee, gripper, now, dt):
        """Advance game state one frame.

        Args:
            ee: Claw (end effector) position in world mm.
            gripper: Gripper openness 0..1.
            now: Current time (seconds).
            dt: Time since last frame (seconds).

        Returns:
            True if a block was scored this frame.
        """
        scored = False

        # Grab: closed claw near an idle block picks it up.
        if self.held is None and gripper < self.GRIP_CLOSE:
            block = self.nearest_grabbable(ee)
            if block is not None:
                block['state'] = 'held'
                self.held = block

        # Carry / release.
        if self.held is not None:
            self.held['pos'] = ee.copy()
            if gripper > self.GRIP_OPEN:
                self.held['state'] = 'falling'
                self.held['vy'] = 0.0
                self.held = None

        # Physics for falling blocks.
        for b in list(self.blocks):
            if b['state'] != 'falling':
                continue
            b['vy'] += self.GRAVITY * dt
            b['pos'][1] -= b['vy'] * dt
            if b['pos'][1] - self.BLOCK / 2 <= self.floor_y:
                b['pos'][1] = self.floor_y + self.BLOCK / 2
                b['vy'] = 0.0
                if self.bin_x[0] <= b['pos'][0] <= self.bin_x[1]:
                    self.blocks.remove(b)
                    self.score += 1
                    self.flash_until = now + 0.6
                    self.spawn_block()
                    scored = True
                else:
                    b['state'] = 'idle'  # landed outside the bin; re-grabbable

        return scored

    def _block_rect(self, viz, b):
        half = self.BLOCK / 2
        tl = viz.world_to_px(b['pos'] + np.array([-half, half]))
        br = viz.world_to_px(b['pos'] + np.array([half, -half]))
        return tl, br

    def draw_background(self, img, viz, ee, gripper, now):
        """Draw floor, bin and ground blocks (called under the arm)."""
        # Floor.
        f0 = viz.world_to_px([-self.reach, self.floor_y])
        f1 = viz.world_to_px([self.reach, self.floor_y])
        cv2.line(img, f0, f1, (90, 85, 80), 2)

        # Bin: two walls and a highlighted base; flashes green on a score.
        binc = (80, 230, 120) if now < self.flash_until else (200, 180, 90)
        wall_h = 85
        for x in self.bin_x:
            p = viz.world_to_px([x, self.floor_y])
            cv2.line(img, p, (p[0], p[1] - int(wall_h * viz.scale)), binc, 3, cv2.LINE_AA)
        b0 = viz.world_to_px([self.bin_x[0], self.floor_y])
        b1 = viz.world_to_px([self.bin_x[1], self.floor_y])
        cv2.line(img, b0, b1, binc, 4)
        cv2.putText(img, 'BIN', (b0[0] + 8, b0[1] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, binc, 2, cv2.LINE_AA)

        # Ground / falling blocks (held block is drawn in the foreground).
        grabbable = (self.nearest_grabbable(ee) if self.held is None and
                     gripper >= self.GRIP_CLOSE else None)
        for b in self.blocks:
            if b['state'] == 'held':
                continue
            tl, br = self._block_rect(viz, b)
            cv2.rectangle(img, tl, br, b['color'], -1)
            edge = (60, 255, 255) if b is grabbable else (30, 30, 30)
            cv2.rectangle(img, tl, br, edge, 2)

    def draw_foreground(self, img, viz, now):
        """Draw the held block and score HUD (called over the arm)."""
        if self.held is not None:
            tl, br = self._block_rect(viz, self.held)
            cv2.rectangle(img, tl, br, self.held['color'], -1)
            cv2.rectangle(img, tl, br, (255, 255, 255), 2)

        cv2.putText(img, f'SCORE: {self.score}', (viz.w - 150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 230, 120), 2, cv2.LINE_AA)
        if now < self.flash_until:
            cv2.putText(img, '+1', (viz.w - 150, 62),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 230, 120), 2, cv2.LINE_AA)


def main():
    pose = PoseDetector(confidence=0.5, model_complexity=1)
    kin = Kinematics.from_config(ARM_CONFIG)
    viz = ArmVisualizer2D(kin, panel_size=(PANEL_H, SIM_W))
    game = PickPlaceGame(viz.reach)

    joints = np.zeros(kin.num_joints)
    gripper = 1.0
    filters = {name: OneEuroFilter(FILTER_MIN_CUTOFF, FILTER_BETA)
               for name in FILTERED_POINTS}
    last_side = None
    last_time = time.time()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('Error: could not open webcam')
        return

    print('Pick & Place - grab blocks with a pinch, drop them in the bin. (n=new block, q=quit)')

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            frame_h, frame_w = frame.shape[:2]
            now = time.time()
            dt = min(now - last_time, 0.1)
            last_time = now

            results = pose.detect(frame)
            frame = pose.draw(frame, results)
            arm = pose.get_arm(results, frame_w, frame_h, prefer='auto')

            tracking = False
            if arm is not None:
                if arm['side'] != last_side:
                    for f in filters.values():
                        f.reset()
                    last_side = arm['side']
                for name in FILTERED_POINTS:
                    arm[name] = filters[name].apply(arm[name])

                target_joints, tracking = arm_to_joints(arm, joints, wrist_gain=WRIST_GAIN)
                if tracking:
                    joints = kin._clamp_joints(target_joints)
                    g = grip_openness(arm)
                    if g is not None:
                        gripper = GRIP_SMOOTHING * g + (1 - GRIP_SMOOTHING) * gripper
            else:
                last_side = None
                for f in filters.values():
                    f.reset()

            ee = kin.forward_kinematics(joints)[:2, 3]
            game.update(ee, gripper, now, dt)

            sim = viz.render(
                joints, gripper=gripper, tracking=tracking,
                pre_draw=lambda img: game.draw_background(img, viz, ee, gripper, now))
            game.draw_foreground(sim, viz, now)

            cam_w = int(frame_w * PANEL_H / frame_h)
            cam = cv2.resize(frame, (cam_w, PANEL_H))
            cv2.putText(cam, 'pinch = grab block  |  spread = drop  |  n new block  q quit',
                        (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            if not tracking:
                cv2.putText(cam, 'Step back so your shoulder, elbow & wrist are visible',
                            (16, PANEL_H // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 220, 255), 2, cv2.LINE_AA)

            cv2.imshow('Pick & Place', np.hstack([cam, sim]))

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('n'):
                game.spawn_block()

    finally:
        cap.release()
        cv2.destroyAllWindows()
        pose.close()
        print(f'Final score: {game.score}')


if __name__ == '__main__':
    main()
