"""
2D visualizer for a planar robotic arm, rendered with OpenCV.

Draws the arm as a connected chain of links with a simple two-jaw gripper at the
end effector, plus the current target and a small heads-up display. Intended for
the virtual-arm proof of concept, but reusable anywhere a lightweight arm render
is needed.
"""

import cv2
import numpy as np

from .kinematics import Kinematics


class ArmVisualizer2D:
    """Render a planar arm (motion in the XY plane) onto a fixed-size panel."""

    def __init__(self, kinematics: Kinematics, panel_size=(480, 480),
                 base_offset=(0.5, 0.5), margin=0.12, bg_color=(35, 32, 30)):
        """Initialize the visualizer.

        Args:
            kinematics: Kinematics solver for the arm being drawn.
            panel_size: (height, width) of the render panel in pixels.
            base_offset: Base position as a fraction of (width, height).
            margin: Fractional border kept clear around the reachable circle.
            bg_color: Panel background color (BGR).
        """
        self.kin = kinematics
        self.h, self.w = panel_size
        self.bg_color = bg_color

        # Maximum reach = sum of link lengths; used to scale world -> pixels.
        self.reach = float(sum(abs(dh.a) for dh in kinematics.dh_params)) or 1.0
        self.base_px = np.array([self.w * base_offset[0], self.h * base_offset[1]])
        self.scale = (min(self.h, self.w) * (0.5 - margin)) / self.reach

    def world_to_px(self, p):
        """Convert an (x, y[, z]) world point (mm) to panel pixel coordinates."""
        x = self.base_px[0] + p[0] * self.scale
        y = self.base_px[1] - p[1] * self.scale  # invert y: image rows grow downward
        return (int(round(x)), int(round(y)))

    def render(self, joint_angles, target_world=None, gripper=1.0,
               gesture=None, tracking=True, pre_draw=None):
        """Render the current arm state and return a BGR image.

        Args:
            joint_angles: Current joint angles.
            target_world: Optional (x, y) target to draw as a crosshair.
            gripper: Gripper openness 0 (closed) .. 1 (open).
            gesture: Optional recognized gesture name to show.
            tracking: Whether a hand is currently driving the arm.
            pre_draw: Optional callable(img) invoked after the background but
                before the arm, for drawing scene elements under the links.

        Returns:
            img: BGR image of shape (panel_h, panel_w, 3).
        """
        img = np.full((self.h, self.w, 3), self.bg_color, dtype=np.uint8)

        # Reachable workspace boundary + axes through the base.
        cv2.circle(img, self.world_to_px([0, 0]),
                   int(self.reach * self.scale), (70, 66, 62), 1, cv2.LINE_AA)
        bx, by = self.world_to_px([0, 0])
        cv2.line(img, (0, by), (self.w, by), (55, 52, 48), 1)
        cv2.line(img, (bx, 0), (bx, self.h), (55, 52, 48), 1)

        if pre_draw is not None:
            pre_draw(img)

        pts = self.kin.joint_positions(joint_angles)  # (num_joints + 1, 3)
        px = [self.world_to_px(p) for p in pts]

        # Target crosshair (drawn under the arm).
        if target_world is not None:
            tx, ty = self.world_to_px(target_world)
            cv2.drawMarker(img, (tx, ty), (90, 90, 240), cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)
            cv2.circle(img, (tx, ty), 12, (90, 90, 240), 1, cv2.LINE_AA)

        # Links.
        for i in range(len(px) - 1):
            cv2.line(img, px[i], px[i + 1], (150, 175, 40), 8, cv2.LINE_AA)

        # Joints (base = square, intermediate = amber dots).
        for i, p in enumerate(px):
            if i == 0:
                cv2.rectangle(img, (p[0] - 9, p[1] - 9), (p[0] + 9, p[1] + 9),
                              (210, 210, 210), -1)
            elif i < len(px) - 1:
                cv2.circle(img, p, 7, (60, 210, 255), -1, cv2.LINE_AA)
                cv2.circle(img, p, 7, (30, 30, 30), 1, cv2.LINE_AA)

        self._draw_gripper(img, pts, gripper)
        self._draw_hud(img, pts, gripper, gesture, tracking)
        return img

    def _draw_gripper(self, img, pts, gripper):
        """Draw a two-jaw gripper at the end effector, aligned with the last link."""
        ee = pts[-1][:2]
        prev = pts[-2][:2]
        d = ee - prev
        n = np.linalg.norm(d)
        if n < 1e-6:
            return
        d = d / n
        perp = np.array([-d[1], d[0]])

        # Convert world directions to pixel space (y is inverted on screen).
        d_px = np.array([d[0], -d[1]])
        perp_px = np.array([perp[0], -perp[1]])

        base = np.array(self.world_to_px(ee), dtype=float)
        half_px = 5.0 + 15.0 * float(np.clip(gripper, 0.0, 1.0))
        jaw_px = 24.0
        for side in (1.0, -1.0):
            root = base + perp_px * (side * half_px)
            tip = root + d_px * jaw_px
            cv2.line(img, tuple(base.astype(int)), tuple(root.astype(int)),
                     (120, 255, 120), 3, cv2.LINE_AA)
            cv2.line(img, tuple(root.astype(int)), tuple(tip.astype(int)),
                     (120, 255, 120), 3, cv2.LINE_AA)

    def _draw_hud(self, img, pts, gripper, gesture, tracking):
        """Draw the heads-up display text overlay."""
        ee = pts[-1]
        lines = [
            ("VIRTUAL ARM", (205, 205, 205)),
            (f"EE: ({ee[0]:.0f}, {ee[1]:.0f}) mm", (170, 170, 170)),
            (f"Gripper: {'OPEN' if gripper > 0.5 else 'CLOSED'} ({gripper * 100:.0f}%)",
             (120, 255, 120)),
        ]
        y = 24
        for txt, col in lines:
            cv2.putText(img, txt, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
            y += 22

        status = "TRACKING" if tracking else "NO HAND"
        status_col = (120, 220, 120) if tracking else (90, 90, 240)
        cv2.putText(img, status, (12, self.h - 14), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, status_col, 2, cv2.LINE_AA)
        if gesture:
            cv2.putText(img, gesture, (self.w - 160, self.h - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 210, 255), 2, cv2.LINE_AA)
