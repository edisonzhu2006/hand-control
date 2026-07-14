"""
Meaty stick-figure avatar rendered from live pose landmarks.

The player's webcam pose drives the avatar's limbs 1:1, but the avatar lives
at a fixed position and scale in the game world — the camera is a controller,
not the view. Limbs are drawn thick with dark outlines and round joint caps
for a chunky fighting-stickman look, and the face reacts to game events.
"""

import cv2
import numpy as np

VIS_THRESHOLD = 0.3


def _unit(v, default):
    n = float(np.linalg.norm(v))
    if n < 1e-6:
        return np.array(default, float)
    return v / n


def pose_from_body(body, vis_threshold=VIS_THRESHOLD):
    """Extract normalized limb directions from live pose landmarks.

    Args:
        body: Body dict from PoseDetector.get_body() (may be None).
        vis_threshold: Minimum visibility to trust a limb; below it the limb
            falls back to a relaxed default so the avatar never glitches.

    Returns:
        Pose dict of unit direction vectors (screen convention, y down):
        arm keys 'lu','lf','ru','rf'; leg keys 'lt','ls','rt','rs';
        plus 'lean' (torso direction) and 'head_dx' (-1..1 head offset).
    """
    def seg(a, b, default):
        if (body is None or body.get(a + '_vis', 0) < vis_threshold or
                body.get(b + '_vis', 0) < vis_threshold):
            return np.array(default, float)
        return _unit(body[b] - body[a], default)

    pose = {
        'lu': seg('l_shoulder', 'l_elbow', (-0.25, 0.97)),
        'lf': seg('l_elbow', 'l_wrist', (-0.15, 0.99)),
        'ru': seg('r_shoulder', 'r_elbow', (0.25, 0.97)),
        'rf': seg('r_elbow', 'r_wrist', (0.15, 0.99)),
        'lt': seg('l_hip', 'l_knee', (-0.10, 0.99)),
        'ls': seg('l_knee', 'l_ankle', (-0.03, 1.0)),
        'rt': seg('r_hip', 'r_knee', (0.10, 0.99)),
        'rs': seg('r_knee', 'r_ankle', (0.03, 1.0)),
        'lean': np.array([0.0, 1.0]),
        'head_dx': 0.0,
    }

    if body is not None:
        sh_ok = (body.get('l_shoulder_vis', 0) > vis_threshold and
                 body.get('r_shoulder_vis', 0) > vis_threshold)
        hip_ok = (body.get('l_hip_vis', 0) > vis_threshold and
                  body.get('r_hip_vis', 0) > vis_threshold)
        if sh_ok and hip_ok:
            shc = (body['l_shoulder'] + body['r_shoulder']) / 2
            hipc = (body['l_hip'] + body['r_hip']) / 2
            lean = _unit(hipc - shc, (0.0, 1.0))
            lean[0] = float(np.clip(lean[0], -0.35, 0.35))  # keep him upright
            pose['lean'] = _unit(lean, (0.0, 1.0))
        if sh_ok and body.get('nose_vis', 0) > vis_threshold:
            shc = (body['l_shoulder'] + body['r_shoulder']) / 2
            half_w = max(float(np.linalg.norm(
                body['r_shoulder'] - body['l_shoulder'])) / 2, 1.0)
            pose['head_dx'] = float(np.clip(
                (body['nose'][0] - shc[0]) / half_w, -1.0, 1.0))
    return pose


def pose_from_angles(angles):
    """Build a target pose (for holes/ghosts) from arm angles in degrees.

    Angles use math convention (0 = screen right, 90 = up); legs and torso
    take the relaxed defaults.
    """
    def direction(deg):
        r = np.radians(deg)
        return np.array([np.cos(r), -np.sin(r)])  # screen y grows downward

    pose = pose_from_body(None)
    pose['lu'] = direction(angles['lu'])
    pose['lf'] = direction(angles['lf'])
    pose['ru'] = direction(angles['ru'])
    pose['rf'] = direction(angles['rf'])
    return pose


class Stickman:
    """Chunky stick-figure avatar with fixed proportions."""

    def __init__(self, height=320):
        h = float(height)
        self.height = h
        self.head_r = 0.11 * h
        self.torso = 0.30 * h
        self.sh_w = 0.30 * h
        self.hip_w = 0.17 * h
        self.upper = 0.17 * h
        self.fore = 0.16 * h
        self.thigh = 0.21 * h
        self.shin = 0.20 * h
        self.limb_t = max(6, int(0.075 * h))

        self.fill = (52, 52, 215)      # meaty red
        self.outline = (30, 30, 110)
        self.extremity = (35, 35, 95)  # hands / feet
        self.glow = (90, 240, 140)     # matched-limb highlight

    def skeleton(self, anchor, pose):
        """Compute all joint positions for a pose.

        Args:
            anchor: (x, y) of the shoulder center in the destination image.
            pose: Pose dict from pose_from_body()/pose_from_angles().

        Returns:
            Dict of named joint positions (np arrays).
        """
        sc = np.array(anchor, float)
        j = {'sc': sc}
        j['l_sh'] = sc + [-self.sh_w / 2, 0]
        j['r_sh'] = sc + [self.sh_w / 2, 0]
        j['hipc'] = sc + pose['lean'] * self.torso
        j['l_hip'] = j['hipc'] + [-self.hip_w / 2, 0]
        j['r_hip'] = j['hipc'] + [self.hip_w / 2, 0]
        j['l_el'] = j['l_sh'] + pose['lu'] * self.upper
        j['l_wr'] = j['l_el'] + pose['lf'] * self.fore
        j['r_el'] = j['r_sh'] + pose['ru'] * self.upper
        j['r_wr'] = j['r_el'] + pose['rf'] * self.fore
        j['l_kn'] = j['l_hip'] + pose['lt'] * self.thigh
        j['l_an'] = j['l_kn'] + pose['ls'] * self.shin
        j['r_kn'] = j['r_hip'] + pose['rt'] * self.thigh
        j['r_an'] = j['r_kn'] + pose['rs'] * self.shin
        j['head'] = sc + [pose['head_dx'] * 0.35 * self.head_r * 2,
                          -(self.head_r * 1.45)]
        return j

    # Arm segments keyed like the game's match flags.
    ARM_SEGS = {'lu': ('l_sh', 'l_el'), 'lf': ('l_el', 'l_wr'),
                'ru': ('r_sh', 'r_el'), 'rf': ('r_el', 'r_wr')}
    LEG_SEGS = [('l_hip', 'l_kn'), ('l_kn', 'l_an'),
                ('r_hip', 'r_kn'), ('r_kn', 'r_an')]

    def _meaty_line(self, img, a, b, t, fill=None):
        pa, pb = tuple(np.round(a).astype(int)), tuple(np.round(b).astype(int))
        cv2.line(img, pa, pb, self.outline, t + 6, cv2.LINE_AA)
        cv2.line(img, pa, pb, fill or self.fill, t, cv2.LINE_AA)

    def draw(self, img, anchor, pose, face='idle', seg_ok=None):
        """Render the avatar. seg_ok tints matched arm segments green."""
        j = self.skeleton(anchor, pose)
        t = self.limb_t
        seg_ok = seg_ok or {}

        # legs behind torso
        for a, b in self.LEG_SEGS:
            self._meaty_line(img, j[a], j[b], t)
        for an, direction in (('l_an', -1), ('r_an', 1)):
            c = tuple(np.round(j[an] + [direction * t * 0.4, t * 0.25]).astype(int))
            cv2.ellipse(img, c, (int(t * 0.95), int(t * 0.55)), 0, 0, 360,
                        self.extremity, -1, cv2.LINE_AA)

        # torso: broad shoulders tapering to the hips
        quad = np.array([j['l_sh'], j['r_sh'], j['r_hip'], j['l_hip']], np.int32)
        cv2.fillPoly(img, [quad], self.fill, cv2.LINE_AA)
        cv2.polylines(img, [quad], True, self.outline, 4, cv2.LINE_AA)

        # arms (matched segments glow green)
        for key, (a, b) in self.ARM_SEGS.items():
            fill = self.glow if seg_ok.get(key) else None
            self._meaty_line(img, j[a], j[b], t, fill=fill)
        for wr in ('l_wr', 'r_wr'):
            cv2.circle(img, tuple(np.round(j[wr]).astype(int)),
                       int(t * 0.75), self.extremity, -1, cv2.LINE_AA)

        # head + face
        hc = tuple(np.round(j['head']).astype(int))
        r = int(self.head_r)
        cv2.circle(img, hc, r, self.fill, -1, cv2.LINE_AA)
        cv2.circle(img, hc, r, self.outline, 4, cv2.LINE_AA)
        self._draw_face(img, hc, r, face, pose['head_dx'])
        return j

    def _draw_face(self, img, hc, r, face, dx):
        ex = int(hc[0] + dx * r * 0.3)
        eye_y = hc[1] - r // 6
        black = (25, 25, 25)
        if face == 'hit':          # X X eyes, flat mouth
            for sx in (-1, 1):
                c = (ex + sx * r // 3, eye_y)
                s = r // 6
                cv2.line(img, (c[0] - s, c[1] - s), (c[0] + s, c[1] + s), black, 3, cv2.LINE_AA)
                cv2.line(img, (c[0] - s, c[1] + s), (c[0] + s, c[1] - s), black, 3, cv2.LINE_AA)
            cv2.line(img, (ex - r // 3, hc[1] + r // 2),
                     (ex + r // 3, hc[1] + r // 2), black, 3, cv2.LINE_AA)
        elif face == 'win':        # ^ ^ eyes, big grin
            for sx in (-1, 1):
                c = (ex + sx * r // 3, eye_y)
                cv2.ellipse(img, c, (r // 6, r // 6), 0, 200, 340, black, 3, cv2.LINE_AA)
            cv2.ellipse(img, (ex, hc[1] + r // 4), (r // 2, r // 3), 0, 20, 160,
                        black, 3, cv2.LINE_AA)
        else:                      # dot eyes, small smile
            for sx in (-1, 1):
                cv2.circle(img, (ex + sx * r // 3, eye_y), max(2, r // 8), black, -1, cv2.LINE_AA)
            cv2.ellipse(img, (ex, hc[1] + r // 4), (r // 3, r // 5), 0, 20, 160,
                        black, 3, cv2.LINE_AA)

    def draw_outline(self, img, anchor, pose, color=(235, 235, 235), t=3):
        """Thin skeleton outline (used for target-pose ghosts/pictograms)."""
        j = self.skeleton(anchor, pose)
        segs = list(self.ARM_SEGS.values()) + self.LEG_SEGS + \
            [('l_sh', 'r_sh'), ('sc', 'hipc'), ('l_hip', 'r_hip')]
        for a, b in segs:
            cv2.line(img, tuple(np.round(j[a]).astype(int)),
                     tuple(np.round(j[b]).astype(int)), color, t, cv2.LINE_AA)
        cv2.circle(img, tuple(np.round(j['head']).astype(int)),
                   int(self.head_r * 0.9), color, t, cv2.LINE_AA)
        return j

    def hole_mask(self, shape, anchor, pose, pad=1.7):
        """Person-shaped hole (255 = hole) for this pose, padded so the avatar
        fits with a little clearance."""
        h, w = shape[:2]
        mask = np.zeros((h, w), np.uint8)
        j = self.skeleton(anchor, pose)
        t = int(self.limb_t * pad * 1.6)

        def pt(p):
            return tuple(np.round(p).astype(int))

        cv2.circle(mask, pt(j['head']), int(self.head_r * pad), 255, -1)
        quad = np.array([j['l_sh'] + [-t / 2, -t / 2], j['r_sh'] + [t / 2, -t / 2],
                         j['r_hip'] + [t / 2, t / 2], j['l_hip'] + [-t / 2, t / 2]], np.int32)
        cv2.fillPoly(mask, [quad], 255)
        for a, b in list(self.ARM_SEGS.values()) + self.LEG_SEGS:
            cv2.line(mask, pt(j[a]), pt(j[b]), 255, t)
        for wr in ('l_wr', 'r_wr'):
            cv2.circle(mask, pt(j[wr]), int(t * 0.7), 255, -1)
        return mask
