/* Hole in the Wall — PixiJS/WebGL renderer.
 * Python streams game state over a websocket; this file only draws and plays
 * sound. Geometry mirrors src/render/stickman.py (Stickman height 300).
 */

const W = 960, H = 540;
const ANCHOR = { x: W / 2, y: 195 };
const PASS_THRESHOLD = 0.70;

// Stickman proportions (height 300; slimmer + smaller head than the Python
// renderer for a cleaner fighting-stickman look).
const SM = (() => {
  const h = 300;
  return {
    headR: 0.095 * h, torso: 0.31 * h, shW: 0.26 * h, hipW: 0.13 * h,
    upper: 0.17 * h, fore: 0.16 * h, thigh: 0.21 * h, shin: 0.20 * h,
    t: 0.062 * h,
  };
})();

const COL = {
  fill: 0xf0ede4, outline: 0x23232b, extremity: 0x23232b, glow: 0x7ce89a,
  rim: 0xf2eee6, rimFit: 0x78e678, sky0: 0x121218, sky1: 0x22222e,
  grid: 0x2d4a52, gridGlow: 0x3e6a74, brick: 0xb8623c, mortar: 0x8a4527,
};

/* ---------------------------------------------------------------- pixi app */

const app = new PIXI.Application({
  width: W, height: H, antialias: true, background: 0x121218,
  preserveDrawingBuffer: true,   // makes the WebGL canvas screenshot-able
  resolution: Math.min(window.devicePixelRatio || 1, 2),   // crisp when CSS-scaled up
  autoDensity: true,
});
document.getElementById('wrap').prepend(app.view);

const root = new PIXI.Container();      // shakeable + zoom-pulsable
root.pivot.set(W / 2, H / 2);
root.position.set(W / 2, H / 2);
app.stage.addChild(root);

const bgLayer = new PIXI.Container();
const wallBehind = new PIXI.Container();
const avatarG = new PIXI.Graphics();
const wallFront = new PIXI.Container();
const fxG = new PIXI.Graphics();
const vignetteLayer = new PIXI.Container();
const uiLayer = new PIXI.Container();
root.addChild(bgLayer, wallBehind, avatarG, wallFront, fxG, vignetteLayer, uiLayer);

try {
  if (app.renderer.type === PIXI.RENDERER_TYPE.WEBGL) {
    root.filters = [new PIXI.filters.AdvancedBloomFilter({
      threshold: 0.55, bloomScale: 0.9, brightness: 1.0, blur: 4,
    })];
  }
} catch (e) { /* pixi-filters CDN missing — fine without bloom */ }

/* -------------------------------------------------------------- background */

function buildBackground() {
  const cv = document.createElement('canvas');
  cv.width = W; cv.height = H;
  const ctx = cv.getContext('2d');
  const horizon = H * 0.52;

  // deep studio sky
  const sky = ctx.createLinearGradient(0, 0, 0, horizon);
  sky.addColorStop(0, '#0d0d16');
  sky.addColorStop(0.7, '#1b1d30');
  sky.addColorStop(1, '#2c2e48');
  ctx.fillStyle = sky;
  ctx.fillRect(0, 0, W, horizon);

  // warm stage glow behind the avatar
  const glow = ctx.createRadialGradient(W / 2, ANCHOR.y + 80, 40, W / 2, ANCHOR.y + 80, 360);
  glow.addColorStop(0, 'rgba(228,206,160,0.17)');
  glow.addColorStop(0.5, 'rgba(214,186,140,0.08)');
  glow.addColorStop(1, 'rgba(200,170,120,0)');
  ctx.fillStyle = glow;
  ctx.fillRect(0, 0, W, H);

  // lighting truss with three fixtures + visible cones
  ctx.fillStyle = '#08080d';
  ctx.fillRect(0, 16, W, 7);
  for (const fx of [W * 0.18, W * 0.5, W * 0.82]) {
    ctx.fillStyle = '#0a0a10';
    ctx.fillRect(fx - 12, 20, 24, 20);
    ctx.fillStyle = '#ffd98c';
    ctx.beginPath(); ctx.arc(fx, 42, 5, 0, 7); ctx.fill();
    const cone = ctx.createLinearGradient(fx, 40, fx, H * 0.92);
    cone.addColorStop(0, 'rgba(240,224,185,0.13)');
    cone.addColorStop(1, 'rgba(240,224,185,0)');
    ctx.fillStyle = cone;
    ctx.beginPath();
    ctx.moveTo(fx - 8, 44); ctx.lineTo(fx + 8, 44);
    ctx.lineTo(fx + (fx < W / 2 ? 150 : fx > W / 2 ? -150 : 0) + 130, H * 0.92);
    ctx.lineTo(fx + (fx < W / 2 ? 150 : fx > W / 2 ? -150 : 0) - 130, H * 0.92);
    ctx.closePath(); ctx.fill();
  }

  // crowd bleachers flanking the stage (silhouette rows of heads)
  for (const side of [0, 1]) {
    const xa = side === 0 ? 0 : W * 0.72;
    for (let row = 0; row < 3; row++) {
      const y = horizon - 14 - row * 22;
      ctx.fillStyle = `rgba(7,8,12,${0.95 - row * 0.12})`;
      ctx.fillRect(xa, y + 8, W * 0.28, 22);
      for (let x = xa + 8 + (row % 2) * 10; x < xa + W * 0.28 - 6; x += 21) {
        ctx.beginPath();
        ctx.arc(x, y + 8, 8 + (Math.random() * 2 | 0), Math.PI, 2 * Math.PI);
        ctx.fill();
      }
    }
  }

  // LED strip along the horizon
  const led = ctx.createLinearGradient(0, horizon - 3, 0, horizon + 5);
  led.addColorStop(0, 'rgba(122,214,255,0.0)');
  led.addColorStop(0.5, 'rgba(122,214,255,0.5)');
  led.addColorStop(1, 'rgba(122,214,255,0.0)');
  ctx.fillStyle = led;
  ctx.fillRect(0, horizon - 4, W, 9);

  // floor
  const floor = ctx.createLinearGradient(0, horizon, 0, H);
  floor.addColorStop(0, '#262b34');
  floor.addColorStop(1, '#0b0d11');
  ctx.fillStyle = floor;
  ctx.fillRect(0, horizon, W, H - horizon);

  // stage platform under the avatar
  ctx.fillStyle = 'rgba(10,11,16,0.55)';
  ctx.beginPath(); ctx.ellipse(W / 2, H * 0.86, 300, 52, 0, 0, 7); ctx.fill();
  ctx.strokeStyle = 'rgba(122,214,255,0.22)';
  ctx.lineWidth = 2;
  ctx.beginPath(); ctx.ellipse(W / 2, H * 0.86, 300, 52, 0, 0, 7); ctx.stroke();
  ctx.strokeStyle = 'rgba(122,214,255,0.10)';
  ctx.beginPath(); ctx.ellipse(W / 2, H * 0.86, 262, 43, 0, 0, 7); ctx.stroke();

  // spotlight pool + soft vertical reflection streak
  const pool = ctx.createRadialGradient(W / 2, H * 0.87, 20, W / 2, H * 0.87, 250);
  pool.addColorStop(0, 'rgba(245,233,208,0.20)');
  pool.addColorStop(1, 'rgba(245,233,208,0)');
  ctx.fillStyle = pool;
  ctx.fillRect(0, horizon, W, H - horizon);
  const refl = ctx.createLinearGradient(0, horizon, 0, H);
  refl.addColorStop(0, 'rgba(228,206,160,0.10)');
  refl.addColorStop(1, 'rgba(228,206,160,0)');
  ctx.fillStyle = refl;
  ctx.fillRect(W / 2 - 90, horizon, 180, H - horizon);

  // dust motes
  for (let i = 0; i < 26; i++) {
    ctx.fillStyle = `rgba(255,255,255,${Math.random() * 0.18 + 0.04})`;
    ctx.beginPath();
    ctx.arc(Math.random() * W, 30 + Math.random() * (horizon - 60),
      Math.random() * 1.2 + 0.3, 0, 7);
    ctx.fill();
  }
  bgLayer.addChild(new PIXI.Sprite(PIXI.Texture.from(cv)));

  // quiet perspective grid on the floor only
  const g = new PIXI.Graphics();
  g.lineStyle(1, 0x39424e, 0.22);
  for (let x = -W; x <= 2 * W; x += W / 6) {
    g.moveTo(x, H).lineTo(W / 2, horizon);
  }
  for (let i = 1; i <= 6; i++) {
    const y = horizon + (H - horizon) * Math.pow(i / 6, 1.8);
    g.lineStyle(1, 0x39424e, 0.10 + 0.16 * (i / 6));
    g.moveTo(0, y).lineTo(W, y);
  }
  bgLayer.addChild(g);
}

function buildVignette() {
  // cinematic dark corners over the scene, under the UI text
  const cv = document.createElement('canvas');
  cv.width = W; cv.height = H;
  const ctx = cv.getContext('2d');
  const vg = ctx.createRadialGradient(W / 2, H * 0.46, H * 0.42, W / 2, H * 0.5, H * 0.95);
  vg.addColorStop(0, 'rgba(0,0,0,0)');
  vg.addColorStop(1, 'rgba(0,0,0,0.42)');
  ctx.fillStyle = vg;
  ctx.fillRect(0, 0, W, H);
  return new PIXI.Sprite(PIXI.Texture.from(cv));
}

function lerpColor(c0, c1, a) {
  const r = ((c0 >> 16) & 255) * (1 - a) + ((c1 >> 16) & 255) * a;
  const g = ((c0 >> 8) & 255) * (1 - a) + ((c1 >> 8) & 255) * a;
  const b = (c0 & 255) * (1 - a) + (c1 & 255) * a;
  return (r << 16) | (g << 8) | b;
}

/* ----------------------------------------------------- skeleton + silhouette */

function dirFromAngle(deg) {
  const r = deg * Math.PI / 180;
  return [Math.cos(r), -Math.sin(r)];
}

function defaultPose() {
  return {
    lu: [-0.25, 0.97], lf: [-0.15, 0.99], ru: [0.25, 0.97], rf: [0.15, 0.99],
    lt: [-0.10, 0.99], ls: [-0.03, 1.0], rt: [0.10, 0.99], rs: [0.03, 1.0],
    lean: [0, 1], head_dx: 0,
  };
}

function poseFromAngles(angles) {
  const p = defaultPose();
  p.lu = dirFromAngle(angles.lu); p.lf = dirFromAngle(angles.lf);
  p.ru = dirFromAngle(angles.ru); p.rf = dirFromAngle(angles.rf);
  if (angles.lean !== undefined) p.lean = dirFromAngle(angles.lean);
  if (angles.legs) {
    for (const k of ['lt', 'ls', 'rt', 'rs']) p[k] = dirFromAngle(angles.legs[k]);
  }
  return p;
}

function skeleton(anchor, pose, dims = SM) {
  const add = (p, d, len) => [p[0] + d[0] * len, p[1] + d[1] * len];
  const sc = [anchor.x, anchor.y];
  const j = { sc };
  j.lSh = [sc[0] - dims.shW / 2, sc[1]];
  j.rSh = [sc[0] + dims.shW / 2, sc[1]];
  j.hipc = add(sc, pose.lean, dims.torso);
  j.lHip = [j.hipc[0] - dims.hipW / 2, j.hipc[1]];
  j.rHip = [j.hipc[0] + dims.hipW / 2, j.hipc[1]];
  j.lEl = add(j.lSh, pose.lu, dims.upper); j.lWr = add(j.lEl, pose.lf, dims.fore);
  j.rEl = add(j.rSh, pose.ru, dims.upper); j.rWr = add(j.rEl, pose.rf, dims.fore);
  j.lKn = add(j.lHip, pose.lt, dims.thigh); j.lAn = add(j.lKn, pose.ls, dims.shin);
  j.rKn = add(j.rHip, pose.rt, dims.thigh); j.rAn = add(j.rKn, pose.rs, dims.shin);
  j.head = [sc[0] + pose.head_dx * 0.35 * dims.headR * 2, sc[1] - dims.headR * 1.45];
  return j;
}

const ARM_SEGS = { lu: ['lSh', 'lEl'], lf: ['lEl', 'lWr'], ru: ['rSh', 'rEl'], rf: ['rEl', 'rWr'] };
const LEG_SEGS = [['lHip', 'lKn'], ['lKn', 'lAn'], ['rHip', 'rKn'], ['rKn', 'rAn']];

/* --------------------------------------------------------------- the wall */

let wallSprite = null;
let wallForPose = null;

// Wall texture extends below the frame so the scaled-down (distant) wall
// doesn't visibly float above the floor.
const WALL_H = H + 220;
// Scale the wall about the hole's center, not the frame center — the hole
// then stays locked on the avatar for the whole approach.
const HOLE_CY = 280;

function buildWallTexture(angles) {
  // Offscreen 2D canvas: bricks, then punch the silhouette hole with
  // destination-out; the pre-stroked wider silhouette leaves a rim ring.
  const cv = document.createElement('canvas');
  cv.width = W; cv.height = WALL_H;
  const ctx = cv.getContext('2d');

  // base coat with top-down lighting
  const base = ctx.createLinearGradient(0, 0, 0, WALL_H);
  base.addColorStop(0, '#b06038');
  base.addColorStop(1, '#7e4226');
  ctx.fillStyle = base;
  ctx.fillRect(0, 0, W, WALL_H);

  // bricks: tonal variation + a light top edge and dark bottom edge per brick
  const step = 44;
  for (let y = 0; y < WALL_H; y += step) {
    const off = (y / step) % 2 ? step : 0;
    for (let x = off - step * 2; x < W; x += step * 2) {
      const shade = 1 - 0.25 * (y / WALL_H);
      const v = (Math.random() * 14 - 7) * shade;
      const r = (172 + v) * shade, gr = (92 + v * 0.6) * shade, b = (56 + v * 0.4) * shade;
      ctx.fillStyle = `rgb(${r | 0},${gr | 0},${b | 0})`;
      ctx.fillRect(x + 2, y + 2, step * 2 - 4, step - 4);
      ctx.fillStyle = 'rgba(255,235,210,0.10)';
      ctx.fillRect(x + 2, y + 2, step * 2 - 4, 3);
      ctx.fillStyle = 'rgba(30,10,4,0.22)';
      ctx.fillRect(x + 2, y + step - 5, step * 2 - 4, 3);
    }
  }
  // soft mortar joints
  ctx.strokeStyle = 'rgba(58,26,14,0.5)';
  ctx.lineWidth = 2;
  for (let y = 0; y < WALL_H; y += step) {
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
    const off = (y / step) % 2 ? step : 0;
    for (let x = off; x < W; x += step * 2) {
      ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x, y + step); ctx.stroke();
    }
  }
  // speckle noise so the surface isn't flat
  for (let i = 0; i < 1200; i++) {
    ctx.fillStyle = Math.random() < 0.5
      ? 'rgba(255,220,190,0.05)' : 'rgba(40,12,4,0.07)';
    ctx.fillRect(Math.random() * W, Math.random() * WALL_H, 2, 2);
  }
  // vignette on the wall itself, centered on the hole
  const vg = ctx.createRadialGradient(W / 2, HOLE_CY, H * 0.3, W / 2, HOLE_CY, WALL_H * 0.85);
  vg.addColorStop(0, 'rgba(0,0,0,0)'); vg.addColorStop(1, 'rgba(16,4,0,0.45)');
  ctx.fillStyle = vg; ctx.fillRect(0, 0, W, WALL_H);

  const pose = poseFromAngles(angles);
  const j = skeleton(ANCHOR, pose);
  const pad = 1.7, t = SM.t * pad * 1.6;

  const tracePath = () => {
    ctx.beginPath();
    for (const [a, b] of [...Object.values(ARM_SEGS), ...LEG_SEGS]) {
      ctx.moveTo(...j[a]); ctx.lineTo(...j[b]);
    }
  };
  const drawSilhouette = (extra) => {
    ctx.lineCap = 'round'; ctx.lineJoin = 'round';
    ctx.lineWidth = t + extra;
    tracePath(); ctx.stroke();
    // capsule torso + shoulder bar, matching the avatar's construction
    ctx.lineWidth = t * 1.7 + extra;
    ctx.beginPath(); ctx.moveTo(...j.sc); ctx.lineTo(...j.hipc); ctx.stroke();
    ctx.lineWidth = t + extra;
    ctx.beginPath(); ctx.moveTo(...j.lSh); ctx.lineTo(...j.rSh); ctx.stroke();
    ctx.beginPath();
    ctx.arc(j.head[0], j.head[1], SM.headR * pad + extra / 2, 0, 7);
    ctx.fill();
    for (const wr of ['lWr', 'rWr']) {
      ctx.beginPath(); ctx.arc(j[wr][0], j[wr][1], t * 0.7 + extra / 2, 0, 7); ctx.fill();
    }
  };

  // rim ring first (warm white), then cut the hole, then a dark inner lip
  // (source-atop only paints on remaining wall) so the cutout reads as deep
  ctx.strokeStyle = ctx.fillStyle = 'rgba(248,240,228,0.95)';
  drawSilhouette(12);
  ctx.globalCompositeOperation = 'destination-out';
  ctx.strokeStyle = ctx.fillStyle = '#000';
  drawSilhouette(0);
  ctx.globalCompositeOperation = 'source-atop';
  ctx.strokeStyle = ctx.fillStyle = 'rgba(20,8,2,0.5)';
  drawSilhouette(7);
  ctx.globalCompositeOperation = 'source-over';

  return PIXI.Texture.from(cv);
}

function ensureWall(state) {
  if (!state.targetAngles) return;
  const key = state.poseName;
  if (wallForPose === key && wallSprite) return;
  if (wallSprite) { wallSprite.destroy(true); }
  wallSprite = new PIXI.Sprite(buildWallTexture(state.targetAngles));
  wallSprite.anchor.set(0.5, HOLE_CY / WALL_H);
  wallSprite.position.set(W / 2, HOLE_CY);
  wallForPose = key;
}

/* ------------------------------------------------------------- the avatar */

function drawAvatar(g, pose, face, segOk, anchor = ANCHOR, live = null) {
  g.clear();
  const j = skeleton(anchor, pose);
  const t = SM.t;
  const line = (a, b, w, color) => {
    g.lineStyle({ width: w, color, cap: PIXI.LINE_CAP.ROUND });
    g.moveTo(...j[a]).lineTo(...j[b]);
  };

  // contact shadow grounds him on the floor
  const feetY = Math.max(j.lAn[1], j.rAn[1]) + t * 0.5;
  g.beginFill(0x000000, 0.32)
    .drawEllipse((j.lAn[0] + j.rAn[0]) / 2, feetY + 6, 86, 13).endFill();

  // outline pass (whole body first, so overlaps merge into one silhouette)
  for (const [a, b] of LEG_SEGS) line(a, b, t + 6, COL.outline);
  line('sc', 'hipc', t * 1.7 + 6, COL.outline);   // capsule torso
  line('lHip', 'rHip', t + 6, COL.outline);
  line('lSh', 'rSh', t + 6, COL.outline);
  for (const [, [a, b]] of Object.entries(ARM_SEGS)) line(a, b, t + 6, COL.outline);

  // matched-arm glow halo under the fill
  for (const [key, [a, b]] of Object.entries(ARM_SEGS)) {
    if (segOk && segOk[key]) line(a, b, t + 15, 0x51e87e);
  }

  // fill pass
  for (const [a, b] of LEG_SEGS) line(a, b, t, COL.fill);
  line('sc', 'hipc', t * 1.7, COL.fill);
  line('lHip', 'rHip', t, COL.fill);   // hip bar: joins the leg roots cleanly
  line('lSh', 'rSh', t, COL.fill);
  for (const [key, [a, b]] of Object.entries(ARM_SEGS)) {
    line(a, b, t, segOk && segOk[key] ? COL.glow : COL.fill);
  }
  g.lineStyle(0);

  // hands: draw the player's real hand skeleton, scaled to the avatar.
  // Both live in the same mirrored screen space, so no rotation is needed.
  const HAND_BONES = [
    [0, 1], [1, 2], [2, 3], [3, 4],          // thumb
    [0, 5], [5, 6], [6, 7], [7, 8],          // index
    [5, 9], [9, 10], [10, 11], [11, 12],     // middle
    [9, 13], [13, 14], [14, 15], [15, 16],   // ring
    [13, 17], [17, 18], [18, 19], [19, 20],  // pinky
    [0, 17],                                  // palm edge
  ];
  for (const [side, wr] of [['l', 'lWr'], ['r', 'rWr']]) {
    const shape = live && live.shapes ? live.shapes[side] : null;
    if (!shape) {
      g.beginFill(COL.extremity).drawCircle(...j[wr], t * 0.62).endFill();
      continue;
    }
    const s = t * 1.05;   // wrist->middle-knuckle distance on the avatar
    const px = (i) => [j[wr][0] + shape[i][0] * s, j[wr][1] + shape[i][1] * s];
    g.lineStyle({ width: 5, color: COL.extremity, cap: PIXI.LINE_CAP.ROUND,
      join: PIXI.LINE_JOIN.ROUND });
    for (const [a, b] of HAND_BONES) {
      g.moveTo(...px(a)).lineTo(...px(b));
    }
    g.lineStyle(0);
    g.beginFill(COL.extremity).drawCircle(...j[wr], t * 0.34).endFill();
  }
  for (const [an, s] of [['lAn', -1], ['rAn', 1]]) {
    g.beginFill(COL.extremity)
      .drawEllipse(j[an][0] + s * t * 0.45, j[an][1] + t * 0.2, t * 0.95, t * 0.5).endFill();
  }

  // head: clean circle, eyes only (no mouth — mouths read goofy at this scale)
  const r = SM.headR;
  g.lineStyle(4, COL.outline).beginFill(COL.fill)
    .drawCircle(...j.head, r).endFill().lineStyle(0);
  // specular highlight ties him into the stage lighting
  g.beginFill(0xffffff, 0.35)
    .drawEllipse(j.head[0] - r * 0.32, j.head[1] - r * 0.42, r * 0.3, r * 0.18).endFill();
  const ex = j.head[0] + pose.head_dx * r * 0.3;
  const ey = j.head[1] - r / 8;
  if (face === 'hit') {
    g.lineStyle(3.2, 0x23232b);
    for (const s of [-1, 1]) {
      const cx = ex + s * r / 3, sz = r / 6;
      g.moveTo(cx - sz, ey - sz).lineTo(cx + sz, ey + sz);
      g.moveTo(cx - sz, ey + sz).lineTo(cx + sz, ey - sz);
    }
    g.lineStyle(0);
  } else if (face === 'win') {
    g.lineStyle(3.2, 0x23232b);
    for (const s of [-1, 1]) {
      const cx = ex + s * r / 3, cy = ey + r / 10, er = r / 6;
      const a0 = 1.15 * Math.PI;
      g.moveTo(cx + er * Math.cos(a0), cy + er * Math.sin(a0));   // no stray connector
      g.arc(cx, cy, er, a0, 1.85 * Math.PI);
    }
    g.lineStyle(0);
  } else {
    // eyes mirror the player's: dot when open, lid line when closed
    const blinks = [live && live.face ? live.face.blinkL : 0,
                    live && live.face ? live.face.blinkR : 0];
    for (const [i, s] of [[0, -1], [1, 1]]) {
      const cx = ex + s * r / 3;
      if (blinks[i] > 0.5) {
        g.lineStyle(3, 0x23232b);
        g.moveTo(cx - r / 7, ey).lineTo(cx + r / 7, ey);
        g.lineStyle(0);
      } else {
        g.beginFill(0x23232b).drawCircle(cx, ey, Math.max(2.2, r / 9)).endFill();
      }
    }
    // mouth: draw the player's real lip contour when available
    const face = live && live.face ? live.face : null;
    const my = j.head[1] + r * 0.38;
    if (face && face.mouth && face.mouth.length > 4) {
      const ms = r * 2.3;   // lip ring is normalized by face width
      const px = face.mouth.map(pt => [ex + pt[0] * ms, my + pt[1] * ms]);
      const openMouth = (face.open || 0) > 0.18;
      g.lineStyle({ width: 2.5, color: 0x23232b, join: PIXI.LINE_JOIN.ROUND });
      if (openMouth) g.beginFill(0x361a1a);
      g.moveTo(...px[0]);
      for (let i = 1; i < px.length; i++) g.lineTo(...px[i]);
      g.closePath();
      if (openMouth) g.endFill();
      g.lineStyle(0);
    } else {
      const smile = face ? face.smile : 0;
      const open = face ? face.open : 0;
      if (open > 0.25) {
        g.lineStyle(3, 0x23232b).beginFill(0x361a1a)
          .drawEllipse(ex, my, r * 0.2 + r * 0.1 * open, r * 0.32 * open)
          .endFill().lineStyle(0);
      } else if (smile > 0.12) {
        g.lineStyle(3.2, 0x23232b);
        const mr = r * (0.2 + 0.26 * smile);
        const a0 = 0.2 * Math.PI, a1 = 0.8 * Math.PI;
        g.moveTo(ex + mr * Math.cos(a0), my - r * 0.14 + mr * Math.sin(a0));
        g.arc(ex, my - r * 0.14, mr, a0, a1);
        g.lineStyle(0);
      } else {
        g.lineStyle(3, 0x23232b);
        g.moveTo(ex - r * 0.16, my).lineTo(ex + r * 0.16, my);
        g.lineStyle(0);
      }
    }
  }
}

/* --------------------------------------------------------------------- ui */

function setText(obj, str) {
  if (obj.text !== str) obj.text = str;
}

const FONT = '"Luckiest Guy", "Arial Black", sans-serif';
const FONT_UI = '"Rubik", "Helvetica Neue", sans-serif';
const mkText = (size, fill, stroke = 0x000000, strokeW = 5, family = FONT) =>
  new PIXI.Text('', new PIXI.TextStyle({
    fontFamily: family, fontSize: size, fill, stroke, strokeThickness: strokeW,
    align: 'center', fontWeight: family === FONT_UI ? '600' : 'normal',
  }));

const ui = {
  poseName: mkText(32, 0xffffff), timer: mkText(22, 0xffffff, 0x000000, 4, FONT_UI),
  score: mkText(18, 0xffffff, 0x000000, 4, FONT_UI),
  score2: mkText(18, 0xffd98c, 0x000000, 4, FONT_UI),
  level: mkText(15, 0x74c7d4, 0x000000, 4, FONT_UI),
  big: mkText(84, 0xffffff, 0x000000, 9),
  sub: mkText(16, 0xdddddd, 0x000000, 4, FONT_UI),
  meterG: new PIXI.Graphics(), heartsG: new PIXI.Graphics(),
  chipG: new PIXI.Graphics(),
  over: new PIXI.Container(),
};
ui.poseName.anchor.set(0.5, 0); ui.poseName.position.set(W / 2, 8);
ui.timer.anchor.set(0.5, 0); ui.timer.position.set(W / 2, 50);
ui.score.position.set(16, 58); ui.score2.position.set(16, 86);
ui.level.position.set(16, 118);
ui.big.anchor.set(0.5); ui.big.position.set(W / 2, H / 2 - 50);
ui.sub.anchor.set(0.5); ui.sub.position.set(W / 2, H - 46);
uiLayer.addChild(ui.meterG, ui.heartsG, ui.chipG, ui.poseName, ui.timer,
  ui.score, ui.score2, ui.level, ui.sub, ui.big, ui.over);

// menu texts
const menuTitle = mkText(58, 0xf0ede4, 0x000000, 8);
const menuSub = mkText(20, 0xdddddd, 0x000000, 4, FONT_UI);
const menuHigh = mkText(16, 0x9aa5b0, 0x000000, 3, FONT_UI);
menuTitle.anchor.set(0.5); menuTitle.position.set(W / 2, 96);
menuSub.anchor.set(0.5); menuSub.position.set(W / 2, H - 74);
menuHigh.anchor.set(0.5); menuHigh.position.set(W / 2, H - 40);
uiLayer.addChild(menuTitle, menuSub, menuHigh);

// lock indicator while holding a matched pose
const lockText = mkText(17, 0x7ce89a, 0x000000, 4, FONT_UI);
lockText.anchor.set(0.5); lockText.position.set(W / 2, 236);
uiLayer.addChild(lockText);

// face/hand requirement indicators
const reqText = mkText(16, 0xffffff, 0x000000, 4, FONT_UI);
reqText.anchor.set(0.5); reqText.position.set(W / 2, 106);
uiLayer.addChild(reqText);

// game-over panel
const overBg = new PIXI.Graphics();
const overTitle = mkText(64, 0xff5555, 0x000000, 8);
const overScore = mkText(36, 0xffffff);
const overHigh = mkText(24, 0xcccccc);
const overMode = mkText(17, 0xd8b06a, 0x000000, 3, FONT_UI);
const overHint = mkText(16, 0xbbbbbb, 0x000000, 3, FONT_UI);
overTitle.anchor.set(0.5); overScore.anchor.set(0.5);
overHigh.anchor.set(0.5); overMode.anchor.set(0.5); overHint.anchor.set(0.5);
overTitle.position.set(W / 2, H / 2 - 90); overScore.position.set(W / 2, H / 2 - 10);
overHigh.position.set(W / 2, H / 2 + 40); overMode.position.set(W / 2, H / 2 + 76);
overHint.position.set(W / 2, H / 2 + 116);
ui.over.addChild(overBg, overTitle, overScore, overHigh, overMode, overHint);

let lastHeartsKey = null;
function drawHearts(lives) {
  if (lives === lastHeartsKey) return;
  lastHeartsKey = lives;
  const g = ui.heartsG;
  g.clear();
  for (let i = 0; i < 3; i++) {
    const c = [26 + i * 44, 30], s = 13, filled = i < lives;
    g.beginFill(filled ? 0xeb4646 : 0x3c2a30, filled ? 1 : 0.9);
    g.drawCircle(c[0] - s / 2, c[1] - s / 4, s / 2 + 1);
    g.drawCircle(c[0] + s / 2, c[1] - s / 4, s / 2 + 1);
    g.drawPolygon([c[0] - s, c[1] - s / 4 + 2, c[0] + s, c[1] - s / 4 + 2, c[0], c[1] + s]);
    g.endFill();
  }
}

function drawMeter(match, threshold = PASS_THRESHOLD) {
  const g = ui.meterG;
  g.clear();
  if (match == null) return;
  const x0 = 200, x1 = W - 200, y = H - 30, h = 10;
  const w = (x1 - x0) * Math.max(0, Math.min(1, match));
  const fit = match >= threshold;
  g.beginFill(0xffffff, 0.12).drawRoundedRect(x0, y, x1 - x0, h, 5).endFill();
  g.beginFill(fit ? 0x52dc78 : 0xe8a13c, 0.95).drawRoundedRect(x0, y, w, h, 5).endFill();
  const px = x0 + (x1 - x0) * threshold;
  g.lineStyle(2, 0xffffff, 0.9).moveTo(px, y - 5).lineTo(px, y + h + 5).lineStyle(0);
}

let lastChipKey = null;
function drawChip(angles) {
  const key = angles ? JSON.stringify(angles) : null;
  if (key === lastChipKey) return;
  lastChipKey = key;
  const g = ui.chipG;
  g.clear();
  if (!angles) return;
  const cx = 72, top = 152;   // left HUD column, clear of the avatar
  g.beginFill(0x000000, 0.45).drawRoundedRect(cx - 56, top, 112, 118, 10).endFill();
  g.lineStyle(1.5, 0xffffff, 0.35).drawRoundedRect(cx - 56, top, 112, 118, 10).lineStyle(0);
  const mini = { headR: 7.9, torso: 21.6, shW: 21.6, hipW: 12.2, upper: 12.2, fore: 11.5, thigh: 15.1, shin: 14.4, t: 5 };
  const j = skeleton({ x: cx, y: top + 34 }, poseFromAngles(angles), mini);
  g.lineStyle({ width: 3, color: 0xffffff, cap: PIXI.LINE_CAP.ROUND, alpha: 0.95 });
  for (const [a, b] of [...Object.values(ARM_SEGS), ...LEG_SEGS,
    [['lSh'], ['rSh']].flat(), [['sc'], ['hipc']].flat(), [['lHip'], ['rHip']].flat()]) {
    g.moveTo(...j[a]).lineTo(...j[b]);
  }
  g.lineStyle(3, 0xffffff, 0.95).drawCircle(j.head[0], j.head[1], mini.headR * 0.9);
  g.lineStyle(0);
}

/* --------------------------------------------------------- fx: shake etc. */

let shake = 0;
let flashA = 0;
let pulse = 0;
const parts = [];
const popups = [];

function confetti() {
  const colors = [0x66e07a, 0xffd35c, 0xff7ab8, 0x7ab8ff, 0xf0ede4];
  for (let i = 0; i < 60; i++) {
    const a = Math.random() * Math.PI - Math.PI;   // upward fan
    const sp = 250 + Math.random() * 520;
    parts.push({
      x: W / 2 + (Math.random() - 0.5) * 160, y: H * 0.45,
      vx: Math.cos(a) * sp * 0.6, vy: -Math.abs(Math.sin(a)) * sp,
      s: 4 + Math.random() * 7, life: 0.8 + Math.random() * 0.8,
      c: colors[(Math.random() * colors.length) | 0],
    });
  }
}

function burst(x, y) {
  for (let i = 0; i < 34; i++) {
    const a = Math.random() * Math.PI * 2, sp = 150 + Math.random() * 500;
    parts.push({
      x, y, vx: Math.cos(a) * sp, vy: -Math.abs(Math.sin(a)) * sp,
      s: 5 + Math.random() * 9, life: 0.5 + Math.random() * 0.6,
      c: lerpColor(0x3f89b8, 0x1e5f8a, Math.random()),
    });
  }
}

function popup(text, x, y, color) {
  const t = mkText(26, color);
  t.anchor.set(0.5); t.position.set(x, y);
  uiLayer.addChild(t);
  popups.push({ t, life: 1.1, y0: y });
  t.text = text;
}

/* ------------------------------------------------------------------ sound */

const AC = window.AudioContext || window.webkitAudioContext;
const audio = AC ? new AC() : null;
const master = audio ? audio.createGain() : null;
if (master) { master.gain.value = 0.75; master.connect(audio.destination); }
function unlockAudio() {
  if (audio && audio.state !== 'running') audio.resume();
}
for (const ev of ['click', 'keydown', 'pointerdown', 'touchstart']) {
  document.addEventListener(ev, unlockAudio);
}

function beep(freq, dur = 0.1, type = 'square', gain = 0.08, when = 0) {
  if (!audio || audio.state !== 'running' || !audioOn) return;
  const t0 = audio.currentTime + when;
  const o = audio.createOscillator(), g = audio.createGain();
  o.type = type; o.frequency.value = freq;
  g.gain.setValueAtTime(0.0001, t0);
  g.gain.linearRampToValueAtTime(gain, t0 + 0.008);   // soft attack, no click
  g.gain.exponentialRampToValueAtTime(0.0001, t0 + dur);
  o.connect(g).connect(master);
  o.start(t0); o.stop(t0 + dur + 0.03);
}

function noiseBurst(dur = 0.25, gain = 0.15, when = 0) {
  if (!audio || audio.state !== 'running' || !audioOn) return;
  const n = Math.floor(audio.sampleRate * dur);
  const buf = audio.createBuffer(1, n, audio.sampleRate);
  const d = buf.getChannelData(0);
  for (let i = 0; i < n; i++) d[i] = (Math.random() * 2 - 1) * (1 - i / n);
  const src = audio.createBufferSource(), g = audio.createGain();
  src.buffer = buf; g.gain.value = gain;
  src.connect(g).connect(master);
  src.start(audio.currentTime + when);
}

let audioOn = true;

// Lookahead-scheduled synth loop; tempo rises as the wall closes in.
const music = {
  next: 0, beat: 0, bpm: 96,
  tick() {
    if (!audio || audio.state !== 'running' || !audioOn) return;
    if (this.next < audio.currentTime - 0.5) this.next = audio.currentTime + 0.02;
    while (this.next < audio.currentTime + 0.15) {
      const when = this.next - audio.currentTime;
      const b = this.beat % 8;
      if (b % 2 === 0) beep(98, 0.13, 'sine', 0.09, when);                 // kick (audible on laptops)
      if (b % 2 === 1) noiseBurst(0.03, 0.018, when);                      // soft brushed hat
      const bass = [0, 0, 3, 5, 0, 0, 7, 5][b];
      beep(110 * Math.pow(2, bass / 12), 0.16, 'triangle', 0.045, when);   // warm bass
      this.next += 60 / this.bpm / 2;
      this.beat++;
    }
  },
};

// Speech-synthesis announcer: off — the macOS TTS voice sounded bad.
let voiceOn = false;
function say(text) {
  if (!voiceOn || !audioOn || !window.speechSynthesis) return;
  const u = new SpeechSynthesisUtterance(text.toLowerCase());
  u.rate = 1.15; u.pitch = 1.05; u.volume = 0.9;
  speechSynthesis.cancel();
  speechSynthesis.speak(u);
}

const SFX = {
  fakeout: () => { beep(880, 0.09, 'square', 0.09); beep(587, 0.16, 'square', 0.09, 0.1); },
  handoff: () => { beep(523, 0.1, 'triangle', 0.08); beep(784, 0.14, 'triangle', 0.08, 0.11); },
  tick: () => beep(660, 0.07, 'square', 0.06),
  go: () => { beep(880, 0.12); beep(1320, 0.15, 'square', 0.07, 0.1); },
  pass: () => { beep(523, 0.09); beep(659, 0.09, 'square', 0.08, 0.08); beep(784, 0.16, 'square', 0.08, 0.16); },
  perfect: () => { [523, 659, 784, 1047].forEach((f, i) => beep(f, 0.12, 'square', 0.09, i * 0.07)); },
  crash: () => { noiseBurst(0.3, 0.18); beep(90, 0.35, 'sawtooth', 0.16); },
  gameover: () => { [392, 330, 262, 196].forEach((f, i) => beep(f, 0.22, 'triangle', 0.1, i * 0.18)); },
  cheer: () => {   // crowd swell: layered noise + a few whistles
    noiseBurst(0.7, 0.10); noiseBurst(0.9, 0.07, 0.15);
    [1568, 1760, 2093].forEach((f, i) => beep(f, 0.16, 'sine', 0.03, 0.1 + i * 0.12));
  },
};

/* ------------------------------------------------------------- state + io */

let S = null;
let ws = null;
let wsUp = false;
const pip = document.getElementById('pip');

function connect() {
  ws = new WebSocket(`ws://${location.host}/ws`);
  ws.onopen = () => { wsUp = true; };
  ws.onclose = () => { wsUp = false; setTimeout(connect, 1500); };
  ws.onerror = () => { try { ws.close(); } catch (e) { /* already closed */ } };
  ws.onmessage = (m) => {
    S = JSON.parse(m.data);
    for (const ev of S.events || []) {
      if (SFX[ev]) SFX[ev]();
      if (ev === 'crash') { shake = 1; flashA = 0.55; burst(W / 2, H * 0.5); }
      if (ev === 'pass' || ev === 'perfect') {
        pulse = 1;
        if (ev === 'perfect' || S.mult >= 3) SFX.cheer();
        const gain = S.lastGain || { points: 100 * S.mult, perfect: ev === 'perfect', bonus: 0 };
        popup(`+${gain.points}${gain.perfect ? '  PERFECT!' : ''}`, W / 2, 150, 0x66ff99);
        if (gain.bonus >= 10) popup(`hold bonus +${gain.bonus}`, W / 2, 186, 0xa8e6ff);
        if (ev === 'perfect') { confetti(); say('perfect!'); }
      }
      if (ev === 'gameover') say('game over');
      if (ev === 'fakeout') {
        popup('FAKE-OUT!', W / 2, 250, 0xffc85c);
        shake = 0.35;
      }
    }
    if (S.pip) pip.src = 'data:image/jpeg;base64,' + S.pip;
  };
}
connect();

function sendKey(key) {
  if (wsUp) ws.send(JSON.stringify({ key }));
}

document.addEventListener('keydown', (e) => {
  if (e.code === 'Space') { sendKey('restart'); e.preventDefault(); }
  if (e.key === 's') sendKey('skip');
  if (e.key === 'd') sendKey('daily');
  if (e.key === 'Escape') sendKey('menu');
  if (e.key === 't') sendKey('toggle2p');
  if (e.key === 'l') sendKey('togglelegs');
  if (e.key === 'm') { audioOn = !audioOn; if (!audioOn) speechSynthesis?.cancel(); }
  if (e.key === 'c' && S && S.state === 'GAME_OVER') downloadScoreCard();
});

function downloadScoreCard() {
  const cv = document.createElement('canvas');
  cv.width = 840; cv.height = 440;
  const ctx = cv.getContext('2d');
  const bg = ctx.createLinearGradient(0, 0, 0, 440);
  bg.addColorStop(0, '#191927'); bg.addColorStop(1, '#0d0d14');
  ctx.fillStyle = bg; ctx.fillRect(0, 0, 840, 440);
  ctx.strokeStyle = '#b8623c'; ctx.lineWidth = 6; ctx.strokeRect(10, 10, 820, 420);
  ctx.textAlign = 'center';
  ctx.fillStyle = '#f0ede4';
  ctx.font = '46px "Luckiest Guy", "Arial Black"';
  ctx.fillText('HOLE IN THE WALL', 420, 92);
  ctx.font = '80px "Luckiest Guy", "Arial Black"';
  ctx.fillStyle = '#7ce89a';
  ctx.fillText(String(S.score), 420, 214);
  ctx.fillStyle = '#aaa';
  ctx.font = '600 22px Rubik, sans-serif';
  const mode = S.mode === 'daily' ? `DAILY CHALLENGE ${S.dailyDate}` : 'ENDLESS MODE';
  ctx.fillText(mode, 420, 268);
  ctx.fillText(`high score ${S.highScore}`, 420, 306);
  ctx.fillStyle = '#666';
  ctx.font = '600 16px Rubik, sans-serif';
  ctx.fillText('strike the pose - fit through the wall', 420, 396);
  const a = document.createElement('a');
  a.download = `hole-in-the-wall-${S.score}.png`;
  a.href = cv.toDataURL('image/png');
  a.click();
}

/* ------------------------------------------------------------ render loop */

buildBackground();
vignetteLayer.addChild(buildVignette());
drawHearts(3);

let smoothScale = 0.22;
let lastBig = '';
let bigPop = 1;
let lastPoseName = '';
let axSmooth = 0;
let holeDxSmooth = 0;
let prevState = '';

app.ticker.add((dt) => {
  const dts = dt / 60;
  if (!S) return;

  // shake decay + pass zoom pulse share the centered root transform
  if (shake > 0.01) {
    root.position.set(W / 2 + (Math.random() - 0.5) * 26 * shake,
      H / 2 + (Math.random() - 0.5) * 26 * shake);
    shake *= Math.pow(0.02, dts);
  } else root.position.set(W / 2, H / 2);
  if (pulse > 0.01) {
    root.scale.set(1 + 0.05 * pulse);
    pulse *= Math.pow(0.01, dts);
  } else root.scale.set(1);

  // music tempo rises as the wall closes in
  music.bpm = 96 + (S.state === 'WALL' ? 74 * S.progress : 0);
  music.tick();

  // announce each new wall's pose
  if (S.state === 'WALL' && S.poseName !== lastPoseName) {
    lastPoseName = S.poseName;
    say(S.poseName);
  }
  if (S.state === 'MENU') lastPoseName = '';

  // avatar breathes and steps sideways with the player
  const breath = Math.sin(performance.now() / 640) * 2.2;
  axSmooth += ((S.ax || 0) - axSmooth) * Math.min(1, dts * 12);
  holeDxSmooth += ((S.holeDx || 0) - holeDxSmooth) * Math.min(1, dts * 14);
  const face = S.state === 'RESULT' ? (S.outcome === 'pass' ? 'win' : 'hit') : 'idle';
  drawAvatar(avatarG, S.pose, face, S.state === 'WALL' ? S.segOk : null,
    { x: ANCHOR.x + axSmooth, y: ANCHOR.y + breath },
    { face: S.faceLive, shapes: S.handShapes });

  // wall (sprite shifts horizontally for offset / sliding holes)
  wallBehind.removeChildren(); wallFront.removeChildren();
  if (S.state === 'WALL') {
    if (prevState !== 'WALL') {
      smoothScale = 0.22;                    // new wall starts at the horizon
      if (wallSprite) wallSprite.alpha = 1;  // undo the fly-through fade
    }
    ensureWall(S);
    const target = 0.22 + 0.78 * Math.pow(S.progress, 2.2);
    smoothScale += (target - smoothScale) * Math.min(1, dts * 14);
    wallSprite.scale.set(smoothScale);
    wallSprite.position.x = W / 2 + holeDxSmooth * smoothScale;
    const fit = (S.match ?? 0) >= (S.passThreshold ?? PASS_THRESHOLD);
    wallSprite.tint = fit ? 0xccffcc : (S.tight ? 0xffe9b8 : 0xffffff);
    (smoothScale < 0.9 ? wallBehind : wallFront).addChild(wallSprite);
  } else if (S.state === 'RESULT' && S.outcome === 'pass' && wallSprite) {
    // brief impact hold (slow-mo beat), then accelerate through the hole
    const zt = Math.max(0, (S.resultT - 0.22) / 0.78);
    const z = 1 + 0.05 * Math.min(S.resultT / 0.22, 1) + 2.6 * Math.pow(zt, 1.6);
    wallSprite.scale.set(z);
    wallSprite.alpha = Math.max(0, 1 - zt * 0.85);
    wallFront.addChild(wallSprite);
  } else {
    smoothScale = 0.22;
    if (wallSprite) wallSprite.alpha = 1;
  }

  // particles
  fxG.clear();
  for (let i = parts.length - 1; i >= 0; i--) {
    const p = parts[i];
    p.life -= dts;
    if (p.life <= 0) { parts.splice(i, 1); continue; }
    p.vy += 1500 * dts; p.x += p.vx * dts; p.y += p.vy * dts;
    fxG.beginFill(p.c).drawRect(p.x - p.s / 2, p.y - p.s / 2, p.s, p.s).endFill();
  }
  // red flash
  if (flashA > 0.01) {
    fxG.beginFill(0xdd2222, flashA).drawRect(0, 0, W, H).endFill();
    flashA *= Math.pow(0.01, dts);
  }
  // popups
  for (let i = popups.length - 1; i >= 0; i--) {
    const p = popups[i];
    p.life -= dts;
    if (p.life <= 0) { p.t.destroy(); popups.splice(i, 1); continue; }
    p.t.position.y -= 50 * dts;
    p.t.alpha = Math.min(1, p.life * 2);
  }

  // HUD
  drawHearts(S.state === 'MENU' ? 0 : S.lives);
  ui.heartsG.visible = S.state !== 'MENU';
  if (S.twoP && S.players && S.players.length === 2) {
    const arrow = (i) => (S.activeP === i ? '> ' : '   ');
    setText(ui.score, S.state === 'MENU' ? '' :
      `${arrow(0)}P1  ${S.players[0].score}   ${'\u2665'.repeat(S.players[0].lives)}`);
    setText(ui.score2, S.state === 'MENU' ? '' :
      `${arrow(1)}P2  ${S.players[1].score}   ${'\u2665'.repeat(S.players[1].lives)}`);
    ui.heartsG.visible = false;
  } else {
    setText(ui.score, S.state === 'MENU' ? '' : `SCORE ${S.score}`);
    setText(ui.score2, '');
  }
  setText(ui.level, S.state === 'MENU' ? '' : `LVL ${S.level}   x${S.mult}`);
  ui.over.visible = false;
  setText(ui.big, ''); setText(ui.sub, '');
  setText(ui.poseName, ''); setText(ui.timer, '');
  setText(menuTitle, ''); setText(menuSub, ''); setText(menuHigh, '');
  setText(lockText, ''); setText(reqText, '');
  drawMeter(null); drawChip(null);

  if (S.state === 'MENU') {
    setText(menuTitle, 'HOLE IN THE WALL');
    menuSub.text = 'SPACE - start    D - daily    ' +
      `T - 2 player: ${S.twoP ? 'ON' : 'OFF'}    ` +
      `L - leg mode: ${S.legMode ? 'ON' : 'OFF'}    M - mute`;
    menuHigh.text = `high score ${S.highScore}` +
      (S.dailyBest ? `    daily best ${S.dailyBest}` : '');
  } else if (S.state === 'HANDOFF') {
    setText(ui.big, `PLAYER ${S.activeP + 1} - GET READY!`);
    ui.sub.text = 'swap places!';
  }

  if (S.state === 'WALL') {
    setText(ui.poseName, S.poseName + (S.tight ? '  -  TIGHT x2!' : ''));
    const reqs = [];
    if (S.faceReq) {
      reqs.push(`${S.faceReq === 'smile' ? 'SMILE' : 'WOW MOUTH'} ${S.segOk && S.segOk.face ? 'OK!' : '...'}`);
    }
    if (S.handsReq) {
      reqs.push(`${S.handsReq.toUpperCase()} HANDS ${S.segOk && S.segOk.hands ? 'OK!' : '...'}`);
    }
    setText(reqText, reqs.join('    '));
    reqText.style.fill = (S.segOk && ((S.faceReq && !S.segOk.face) ||
      (S.handsReq && !S.segOk.hands))) ? 0xffc85c : 0x7ce89a;
    setText(ui.timer, `${S.timeLeft.toFixed(1)}s`);
    ui.timer.style.fill = S.timeLeft > 2 ? 0xffffff : 0xff6666;
    drawMeter(S.match, S.passThreshold ?? PASS_THRESHOLD);
    drawChip(S.targetAngles);
    if (!S.tracked) ui.sub.text = 'Step back so the camera sees both your arms';
    else if (S.match == null) ui.sub.text = S.legMode
      ? 'Step back - leg mode needs your full body in view'
      : 'Move back a bit - both arms need to be in view';
    if ((S.match ?? 0) >= (S.passThreshold ?? PASS_THRESHOLD) && S.holdT > 0.15) {
      lockText.text = `LOCKED  +${Math.floor(Math.min(S.holdT, 2) * 30)}`;
    }
    const stepErr = (S.holeDx || 0) - (S.ax || 0);
    if (S.segOk && S.segOk.pos === false && Math.abs(stepErr) > 55) {
      lockText.text = stepErr > 0 ? 'STEP RIGHT  -->' : '<--  STEP LEFT';
      lockText.style.fill = 0xffc85c;
    } else {
      lockText.style.fill = 0x7ce89a;
    }
  } else if (S.state === 'COUNTDOWN') {
    setText(ui.big, S.countdown > 0 ? String(S.countdown) : 'GO!');
    setText(ui.sub, 'Mirror the stickman with your body');
  } else if (S.state === 'RESULT') {
    setText(ui.big, S.outcome === 'pass' ? 'THROUGH!' : 'CRASHED!');
    ui.big.style.fill = S.outcome === 'pass' ? 0x66ff99 : 0xff5555;
  } else if (S.state === 'GAME_OVER') {
    ui.over.visible = true;
    overBg.clear().beginFill(0x000000, 0.72).drawRect(0, 0, W, H).endFill();
    if (S.twoP && S.players && S.players.length === 2) {
      overTitle.text = S.winner === -1 ? 'TIE GAME!' : `PLAYER ${S.winner + 1} WINS!`;
      overTitle.style.fill = 0xffd98c;
      overScore.text = `P1  ${S.players[0].score}    -    P2  ${S.players[1].score}`;
      overHigh.text = '';
      overMode.text = '2 PLAYER';
    } else {
      overTitle.text = 'GAME OVER';
      overTitle.style.fill = 0xff5555;
      overScore.text = `SCORE  ${S.score}`;
      overHigh.text = S.newRecord ? `NEW HIGH SCORE!  ${S.highScore}` : `HIGH SCORE  ${S.highScore}`;
      overHigh.style.fill = S.newRecord ? 0x66ff99 : 0xcccccc;
      overMode.text = S.mode === 'daily'
        ? `DAILY ${S.dailyDate}   best today ${S.dailyBest}` : 'ENDLESS';
    }
    overHint.text = 'SPACE play again  |  D daily  |  C save score card';
  }

  // pop-in on every big-text change (countdown digits, GO!, THROUGH!, CRASHED!)
  if (ui.big.text !== lastBig) {
    lastBig = ui.big.text;
    if (lastBig) bigPop = 0;
  }
  bigPop = Math.min(1, bigPop + dts * 3.5);
  const over = 1 - bigPop;
  ui.big.scale.set(1 + 0.7 * over * over);
  prevState = S.state;
});
