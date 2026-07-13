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
});
document.getElementById('wrap').prepend(app.view);

const root = new PIXI.Container();      // shakeable
app.stage.addChild(root);

const bgLayer = new PIXI.Container();
const wallBehind = new PIXI.Container();
const avatarG = new PIXI.Graphics();
const wallFront = new PIXI.Container();
const fxG = new PIXI.Graphics();
const uiLayer = new PIXI.Container();
root.addChild(bgLayer, wallBehind, avatarG, wallFront, fxG, uiLayer);

try {
  if (app.renderer.type === PIXI.RENDERER_TYPE.WEBGL) {
    root.filters = [new PIXI.filters.AdvancedBloomFilter({
      threshold: 0.55, bloomScale: 0.9, brightness: 1.0, blur: 4,
    })];
  }
} catch (e) { /* pixi-filters CDN missing — fine without bloom */ }

/* -------------------------------------------------------------- background */

function buildBackground() {
  const g = new PIXI.Graphics();
  const horizon = H * 0.52;
  // sky gradient in bands
  for (let i = 0; i < 40; i++) {
    const a = i / 39;
    const c = lerpColor(COL.sky0, COL.sky1, a);
    g.beginFill(c).drawRect(0, (horizon / 40) * i, W, horizon / 40 + 1).endFill();
  }
  // stage glow behind the avatar — silhouette contrast, TV-studio feel
  for (let i = 9; i >= 1; i--) {
    g.beginFill(0xd8c9a8, 0.018 * i)
      .drawEllipse(W / 2, ANCHOR.y + 90, 90 + i * 34, 130 + i * 26).endFill();
  }
  // faint horizon line
  for (let i = 3; i >= 1; i--) {
    g.beginFill(COL.gridGlow, 0.05 * i)
      .drawEllipse(W / 2, horizon, W * 0.55, 4 + i * 4).endFill();
  }
  // floor
  for (let i = 0; i < 30; i++) {
    const a = i / 29;
    const c = lerpColor(0x1c2126, 0x0d0f12, a);
    g.beginFill(c).drawRect(0, horizon + ((H - horizon) / 30) * i, W,
      (H - horizon) / 30 + 1).endFill();
  }
  // quiet perspective grid
  g.lineStyle(1.5, COL.grid, 0.5);
  for (let x = -W; x <= 2 * W; x += W / 6) {
    g.moveTo(x, H).lineTo(W / 2, horizon);
  }
  for (let i = 1; i <= 6; i++) {
    const y = horizon + (H - horizon) * Math.pow(i / 6, 1.8);
    g.lineStyle(1.5, COL.grid, 0.25 + 0.3 * (i / 6));
    g.moveTo(0, y).lineTo(W, y);
  }
  // spotlight pool under the avatar
  for (let i = 5; i >= 1; i--) {
    g.beginFill(0xf5e9d0, 0.02 * i).drawEllipse(W / 2, H * 0.9, 150 + i * 16, 24 + i * 4).endFill();
  }
  bgLayer.addChild(g);

  // sparse dust motes
  const stars = new PIXI.Graphics();
  for (let i = 0; i < 34; i++) {
    stars.beginFill(0xffffff, Math.random() * 0.25 + 0.05)
      .drawCircle(Math.random() * W, Math.random() * horizon * 0.9,
        Math.random() * 1.3 + 0.3).endFill();
  }
  bgLayer.addChild(stars);
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

function buildWallTexture(angles) {
  // Offscreen 2D canvas: bricks, then punch the silhouette hole with
  // destination-out; the pre-stroked wider silhouette leaves a rim ring.
  const cv = document.createElement('canvas');
  cv.width = W; cv.height = H;
  const ctx = cv.getContext('2d');

  ctx.fillStyle = '#a85835';
  ctx.fillRect(0, 0, W, H);
  // per-brick tonal variation, then soft mortar lines
  const step = 44;
  for (let y = 0; y < H; y += step) {
    const off = (y / step) % 2 ? step : 0;
    for (let x = off - step * 2; x < W; x += step * 2) {
      const v = Math.random() * 16 - 8;
      ctx.fillStyle = `rgb(${168 + v},${88 + v * 0.6},${53 + v * 0.4})`;
      ctx.fillRect(x + 2, y + 2, step * 2 - 4, step - 4);
    }
  }
  ctx.strokeStyle = 'rgba(90,42,22,0.55)';
  ctx.lineWidth = 3;
  for (let y = 0; y < H; y += step) {
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
    const off = (y / step) % 2 ? step : 0;
    for (let x = off; x < W; x += step * 2) {
      ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x, y + step); ctx.stroke();
    }
  }
  // subtle vignette on the wall
  const vg = ctx.createRadialGradient(W / 2, H / 2, H * 0.3, W / 2, H / 2, H);
  vg.addColorStop(0, 'rgba(0,0,0,0)'); vg.addColorStop(1, 'rgba(20,5,0,0.4)');
  ctx.fillStyle = vg; ctx.fillRect(0, 0, W, H);

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

  // rim ring first (white), then cut the hole
  ctx.strokeStyle = ctx.fillStyle = 'rgba(255,255,255,0.95)';
  drawSilhouette(12);
  ctx.globalCompositeOperation = 'destination-out';
  ctx.strokeStyle = ctx.fillStyle = '#000';
  drawSilhouette(0);
  ctx.globalCompositeOperation = 'source-over';

  return PIXI.Texture.from(cv);
}

// Scale the wall about the hole's center, not the frame center — the hole
// then stays locked on the avatar for the whole approach.
const HOLE_CY = 280;

function ensureWall(state) {
  if (!state.targetAngles) return;
  const key = state.poseName;
  if (wallForPose === key && wallSprite) return;
  if (wallSprite) { wallSprite.destroy(true); }
  wallSprite = new PIXI.Sprite(buildWallTexture(state.targetAngles));
  wallSprite.anchor.set(0.5, HOLE_CY / H);
  wallSprite.position.set(W / 2, HOLE_CY);
  wallForPose = key;
}

/* ------------------------------------------------------------- the avatar */

function drawAvatar(g, pose, face, segOk) {
  g.clear();
  const j = skeleton(ANCHOR, pose);
  const t = SM.t;
  const line = (a, b, w, color) => {
    g.lineStyle({ width: w, color, cap: PIXI.LINE_CAP.ROUND });
    g.moveTo(...j[a]).lineTo(...j[b]);
  };

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

  // hands / feet in the outline tone
  for (const wr of ['lWr', 'rWr']) {
    g.beginFill(COL.extremity).drawCircle(...j[wr], t * 0.62).endFill();
  }
  for (const [an, s] of [['lAn', -1], ['rAn', 1]]) {
    g.beginFill(COL.extremity)
      .drawEllipse(j[an][0] + s * t * 0.45, j[an][1] + t * 0.2, t * 0.95, t * 0.5).endFill();
  }

  // head: clean circle, eyes only (no mouth — mouths read goofy at this scale)
  const r = SM.headR;
  g.lineStyle(4, COL.outline).beginFill(COL.fill)
    .drawCircle(...j.head, r).endFill().lineStyle(0);
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
    g.beginFill(0x23232b);
    for (const s of [-1, 1]) g.drawCircle(ex + s * r / 3, ey, Math.max(2.2, r / 9));
    g.endFill();
  }
}

/* --------------------------------------------------------------------- ui */

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
  level: mkText(15, 0x74c7d4, 0x000000, 4, FONT_UI),
  big: mkText(84, 0xffffff, 0x000000, 9),
  sub: mkText(16, 0xdddddd, 0x000000, 4, FONT_UI),
  meterG: new PIXI.Graphics(), heartsG: new PIXI.Graphics(),
  chipG: new PIXI.Graphics(),
  over: new PIXI.Container(),
};
ui.poseName.anchor.set(0.5, 0); ui.poseName.position.set(W / 2, 8);
ui.timer.anchor.set(0.5, 0); ui.timer.position.set(W / 2, 50);
ui.score.position.set(16, 58); ui.level.position.set(16, 92);
ui.big.anchor.set(0.5); ui.big.position.set(W / 2, H / 2 - 50);
ui.sub.anchor.set(0.5); ui.sub.position.set(W / 2, H - 46);
uiLayer.addChild(ui.meterG, ui.heartsG, ui.chipG, ui.poseName, ui.timer,
  ui.score, ui.level, ui.sub, ui.big, ui.over);

// game-over panel
const overBg = new PIXI.Graphics();
const overTitle = mkText(64, 0xff5555, 0x000000, 8);
const overScore = mkText(36, 0xffffff);
const overHigh = mkText(24, 0xcccccc);
const overHint = mkText(18, 0xbbbbbb);
overTitle.anchor.set(0.5); overScore.anchor.set(0.5);
overHigh.anchor.set(0.5); overHint.anchor.set(0.5);
overTitle.position.set(W / 2, H / 2 - 90); overScore.position.set(W / 2, H / 2 - 10);
overHigh.position.set(W / 2, H / 2 + 40); overHint.position.set(W / 2, H / 2 + 90);
ui.over.addChild(overBg, overTitle, overScore, overHigh, overHint);

function drawHearts(lives) {
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

function drawMeter(match) {
  const g = ui.meterG;
  g.clear();
  if (match == null) return;
  const x0 = 200, x1 = W - 200, y = H - 30, h = 10;
  const w = (x1 - x0) * Math.max(0, Math.min(1, match));
  const fit = match >= PASS_THRESHOLD;
  g.beginFill(0xffffff, 0.12).drawRoundedRect(x0, y, x1 - x0, h, 5).endFill();
  g.beginFill(fit ? 0x52dc78 : 0xe8a13c, 0.95).drawRoundedRect(x0, y, w, h, 5).endFill();
  const px = x0 + (x1 - x0) * PASS_THRESHOLD;
  g.lineStyle(2, 0xffffff, 0.9).moveTo(px, y - 5).lineTo(px, y + h + 5).lineStyle(0);
}

function drawChip(angles) {
  const g = ui.chipG;
  g.clear();
  if (!angles) return;
  const cx = 72, top = 130;   // left HUD column, clear of the avatar
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
const parts = [];
const popups = [];

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
document.addEventListener('click', () => audio && audio.resume(), { once: true });

function beep(freq, dur = 0.1, type = 'square', gain = 0.08, when = 0) {
  if (!audio || audio.state !== 'running') return;
  const t0 = audio.currentTime + when;
  const o = audio.createOscillator(), g = audio.createGain();
  o.type = type; o.frequency.value = freq;
  g.gain.setValueAtTime(gain, t0);
  g.gain.exponentialRampToValueAtTime(0.0001, t0 + dur);
  o.connect(g).connect(audio.destination);
  o.start(t0); o.stop(t0 + dur + 0.02);
}

function noiseBurst(dur = 0.25, gain = 0.15) {
  if (!audio || audio.state !== 'running') return;
  const n = audio.sampleRate * dur;
  const buf = audio.createBuffer(1, n, audio.sampleRate);
  const d = buf.getChannelData(0);
  for (let i = 0; i < n; i++) d[i] = (Math.random() * 2 - 1) * (1 - i / n);
  const src = audio.createBufferSource(), g = audio.createGain();
  src.buffer = buf; g.gain.value = gain;
  src.connect(g).connect(audio.destination);
  src.start();
}

const SFX = {
  tick: () => beep(660, 0.07, 'square', 0.06),
  go: () => { beep(880, 0.12); beep(1320, 0.15, 'square', 0.07, 0.1); },
  pass: () => { beep(523, 0.09); beep(659, 0.09, 'square', 0.08, 0.08); beep(784, 0.16, 'square', 0.08, 0.16); },
  perfect: () => { [523, 659, 784, 1047].forEach((f, i) => beep(f, 0.12, 'square', 0.09, i * 0.07)); },
  crash: () => { noiseBurst(0.3, 0.18); beep(90, 0.35, 'sawtooth', 0.16); },
  gameover: () => { [392, 330, 262, 196].forEach((f, i) => beep(f, 0.22, 'triangle', 0.1, i * 0.18)); },
};

/* ------------------------------------------------------------- state + io */

let S = null;
const pip = document.getElementById('pip');

const ws = new WebSocket(`ws://${location.host}/ws`);
ws.onmessage = (m) => {
  S = JSON.parse(m.data);
  for (const ev of S.events || []) {
    if (SFX[ev]) SFX[ev]();
    if (ev === 'crash') { shake = 1; flashA = 0.55; burst(W / 2, H * 0.5); }
    if (ev === 'pass' || ev === 'perfect') {
      popup(`+${100 * S.mult + (ev === 'perfect' ? 50 : 0)}${ev === 'perfect' ? ' PERFECT!' : ''}`,
        W / 2, 150, 0x66ff99);
    }
  }
  if (S.pip) pip.src = 'data:image/jpeg;base64,' + S.pip;
};

document.addEventListener('keydown', (e) => {
  if (e.code === 'Space') { ws.send(JSON.stringify({ key: 'restart' })); e.preventDefault(); }
  if (e.key === 's') ws.send(JSON.stringify({ key: 'skip' }));
});

/* ------------------------------------------------------------ render loop */

buildBackground();
drawHearts(3);

let smoothScale = 0.22;

app.ticker.add((dt) => {
  const dts = dt / 60;
  if (!S) return;

  // shake decay
  if (shake > 0.01) {
    root.position.set((Math.random() - 0.5) * 26 * shake, (Math.random() - 0.5) * 26 * shake);
    shake *= Math.pow(0.02, dts);
  } else root.position.set(0, 0);

  // avatar
  const face = S.state === 'RESULT' ? (S.outcome === 'pass' ? 'win' : 'hit') : 'idle';
  drawAvatar(avatarG, S.pose, face, S.state === 'WALL' ? S.segOk : null);

  // wall
  wallBehind.removeChildren(); wallFront.removeChildren();
  if (S.state === 'WALL') {
    ensureWall(S);
    const target = 0.22 + 0.78 * Math.pow(S.progress, 2.2);
    smoothScale += (target - smoothScale) * Math.min(1, dts * 14);
    wallSprite.scale.set(smoothScale);
    wallSprite.tint = (S.match ?? 0) >= PASS_THRESHOLD ? 0xccffcc : 0xffffff;
    (smoothScale < 0.9 ? wallBehind : wallFront).addChild(wallSprite);
  } else if (S.state === 'RESULT' && S.outcome === 'pass' && wallSprite) {
    const z = 1 + 2.4 * Math.pow(S.resultT, 1.5);
    wallSprite.scale.set(z);
    wallSprite.alpha = Math.max(0, 1 - S.resultT * 0.8);
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
  drawHearts(S.lives);
  ui.score.text = `SCORE ${S.score}`;
  ui.level.text = `LVL ${S.level}   x${S.mult}`;
  ui.over.visible = false;
  ui.big.text = ''; ui.sub.text = '';
  ui.poseName.text = ''; ui.timer.text = '';
  drawMeter(null); drawChip(null);

  if (S.state === 'WALL') {
    ui.poseName.text = S.poseName;
    ui.timer.text = `${S.timeLeft.toFixed(1)}s`;
    ui.timer.style.fill = S.timeLeft > 2 ? 0xffffff : 0xff6666;
    drawMeter(S.match);
    drawChip(S.targetAngles);
    if (!S.tracked) ui.sub.text = 'Step back so the camera sees both your arms';
  } else if (S.state === 'COUNTDOWN') {
    ui.big.text = S.countdown > 0 ? String(S.countdown) : 'GO!';
    const pulse = 1 + 0.25 * (1 - ((S.countdown ?? 0) % 1));
    ui.big.scale.set(1);
    ui.sub.text = 'Mirror the stickman with your body';
  } else if (S.state === 'RESULT') {
    ui.big.text = S.outcome === 'pass' ? 'THROUGH!' : 'CRASHED!';
    ui.big.style.fill = S.outcome === 'pass' ? 0x66ff99 : 0xff5555;
  } else if (S.state === 'GAME_OVER') {
    ui.over.visible = true;
    overBg.clear().beginFill(0x000000, 0.72).drawRect(0, 0, W, H).endFill();
    overTitle.text = 'GAME OVER';
    overScore.text = `SCORE  ${S.score}`;
    overHigh.text = S.newRecord ? `NEW HIGH SCORE!  ${S.highScore}` : `HIGH SCORE  ${S.highScore}`;
    overHigh.style.fill = S.newRecord ? 0x66ff99 : 0xcccccc;
    overHint.text = 'press SPACE to play again';
  }
});
