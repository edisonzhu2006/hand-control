/* True-3D scene in the designed art style: cel-shaded ivory stickman with
 * dark inverted-hull outlines, camera-facing face billboard (same eyes +
 * volumetric lips as the 2D renderer), hand-skeleton billboards at the
 * wrists, and the wall as an alpha-cut plane approaching in real z.
 *
 * pose3d: meters, shoulder-anchored, x right / y DOWN / z away from camera.
 * Three is y-up, camera looks down -z: we negate y and z.
 */

const T3 = (() => {
  if (typeof THREE === 'undefined') return null;

  const PXPM = 195;          // wall-texture pixels per meter
  const SHOULDER_Y = 1.16;   // avatar shoulder height above the floor (m)
  const IVORY = 0xf0ede4, DARK = 0x23232b;

  let renderer, scene, camera, wallMesh, wallTex;
  let faceSprite, faceCanvas, faceTex;
  const handSprites = {};    // side -> {sprite, canvas, tex}
  const limbs = {};
  const joints = {};
  let group, headMesh, hairMesh;
  let ok = false;

  // outfit shop (H to cycle): shirt / shorts / hair colors
  const LOOKS = [
    { shirt: 0xd73434, shorts: 0x2a3a5c, hair: 0x3a2a1e },   // classic red
    { shirt: 0x2e8f5b, shorts: 0x23232b, hair: 0x111111 },   // forest
    { shirt: 0x8458c9, shorts: 0x3a2a1e, hair: 0xd6b03c },   // wizard blond
    { shirt: 0xf0ede4, shorts: 0xd73434, hair: 0x777777 },   // classic ivory
  ];
  let lookIdx = 0;
  function setLook(i) {
    lookIdx = ((i % LOOKS.length) + LOOKS.length) % LOOKS.length;
    const L = LOOKS[lookIdx];
    for (const key of ['tor', 'shB']) limbs[key].material.color.setHex(L.shirt);
    for (const key of ['luA', 'ruA']) limbs[key].material.color.setHex(L.shirt);
    for (const key of ['hpB', 'ltL', 'rtL']) limbs[key].material.color.setHex(L.shorts);
    if (hairMesh) hairMesh.material.color.setHex(L.hair);
  }

  const LIMB_DEFS = [
    ['luA', 'l_shoulder', 'l_elbow', 0.05], ['lfA', 'l_elbow', 'l_wrist', 0.045],
    ['ruA', 'r_shoulder', 'r_elbow', 0.05], ['rfA', 'r_elbow', 'r_wrist', 0.045],
    ['ltL', 'l_hip', 'l_knee', 0.06], ['lsL', 'l_knee', 'l_ankle', 0.055],
    ['rtL', 'r_hip', 'r_knee', 0.06], ['rsL', 'r_knee', 'r_ankle', 0.055],
    ['shB', 'l_shoulder', 'r_shoulder', 0.055],
    ['hpB', 'l_hip', 'r_hip', 0.055],
    ['tor', 'shc', 'hipc', 0.095],
  ];

  function toonMat(color) {
    // lit material: depth reads through shading gradients as limbs rotate
    return new THREE.MeshStandardMaterial({ color, roughness: 0.55, metalness: 0.05 });
  }

  function outlined(geo, color, hull = 1.22) {
    // designed look: flat fill + dark outline (inverted hull)
    const m = new THREE.Mesh(geo, toonMat(color));
    m.castShadow = true;
    const o = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({
      color: DARK, side: THREE.BackSide }));
    o.scale.set(hull, 1.02, hull);
    m.add(o);
    return m;
  }

  function init(width, height) {
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
      renderer.setSize(width, height, false);
      renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
      renderer.shadowMap.enabled = true;

      scene = new THREE.Scene();
      scene.fog = new THREE.Fog(0x121218, 9, 24);
      camera = new THREE.PerspectiveCamera(38, width / height, 0.1, 60);
      camera.position.set(0, 1.35, 3.9);
      camera.lookAt(0, 1.0, 0);

      scene.add(new THREE.AmbientLight(0xffffff, 0.5));
      const key = new THREE.SpotLight(0xf5e9d0, 700, 30, 0.55, 0.6);
      key.position.set(0, 6.5, 3.5);
      key.castShadow = true;
      scene.add(key);
      const rim = new THREE.DirectionalLight(0x7ad6ff, 0.5);
      rim.position.set(-3, 3, -4);
      scene.add(rim);

      const floor = new THREE.Mesh(
        new THREE.CircleGeometry(14, 48),
        new THREE.MeshStandardMaterial({ color: 0x1d222a, roughness: 0.85 }));
      floor.rotation.x = -Math.PI / 2;
      floor.receiveShadow = true;
      scene.add(floor);
      const grid = new THREE.GridHelper(24, 24, 0x39424e, 0x272d36);
      grid.position.y = 0.002;
      scene.add(grid);

      group = new THREE.Group();
      scene.add(group);
      for (const [key2, , , rad] of LIMB_DEFS) {
        const m = outlined(new THREE.CapsuleGeometry(rad, 1, 4, 10), IVORY, 1.3);
        limbs[key2] = m;
        group.add(m);
      }
      for (const name of ['l_shoulder', 'r_shoulder', 'l_elbow', 'r_elbow',
        'l_hip', 'r_hip', 'l_knee', 'r_knee']) {
        const s = outlined(new THREE.SphereGeometry(0.055, 12, 10), IVORY);
        joints[name] = s;
        group.add(s);
      }
      for (const name of ['l_wrist', 'r_wrist', 'l_ankle', 'r_ankle']) {
        const s = outlined(new THREE.SphereGeometry(0.06, 12, 10), DARK, 1.0);
        joints[name] = s;
        group.add(s);
      }

      headMesh = outlined(new THREE.SphereGeometry(0.17, 24, 18), IVORY, 1.14);
      group.add(headMesh);
      // hair cap: the head's orientation landmark (rotates with the head)
      hairMesh = new THREE.Mesh(
        new THREE.SphereGeometry(0.178, 24, 12, 0, Math.PI * 2, 0, Math.PI * 0.55),
        toonMat(0x3a2a1e));
      hairMesh.rotation.x = -0.35;
      headMesh.add(hairMesh);

      // face: camera-facing billboard drawn with the designed 2D face
      faceCanvas = document.createElement('canvas');
      faceCanvas.width = faceCanvas.height = 256;
      faceTex = new THREE.CanvasTexture(faceCanvas);
      faceSprite = new THREE.Sprite(new THREE.SpriteMaterial({
        map: faceTex, transparent: true }));
      faceSprite.scale.set(0.42, 0.42, 1);
      group.add(faceSprite);

      // hand-skeleton billboards
      for (const side of ['l', 'r']) {
        const cv = document.createElement('canvas');
        cv.width = cv.height = 128;
        const tex = new THREE.CanvasTexture(cv);
        const sp = new THREE.Sprite(new THREE.SpriteMaterial({
          map: tex, transparent: true }));
        sp.scale.set(0.36, 0.36, 1);
        handSprites[side] = { sprite: sp, canvas: cv, tex };
        group.add(sp);
      }

      wallMesh = new THREE.Mesh(
        new THREE.PlaneGeometry(960 / PXPM, 760 / PXPM),
        new THREE.MeshStandardMaterial({
          transparent: true, alphaTest: 0.5, roughness: 0.9,
          side: THREE.DoubleSide }));
      wallMesh.castShadow = true;
      wallMesh.visible = false;
      scene.add(wallMesh);

      setLook(0);   // dress him: clothing makes 3D rotation readable
      ok = true;
    } catch (e) {
      ok = false;
    }
    return ok;
  }

  const V = (p) => new THREE.Vector3(p[0], -p[1], -p[2]);

  function setLimb(key, a, b) {
    const m = limbs[key];
    if (!a || !b) { m.visible = false; return; }
    m.visible = true;
    const va = V(a), vb = V(b);
    m.position.copy(va.clone().add(vb).multiplyScalar(0.5));
    m.scale.set(1, Math.max(0.05, va.distanceTo(vb)), 1);
    m.quaternion.setFromUnitVectors(
      new THREE.Vector3(0, 1, 0), vb.clone().sub(va).normalize());
  }

  function drawFace(live, gameFace) {
    const ctx = faceCanvas.getContext('2d');
    ctx.clearRect(0, 0, 256, 256);
    const cx = 128, cy = 118, r = 88;
    const f = live && live.face ? live.face : null;
    ctx.fillStyle = '#23232b';
    ctx.strokeStyle = '#23232b';
    ctx.lineWidth = 7;

    // eyes (X-eyes on crash, happy arcs on pass, live blinks otherwise)
    for (const [i, sx] of [[0, -1], [1, 1]]) {
      const ex = cx + sx * r * 0.34, ey = cy - r * 0.18;
      if (gameFace === 'hit') {
        ctx.beginPath();
        ctx.moveTo(ex - 12, ey - 12); ctx.lineTo(ex + 12, ey + 12);
        ctx.moveTo(ex - 12, ey + 12); ctx.lineTo(ex + 12, ey - 12);
        ctx.stroke();
      } else if (gameFace === 'win') {
        ctx.beginPath(); ctx.arc(ex, ey + 5, 12, 1.15 * Math.PI, 1.85 * Math.PI);
        ctx.stroke();
      } else {
        const bl = f ? (i === 0 ? f.blinkL : f.blinkR) : 0;
        if (bl > 0.5) {
          ctx.beginPath(); ctx.moveTo(ex - 12, ey); ctx.lineTo(ex + 12, ey); ctx.stroke();
        } else {
          ctx.beginPath(); ctx.arc(ex, ey, 9, 0, 7); ctx.fill();
        }
      }
    }

    // volumetric mouth: same treatment as the designed 2D face
    const my = cy + r * 0.35;
    if (f && f.mouth && f.mouth.o) {
      const ms = r * 2.1;
      const path = (ring) => {
        ctx.beginPath();
        ring.forEach((pt, i) => {
          const x = cx + pt[0] * ms, y = my + pt[1] * ms;
          if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        });
        ctx.closePath();
      };
      ctx.save();
      ctx.fillStyle = 'rgba(0,0,0,0.14)';
      ctx.translate(0, 5); path(f.mouth.o); ctx.fill();
      ctx.restore();
      ctx.fillStyle = '#c9847a'; ctx.strokeStyle = '#8a4a42'; ctx.lineWidth = 4;
      path(f.mouth.o); ctx.fill(); ctx.stroke();
      const openAmt = f.open || 0;
      if (openAmt > 0.12) {
        ctx.fillStyle = `rgba(42,17,20,${Math.min(1, (openAmt - 0.12) / 0.15)})`;
        path(f.mouth.i); ctx.fill();
      } else {
        ctx.strokeStyle = 'rgba(122,64,56,0.9)'; ctx.lineWidth = 3;
        path(f.mouth.i); ctx.stroke();
      }
    } else {
      ctx.beginPath();
      ctx.arc(cx, my, 20, 0.15 * Math.PI, 0.85 * Math.PI);
      ctx.stroke();
    }
    faceTex.needsUpdate = true;
  }

  const HAND_BONES = [
    [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8],
    [5, 9], [9, 10], [10, 11], [11, 12], [9, 13], [13, 14], [14, 15], [15, 16],
    [13, 17], [17, 18], [18, 19], [19, 20], [0, 17],
  ];

  function drawHand(side, shape) {
    const { canvas, tex, sprite } = handSprites[side];
    if (!shape) { sprite.visible = false; return; }
    sprite.visible = true;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, 128, 128);
    ctx.strokeStyle = '#23232b';
    ctx.lineWidth = 9;
    ctx.lineCap = 'round';
    const s = 26;
    for (const [a, b] of HAND_BONES) {
      ctx.beginPath();
      ctx.moveTo(64 + shape[a][0] * s, 64 + shape[a][1] * s);
      ctx.lineTo(64 + shape[b][0] * s, 64 + shape[b][1] * s);
      ctx.stroke();
    }
    tex.needsUpdate = true;
  }

  function setWallTexture(canvas) {
    if (wallTex) wallTex.dispose();
    wallTex = new THREE.CanvasTexture(canvas);
    wallTex.colorSpace = THREE.SRGBColorSpace;
    wallMesh.material.map = wallTex;
    wallMesh.material.needsUpdate = true;
  }

  let orbitT = 0;

  function update(S, live, ax, depthScale, gameFace) {
    if (!ok) return;
    orbitT += 0.016;
    const a = Math.sin(orbitT * 0.35) * 0.30;   // gentle sway around the stage
    camera.position.set(Math.sin(a) * 3.9, 1.35, Math.cos(a) * 3.9);
    camera.lookAt(0, 1.0, 0);
    const P = live && live.p3;
    group.visible = !!(P && P.l_shoulder && P.r_shoulder);
    if (group.visible) {
      const shc = [(P.l_shoulder[0] + P.r_shoulder[0]) / 2,
                   (P.l_shoulder[1] + P.r_shoulder[1]) / 2,
                   (P.l_shoulder[2] + P.r_shoulder[2]) / 2];
      const rel = (n) => P[n]
        ? [P[n][0] - shc[0], P[n][1] - shc[1], P[n][2] - shc[2]] : null;
      const J = {};
      for (const n of Object.keys(joints)) J[n] = rel(n);
      if (!J.l_hip) J.l_hip = [-0.09, 0.45, 0];
      if (!J.r_hip) J.r_hip = [0.09, 0.45, 0];
      if (!J.l_knee) J.l_knee = [J.l_hip[0] - 0.02, J.l_hip[1] + 0.4, 0];
      if (!J.r_knee) J.r_knee = [J.r_hip[0] + 0.02, J.r_hip[1] + 0.4, 0];
      if (!J.l_ankle) J.l_ankle = [J.l_knee[0] - 0.01, J.l_knee[1] + 0.38, 0];
      if (!J.r_ankle) J.r_ankle = [J.r_knee[0] + 0.01, J.r_knee[1] + 0.38, 0];
      J.shc = [0, 0, 0];
      J.hipc = [(J.l_hip[0] + J.r_hip[0]) / 2, (J.l_hip[1] + J.r_hip[1]) / 2,
                (J.l_hip[2] + J.r_hip[2]) / 2];

      for (const [key, aN, bN] of LIMB_DEFS) setLimb(key, J[aN], J[bN]);
      for (const n of Object.keys(joints)) {
        joints[n].visible = !!J[n];
        if (J[n]) joints[n].position.copy(V(J[n]));
      }
      const nose = rel('nose') || [0, -0.25, -0.05];
      const hp = V([nose[0] * 0.55, nose[1] * 0.8 - 0.07, nose[2] * 0.6]);
      headMesh.position.copy(hp);
      headMesh.rotation.set(-nose[1] * 0.8, -nose[0] * 1.6, 0);
      faceSprite.position.set(hp.x, hp.y, hp.z + 0.17);
      drawFace(live, gameFace);

      for (const [side, wr] of [['l', 'l_wrist'], ['r', 'r_wrist']]) {
        const sp = handSprites[side].sprite;
        drawHand(side, live && live.shapes ? live.shapes[side] : null);
        if (J[wr] && sp.visible) {
          const wp = V(J[wr]);
          sp.position.set(wp.x, wp.y, wp.z + 0.12);
        } else {
          sp.visible = false;
        }
      }

      group.position.set(ax / PXPM, SHOULDER_Y, 0);
      const ds = depthScale || 1;
      group.scale.set(ds, ds, ds);
    }

    if ((S.state === 'WALL' || (S.state === 'RESULT' && S.outcome === 'pass'))
        && wallTex) {
      wallMesh.visible = true;
      const z = S.state === 'WALL'
        ? -16 * (1 - Math.pow(S.progress, 2.2))
        : 6.0 * Math.pow(S.resultT, 1.4);
      // texture row HOLE_CY (280px) must sit at avatar chest height
      const chestY = SHOULDER_Y - 0.35;
      const planeCenterY = chestY + (280 - 760 / 2) / PXPM * -1;
      wallMesh.position.set((S.holeDx || 0) / PXPM, planeCenterY, z);
      const dr = 1 + 0.18 * (S.depthReq || 0);
      wallMesh.scale.set(dr, dr, 1);
    } else {
      wallMesh.visible = false;
    }

    renderer.render(scene, camera);
  }

  return {
    init, update, setWallTexture, setLook,
    nextLook: () => setLook(lookIdx + 1),
    drawFaceTexture: () => {},   // folded into update()
    get ok() { return ok; },
    get canvas() { return renderer ? renderer.domElement : null; },
  };
})();
