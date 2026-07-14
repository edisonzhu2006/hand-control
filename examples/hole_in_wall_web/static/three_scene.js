/* True-3D scene: floor, lights, approaching wall, and a capsule-limb stickman
 * driven by the player's 3D world landmarks. Renders on its own WebGL canvas
 * under the (transparent) Pixi HUD. Falls back cleanly: if init fails, the
 * classic 2.5D Pixi renderer keeps running.
 *
 * Coordinate mapping: pose3d is meters, shoulder-anchored, x right / y DOWN /
 * z away from camera. Three is y UP, camera looks down -z. We negate y and z:
 * player-toward-camera (data z<0) becomes three z>0 (toward the viewer).
 */

const T3 = (() => {
  if (typeof THREE === 'undefined') return null;

  const PXPM = 195;          // texture pixels per meter (matches avatar scale)
  const SHOULDER_Y = 0.62;   // avatar shoulder height above the floor (m)

  let renderer, scene, camera, wallMesh, wallTex, wallCanvas;
  let faceCanvas, faceTex, headMesh;
  const limbs = {};          // name -> {mesh, from, to, radius}
  const joints = {};         // name -> sphere mesh
  let group;                 // whole avatar
  let ok = false;

  const IVORY = 0xf0ede4, DARK = 0x23232b;

  const LIMB_DEFS = [
    ['luA', 'l_shoulder', 'l_elbow', 0.045], ['lfA', 'l_elbow', 'l_wrist', 0.04],
    ['ruA', 'r_shoulder', 'r_elbow', 0.045], ['rfA', 'r_elbow', 'r_wrist', 0.04],
    ['ltL', 'l_hip', 'l_knee', 0.055], ['lsL', 'l_knee', 'l_ankle', 0.05],
    ['rtL', 'r_hip', 'r_knee', 0.055], ['rsL', 'r_knee', 'r_ankle', 0.05],
    ['shB', 'l_shoulder', 'r_shoulder', 0.05],
    ['hpB', 'l_hip', 'r_hip', 0.05],
    ['tor', 'shc', 'hipc', 0.085],
  ];

  function init(width, height) {
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
      renderer.setSize(width, height, false);
      renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
      renderer.shadowMap.enabled = true;

      scene = new THREE.Scene();
      scene.fog = new THREE.Fog(0x121218, 8, 22);
      camera = new THREE.PerspectiveCamera(38, width / height, 0.1, 60);
      camera.position.set(0, 1.15, 3.6);
      camera.lookAt(0, 0.65, 0);

      scene.add(new THREE.AmbientLight(0xffffff, 0.55));
      const key = new THREE.SpotLight(0xf5e9d0, 900, 30, 0.5, 0.5);
      key.position.set(0, 6.5, 3.5);
      key.castShadow = true;
      scene.add(key);
      const rim = new THREE.DirectionalLight(0x7ad6ff, 0.7);
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
      const pool = new THREE.Mesh(
        new THREE.CircleGeometry(1.4, 40),
        new THREE.MeshBasicMaterial({ color: 0x3a3428, transparent: true, opacity: 0.5 }));
      pool.rotation.x = -Math.PI / 2;
      pool.position.y = 0.004;
      scene.add(pool);

      // avatar rig
      group = new THREE.Group();
      scene.add(group);
      const mat = new THREE.MeshStandardMaterial({ color: IVORY, roughness: 0.6 });
      const darkMat = new THREE.MeshStandardMaterial({ color: DARK, roughness: 0.7 });
      for (const [key2, , , rad] of LIMB_DEFS) {
        const m = new THREE.Mesh(
          new THREE.CapsuleGeometry(rad, 1, 4, 10),
          key2 === 'tor' ? mat : mat);
        m.castShadow = true;
        limbs[key2] = m;
        group.add(m);
      }
      for (const name of ['l_shoulder', 'r_shoulder', 'l_elbow', 'r_elbow',
        'l_hip', 'r_hip', 'l_knee', 'r_knee']) {
        const s = new THREE.Mesh(new THREE.SphereGeometry(0.05, 12, 10), mat);
        s.castShadow = true;
        joints[name] = s;
        group.add(s);
      }
      for (const name of ['l_wrist', 'r_wrist']) {
        const s = new THREE.Mesh(new THREE.SphereGeometry(0.055, 12, 10), darkMat);
        s.castShadow = true;
        joints[name] = s;
        group.add(s);
      }
      for (const name of ['l_ankle', 'r_ankle']) {
        const s = new THREE.Mesh(new THREE.SphereGeometry(0.06, 12, 10), darkMat);
        s.castShadow = true;
        joints[name] = s;
        group.add(s);
      }

      // head with a live canvas face
      faceCanvas = document.createElement('canvas');
      faceCanvas.width = faceCanvas.height = 256;
      faceTex = new THREE.CanvasTexture(faceCanvas);
      headMesh = new THREE.Mesh(
        new THREE.SphereGeometry(0.16, 24, 18),
        [
          new THREE.MeshStandardMaterial({ color: IVORY, roughness: 0.6 }),
        ][0]);
      headMesh.material = new THREE.MeshStandardMaterial({
        color: 0xffffff, roughness: 0.6, map: faceTex });
      headMesh.castShadow = true;
      group.add(headMesh);

      // wall plane (texture swapped per wall)
      wallCanvas = null;
      wallTex = null;
      wallMesh = new THREE.Mesh(
        new THREE.PlaneGeometry(960 / PXPM, 760 / PXPM),
        new THREE.MeshStandardMaterial({
          transparent: true, alphaTest: 0.5, roughness: 0.9,
          side: THREE.DoubleSide }));
      wallMesh.castShadow = true;
      wallMesh.visible = false;
      scene.add(wallMesh);

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
    const mid = va.clone().add(vb).multiplyScalar(0.5);
    const len = Math.max(0.05, va.distanceTo(vb));
    m.position.copy(mid);
    m.scale.set(1, len, 1);
    m.quaternion.setFromUnitVectors(
      new THREE.Vector3(0, 1, 0), vb.clone().sub(va).normalize());
  }

  function drawFaceTexture(face, live) {
    const ctx = faceCanvas.getContext('2d');
    ctx.fillStyle = '#f0ede4';
    ctx.fillRect(0, 0, 256, 256);
    // face lives on the +z hemisphere; sphere UV front center ~ (0.5, 0.55)
    const cx = 128, cy = 132, r = 52;
    const blinkL = live && live.face ? live.face.blinkL : 0;
    const blinkR = live && live.face ? live.face.blinkR : 0;
    ctx.fillStyle = '#23232b';
    ctx.strokeStyle = '#23232b';
    ctx.lineWidth = 5;
    for (const [bl, sx] of [[blinkL, -1], [blinkR, 1]]) {
      const ex = cx + sx * r * 0.55, ey = cy - r * 0.35;
      if (bl > 0.5) {
        ctx.beginPath(); ctx.moveTo(ex - 10, ey); ctx.lineTo(ex + 10, ey); ctx.stroke();
      } else {
        ctx.beginPath(); ctx.arc(ex, ey, 8, 0, 7); ctx.fill();
      }
    }
    const mouth = live && live.face ? live.face.mouth : null;
    if (mouth && mouth.o) {
      const ms = r * 2.6;
      const my = cy + r * 0.55;
      ctx.fillStyle = '#c9847a';
      ctx.strokeStyle = '#8a4a42';
      ctx.lineWidth = 3;
      ctx.beginPath();
      mouth.o.forEach((pt, i) => {
        const x = cx + pt[0] * ms, y = my + pt[1] * ms;
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      });
      ctx.closePath(); ctx.fill(); ctx.stroke();
      const openAmt = live.face.open || 0;
      if (openAmt > 0.12) {
        ctx.fillStyle = `rgba(42,17,20,${Math.min(1, (openAmt - 0.12) / 0.15)})`;
        ctx.beginPath();
        mouth.i.forEach((pt, i) => {
          const x = cx + pt[0] * ms, y = my + pt[1] * ms;
          if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        });
        ctx.closePath(); ctx.fill();
      }
    } else {
      ctx.beginPath();
      ctx.arc(cx, cy + r * 0.5, 14, 0.15 * Math.PI, 0.85 * Math.PI);
      ctx.stroke();
    }
    faceTex.needsUpdate = true;
  }

  function setWallTexture(canvas) {
    wallCanvas = canvas;
    if (wallTex) wallTex.dispose();
    wallTex = new THREE.CanvasTexture(canvas);
    wallTex.colorSpace = THREE.SRGBColorSpace;
    wallMesh.material.map = wallTex;
    wallMesh.material.needsUpdate = true;
  }

  function update(S, live, ax, depthScale) {
    if (!ok) return;
    const P = live && live.p3;
    group.visible = !!P;
    if (P && P.l_shoulder && P.r_shoulder) {
      const shc = [(P.l_shoulder[0] + P.r_shoulder[0]) / 2,
                   (P.l_shoulder[1] + P.r_shoulder[1]) / 2,
                   (P.l_shoulder[2] + P.r_shoulder[2]) / 2];
      const rel = (name) => P[name]
        ? [P[name][0] - shc[0], P[name][1] - shc[1], P[name][2] - shc[2]]
        : null;
      const J = {};
      for (const name of Object.keys(joints)) J[name] = rel(name);
      // default legs when untracked: straight down from hips
      if (!J.l_hip) J.l_hip = [-0.09, 0.45, 0];
      if (!J.r_hip) J.r_hip = [0.09, 0.45, 0];
      if (!J.l_knee) J.l_knee = [J.l_hip[0] - 0.02, J.l_hip[1] + 0.4, 0];
      if (!J.r_knee) J.r_knee = [J.r_hip[0] + 0.02, J.r_hip[1] + 0.4, 0];
      if (!J.l_ankle) J.l_ankle = [J.l_knee[0] - 0.01, J.l_knee[1] + 0.38, 0];
      if (!J.r_ankle) J.r_ankle = [J.r_knee[0] + 0.01, J.r_knee[1] + 0.38, 0];
      J.shc = [0, 0, 0];
      J.hipc = [(J.l_hip[0] + J.r_hip[0]) / 2, (J.l_hip[1] + J.r_hip[1]) / 2,
                (J.l_hip[2] + J.r_hip[2]) / 2];

      for (const [key, aN, bN] of LIMB_DEFS.map(d => [d[0], d[1], d[2]])) {
        setLimb(key, J[aN], J[bN]);
      }
      for (const name of Object.keys(joints)) {
        if (J[name]) { joints[name].visible = true; joints[name].position.copy(V(J[name])); }
        else joints[name].visible = false;
      }
      const nose = rel('nose') || [0, -0.25, -0.05];
      headMesh.position.copy(V([nose[0] * 0.55, nose[1] * 0.8 - 0.06, nose[2] * 0.6]));
      // face the camera-ish, tilting with the nose offset
      headMesh.rotation.set(-nose[1] * 0.6, nose[0] * 1.2, 0);

      group.position.set(ax / PXPM, SHOULDER_Y + 0.62, 0);
      const ds = depthScale || 1;
      group.scale.set(ds, ds, ds);
    }

    // wall approach in real z
    if (S.state === 'WALL' || (S.state === 'RESULT' && S.outcome === 'pass')) {
      wallMesh.visible = !!wallTex;
      let zt;
      if (S.state === 'WALL') {
        zt = -16 * (1 - Math.pow(S.progress, 2.2));
      } else {
        zt = 2.5 * Math.pow(S.resultT, 1.4);   // flies past the camera
      }
      wallMesh.position.set((S.holeDx || 0) / PXPM, 760 / PXPM / 2 - (280 - 195) / PXPM - SHOULDER_Y + 1.05, zt);
      const dr = 1 + 0.18 * (S.depthReq || 0);
      wallMesh.scale.set(dr, dr, 1);
    } else {
      wallMesh.visible = false;
    }

    renderer.render(scene, camera);
  }

  return {
    init, update, setWallTexture, drawFaceTexture,
    get ok() { return ok; },
    get canvas() { return renderer ? renderer.domElement : null; },
  };
})();
