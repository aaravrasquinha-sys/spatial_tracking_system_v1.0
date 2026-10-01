import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { PLYLoader } from "three/addons/loaders/PLYLoader.js";
import { VRButton } from "three/addons/webxr/VRButton.js";
import { XRControllerModelFactory } from "three/addons/webxr/XRControllerModelFactory.js";

import { PoiClient, wsUrl } from "../common/ws-client.js";
import { SceneClient } from "../common/scene-client.js";
import { validateTracksMessage, warnOnce } from "../common/validate.js";

/*
 * Coordinate convention (SCHEMA.md / the master plan's section 3):
 *   room/local frame: +X right, +Y "forward" (floor-referenced,
 *   right-handed), Z up, floor at z=0.
 *   three.js: Y up.
 * The ONE conversion point is roomToThree() below and the single
 * `roomRoot` group everything else is parented under -- nothing else
 * in this file touches the axis order.
 */
function roomToThree(p, out = new THREE.Vector3()) {
  return out.set(p[0], p[2], -p[1]);
}

// ---------------------------------------------------------------- tokens

const css = getComputedStyle(document.documentElement);
const cssVar = (name) => css.getPropertyValue(name).trim();
const COLORS = {
  confirmed: new THREE.Color(cssVar("--state-confirmed") || "#49e8c8"),
  coasting: new THREE.Color(cssVar("--state-coasting") || "#eeb43e"),
  lost: new THREE.Color(cssVar("--state-lost") || "#ff6a6a"),
  tentative: new THREE.Color(cssVar("--state-tentative") || "#6b7785"),
  grid: new THREE.Color(cssVar("--grid-line") || "#1b232c"),
  camera: new THREE.Color(cssVar("--text") || "#a9b6c2"),
  frustum: new THREE.Color(cssVar("--src-depth") || "#5b9dff"),
};

// ---------------------------------------------------------------- renderer / scene

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.xr.enabled = true;
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.domElement.classList.add("vr-canvas");
document.body.appendChild(renderer.domElement);
document.body.appendChild(VRButton.createButton(renderer));

const scene3d = new THREE.Scene();
scene3d.background = new THREE.Color(cssVar("--bg") || "#0a0e13");
scene3d.fog = new THREE.Fog(scene3d.background.getHex(), 8, 30);

// Desktop preview camera (also used as the XR camera's parent rig).
const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.05, 100);
camera.position.set(0, 1.6, 3.2);

// The XR rig: moving this group is how dollhouse-scale and
// life-size teleport locomotion both work, without ever touching
// the XR camera itself (which WebXR owns).
const rig = new THREE.Group();
rig.add(camera);
scene3d.add(rig);

scene3d.add(new THREE.HemisphereLight(0x8fa6bd, 0x0b0e12, 0.9));
const keyLight = new THREE.DirectionalLight(0xffffff, 0.6);
keyLight.position.set(3, 6, 2);
scene3d.add(keyLight);

// Everything room-relative lives under this one group -- see
// roomToThree()'s docstring above.
const roomRoot = new THREE.Group();
scene3d.add(roomRoot);

// ---------------------------------------------------------------- modes

const MODE = { DOLLHOUSE: "dollhouse", LIFESIZE: "lifesize" };
let mode = MODE.DOLLHOUSE;
const DOLLHOUSE_SCALE = 1 / 20;
const DOLLHOUSE_TABLE_HEIGHT = 0.95;
const DOLLHOUSE_DISTANCE = 0.55;

function applyMode() {
  if (mode === MODE.DOLLHOUSE) {
    roomRoot.scale.setScalar(DOLLHOUSE_SCALE);
    roomRoot.position.set(0, DOLLHOUSE_TABLE_HEIGHT, -DOLLHOUSE_DISTANCE);
    rig.position.set(0, 0, 0);
  } else {
    roomRoot.scale.setScalar(1);
    roomRoot.position.set(0, 0, 0);
    // rig.position is the life-size "teleport" position -- left as-is
    // when switching INTO life-size so the viewer doesn't get moved
    // underfoot; teleportTo() below is what actually relocates it.
  }
}
applyMode();

// ---------------------------------------------------------------- environment (Module 1 fallback chain)

const floorGroup = new THREE.Group();
roomRoot.add(floorGroup);

function buildProceduralFloor(halfExtent = 4) {
  floorGroup.clear();
  const grid = new THREE.GridHelper(halfExtent * 2, halfExtent * 2, COLORS.grid, COLORS.grid);
  grid.material.opacity = 0.55;
  grid.material.transparent = true;
  floorGroup.add(grid);

  const floorMat = new THREE.MeshStandardMaterial({ color: 0x0d1319, roughness: 1, metalness: 0 });
  const floorMesh = new THREE.Mesh(new THREE.PlaneGeometry(halfExtent * 2, halfExtent * 2), floorMat);
  floorMesh.rotation.x = -Math.PI / 2;
  floorMesh.position.y = -0.005;
  floorGroup.add(floorMesh);
}
buildProceduralFloor();

const gltfLoader = new GLTFLoader();
const plyLoader = new PLYLoader();

async function loadEnvironment(doc) {
  const meshAsset = (doc.assets || []).find((a) => a.kind === "mesh");
  const pointsAsset = (doc.assets || []).find((a) => a.kind === "points");

  if (meshAsset) {
    try {
      const gltf = await gltfLoader.loadAsync(meshAsset.url);
      floorGroup.clear();
      floorGroup.add(gltf.scene);
      return;
    } catch (e) {
      console.warn("failed to load map mesh, falling back:", e);
    }
  }
  if (pointsAsset) {
    try {
      const geometry = await plyLoader.loadAsync(pointsAsset.url);
      floorGroup.clear();
      const mat = new THREE.PointsMaterial({ size: 0.01, vertexColors: geometry.hasAttribute("color") });
      floorGroup.add(new THREE.Points(geometry, mat));
      return;
    } catch (e) {
      console.warn("failed to load map points, falling back:", e);
    }
  }
  // Fallback 3: floor grid + camera only (Phase A, or Module 1 not
  // built yet) -- buildProceduralFloor() already ran above.
  buildProceduralFloor();
}

// ---------------------------------------------------------------- camera marker + frustum

const cameraMarker = new THREE.Group();
roomRoot.add(cameraMarker);
{
  const body = new THREE.Mesh(
    new THREE.BoxGeometry(0.08, 0.08, 0.12),
    new THREE.MeshStandardMaterial({ color: COLORS.camera, roughness: 0.6 })
  );
  cameraMarker.add(body);
}
const frustumLines = new THREE.LineSegments(
  new THREE.BufferGeometry(),
  new THREE.LineBasicMaterial({ color: COLORS.frustum, transparent: true, opacity: 0.5 })
);
roomRoot.add(frustumLines);

const deadZoneMesh = new THREE.Mesh(
  new THREE.RingGeometry(0.01, 6, 48, 1, 0, Math.PI * 2),
  new THREE.MeshBasicMaterial({ color: 0x1a1010, transparent: true, opacity: 0.35, side: THREE.DoubleSide })
);
deadZoneMesh.rotation.x = -Math.PI / 2;
deadZoneMesh.position.y = 0.002;
deadZoneMesh.visible = false;
roomRoot.add(deadZoneMesh);

function updateCameraAndFrustum(doc) {
  const t = doc.camera.t; // [x, y, height] in room coords
  const R = doc.camera.R; // 3x3, room = R @ cam + t

  const posRoom = [t[0], t[1], t[2]];
  roomToThree(posRoom, cameraMarker.position);

  // Camera forward in room space is R's third column (R @ [0,0,1]).
  const fwdRoom = [R[0][2], R[1][2], R[2][2]];
  const fwd3 = roomToThree([t[0] + fwdRoom[0], t[1] + fwdRoom[1], t[2] + fwdRoom[2]]);
  cameraMarker.lookAt(fwd3);

  const fovH = ((doc.fov_deg && doc.fov_deg[0]) || 60) * (Math.PI / 180);
  const fovV = ((doc.fov_deg && doc.fov_deg[1]) || 40) * (Math.PI / 180);
  const range = 6;
  const corners = [
    [Math.tan(fovH / 2), Math.tan(fovV / 2)],
    [-Math.tan(fovH / 2), Math.tan(fovV / 2)],
    [-Math.tan(fovH / 2), -Math.tan(fovV / 2)],
    [Math.tan(fovH / 2), -Math.tan(fovV / 2)],
  ].map(([hx, hy]) => new THREE.Vector3(hx * range, hy * range, range));

  const origin = new THREE.Vector3(0, 0, 0);
  const pts = [];
  for (const c of corners) { pts.push(origin.clone(), c.clone()); }
  for (let i = 0; i < 4; i++) { pts.push(corners[i].clone(), corners[(i + 1) % 4].clone()); }
  frustumLines.geometry.dispose();
  frustumLines.geometry = new THREE.BufferGeometry().setFromPoints(pts);
  frustumLines.position.copy(cameraMarker.position);
  frustumLines.quaternion.copy(cameraMarker.quaternion);
}

// ---------------------------------------------------------------- track objects

const TRAIL_SECONDS = 4.0;
const SAMPLE_BUFFER_MAX = 90; // ~3s at 30Hz
const RENDER_DELAY_S = 0.1;
const EXTRAPOLATE_MAX_S = 0.15;
const STALE_REMOVE_S = 3.0; // no fresh sample for this long -> drop the object

class TrackObject {
  constructor(id) {
    this.id = id;
    this.samples = []; // {t, p:[x,y,z], v:[x,y,z], state, src, height}
    this.lastSeenWall = performance.now();

    this.group = new THREE.Group();

    const geo = new THREE.CapsuleGeometry(0.18, 1.2, 4, 8);
    geo.translate(0, 0.18 + 0.6, 0);
    this.material = new THREE.MeshStandardMaterial({
      color: COLORS.tentative, emissive: COLORS.tentative, emissiveIntensity: 0.35, roughness: 0.5,
    });
    this.capsule = new THREE.Mesh(geo, this.material);
    this.group.add(this.capsule);

    this.label = makeLabelSprite(`#${id}`);
    this.label.position.set(0, 2.0, 0);
    this.group.add(this.label);

    this.trailGeom = new THREE.BufferGeometry();
    this.trailMat = new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.8 });
    this.trail = new THREE.Line(this.trailGeom, this.trailMat);
    this.group.add(this.trail);

    this.ellipse = new THREE.Mesh(
      new THREE.RingGeometry(0.01, 1, 48),
      new THREE.MeshBasicMaterial({ color: COLORS.tentative, transparent: true, opacity: 0.5, side: THREE.DoubleSide })
    );
    this.ellipse.rotation.x = -Math.PI / 2;
    this.ellipse.position.y = 0.005;
    this.group.add(this.ellipse);

    roomRoot.add(this.group);
  }

  pushSample(tr, t_capture) {
    this.samples.push({
      t: t_capture, p: tr.p, v: tr.v, state: tr.state, src: tr.src,
      height: tr.height, cov_xy: tr.cov_xy,
    });
    if (this.samples.length > SAMPLE_BUFFER_MAX) this.samples.shift();
    this.lastSeenWall = performance.now();
  }

  interpolatedAt(targetT) {
    const s = this.samples;
    if (s.length === 0) return null;
    if (targetT <= s[0].t) return s[0];
    const last = s[s.length - 1];
    if (targetT >= last.t) {
      const dt = Math.min(targetT - last.t, EXTRAPOLATE_MAX_S);
      return {
        ...last,
        p: [last.p[0] + last.v[0] * dt, last.p[1] + last.v[1] * dt, 0],
      };
    }
    for (let i = 1; i < s.length; i++) {
      if (s[i].t >= targetT) {
        const a = s[i - 1], b = s[i];
        const span = Math.max(b.t - a.t, 1e-6);
        const f = (targetT - a.t) / span;
        return {
          ...b,
          p: [a.p[0] + (b.p[0] - a.p[0]) * f, a.p[1] + (b.p[1] - a.p[1]) * f, 0],
        };
      }
    }
    return last;
  }

  update(targetT, layers) {
    const sample = this.interpolatedAt(targetT);
    if (!sample) return;

    roomToThree(sample.p, this.group.position);
    const color = COLORS[sample.state] || COLORS.tentative;
    this.material.color.copy(color);
    this.material.emissive.copy(color);
    this.material.opacity = sample.state === "lost" ? 0.4 : 1;
    this.material.transparent = sample.state === "lost";

    const h = sample.height || 1.72;
    this.capsule.scale.set(1, h / 1.72, 1);

    if (layers.ellipses && sample.cov_xy) {
      const [a, b, c] = sample.cov_xy;
      const trace = a + c, det = a * c - b * b;
      const disc = Math.sqrt(Math.max(0, trace * trace / 4 - det));
      const l1 = Math.max(trace / 2 + disc, 1e-6), l2 = Math.max(trace / 2 - disc, 1e-6);
      const angle = Math.abs(b) < 1e-12 ? 0 : 0.5 * Math.atan2(2 * b, a - c);
      const k = 2;
      this.ellipse.scale.set(k * Math.sqrt(l1), k * Math.sqrt(l2), 1);
      this.ellipse.rotation.z = angle;
      this.ellipse.material.color.copy(color);
      this.ellipse.visible = true;
    } else {
      this.ellipse.visible = false;
    }

    this._updateTrail(targetT, layers.trails, color);

    const dashed = sample.state === "coasting" || sample.src === "predicted";
    this.capsule.material.wireframe = false;
    this.group.visible = true;
    this.capsule.visible = true;
    if (dashed) this.material.opacity = Math.min(this.material.opacity, 0.6), (this.material.transparent = true);
  }

  _updateTrail(targetT, visible, color) {
    this.trail.visible = visible;
    if (!visible) return;
    const cutoff = targetT - TRAIL_SECONDS;
    const pts = [];
    const colors = [];
    for (const s of this.samples) {
      if (s.t < cutoff || s.t > targetT) continue;
      const v3 = roomToThree(s.p);
      // Trail is drawn relative to the group's own (already-updated)
      // position, so subtract it back out.
      pts.push(v3.x - this.group.position.x, 0.02, v3.z - this.group.position.z);
      const age = Math.max(0, targetT - s.t);
      const alpha = Math.max(0, 1 - age / TRAIL_SECONDS);
      colors.push(color.r, color.g, color.b, alpha);
    }
    if (pts.length < 6) { this.trail.visible = false; return; }
    this.trailGeom.setAttribute("position", new THREE.Float32BufferAttribute(pts, 3));
    // Plain vertex color (no alpha channel support in LineBasicMaterial
    // vertexColors) -- fade the material's own opacity by the newest
    // sample's age instead, which is visually close enough at trail length.
    const flatColors = [];
    for (let i = 0; i < pts.length / 3; i++) flatColors.push(color.r, color.g, color.b);
    this.trailGeom.setAttribute("color", new THREE.Float32BufferAttribute(flatColors, 3));
    this.trailGeom.computeBoundingSphere();
  }

  dispose() {
    roomRoot.remove(this.group);
    this.capsule.geometry.dispose();
    this.material.dispose();
    this.trailGeom.dispose();
    this.trailMat.dispose();
    this.ellipse.geometry.dispose();
    this.ellipse.material.dispose();
    this.label.material.map.dispose();
    this.label.material.dispose();
  }
}

function makeLabelSprite(text) {
  const canvas = document.createElement("canvas");
  canvas.width = 128; canvas.height = 48;
  const ctx2d = canvas.getContext("2d");
  ctx2d.fillStyle = "rgba(10,14,19,0.7)";
  ctx2d.fillRect(0, 0, canvas.width, canvas.height);
  ctx2d.font = "600 28px 'IBM Plex Mono', monospace";
  ctx2d.fillStyle = "#e9eef3";
  ctx2d.textAlign = "center";
  ctx2d.textBaseline = "middle";
  ctx2d.fillText(text, canvas.width / 2, canvas.height / 2);
  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  const mat = new THREE.SpriteMaterial({ map: tex, depthTest: false, transparent: true });
  const sprite = new THREE.Sprite(mat);
  sprite.scale.set(0.4, 0.15, 1);
  sprite.renderOrder = 10;
  return sprite;
}

const tracks = new Map(); // id -> TrackObject

function ensureTrackObject(id) {
  let obj = tracks.get(id);
  if (!obj) { obj = new TrackObject(id); tracks.set(id, obj); }
  return obj;
}

function pruneStaleTracks() {
  const now = performance.now();
  for (const [id, obj] of tracks) {
    if (now - obj.lastSeenWall > STALE_REMOVE_S * 1000) {
      obj.dispose();
      tracks.delete(id);
    }
  }
}

// ---------------------------------------------------------------- connection + state

const layers = { trails: true, ellipses: true, deadZones: false };
let lastTracksMsg = null;
let sceneDoc = null;

const client = new PoiClient({ url: wsUrl() });
const sceneClient = new SceneClient();

const $ = (id) => document.getElementById(id);
const hudDot = $("conn-dot"), hudLabel = $("conn-label"), hudStats = $("hud-stats");
const frameBadge = $("frame-badge"), provisionalBanner = $("provisional-banner");
const perfFps = $("perf-fps");

client.addEventListener("status", (e) => {
  hudDot.className = "dot is-" + e.detail.state;
  hudLabel.textContent = {
    connecting: "connecting", open: "live", closed: "reconnecting\u2026", stalled: "stream stalled",
  }[e.detail.state] || e.detail.state;
});

client.addEventListener("tracks", (e) => {
  const msg = e.detail;
  warnOnce(validateTracksMessage(msg));
  lastTracksMsg = msg;
  sceneClient.noticeTracksMessage(msg).catch(() => {});

  const isRoom = msg.frame === "room";
  frameBadge.textContent = msg.frame;
  frameBadge.className = "frame-badge " + (isRoom ? "is-room" : "is-local");
  provisionalBanner.hidden = isRoom;

  for (const tr of msg.tracks) {
    ensureTrackObject(tr.id).pushSample(tr, msg.t_capture);
  }
});

client.addEventListener("health", (e) => {
  const msg = e.detail;
  const lat = msg.latency_ms == null ? "clock unsynced" : Math.round(msg.latency_ms) + " ms";
  hudStats.textContent = msg.fps.toFixed(1) + " fps \u00b7 " + lat;
});

sceneClient.addEventListener("scene", (e) => {
  sceneDoc = e.detail;
  loadEnvironment(sceneDoc);
  updateCameraAndFrustum(sceneDoc);
});

client.connect();
sceneClient.fetch().catch((err) => console.warn("initial /api/scene fetch failed:", err));

// ---------------------------------------------------------------- controls (2D overlay, pre-VR)

const btnMode = $("btn-mode"), btnTrails = $("btn-trails"), btnEllipses = $("btn-ellipses"), btnDeadZones = $("btn-deadzones");

btnMode.addEventListener("click", () => {
  mode = mode === MODE.DOLLHOUSE ? MODE.LIFESIZE : MODE.DOLLHOUSE;
  btnMode.textContent = mode === MODE.DOLLHOUSE ? "Dollhouse" : "Life-size";
  applyMode();
});
btnTrails.addEventListener("click", () => {
  layers.trails = !layers.trails;
  btnTrails.classList.toggle("is-active", layers.trails);
});
btnEllipses.addEventListener("click", () => {
  layers.ellipses = !layers.ellipses;
  btnEllipses.classList.toggle("is-active", layers.ellipses);
});
btnDeadZones.addEventListener("click", () => {
  layers.deadZones = !layers.deadZones;
  btnDeadZones.classList.toggle("is-active", layers.deadZones);
  deadZoneMesh.visible = layers.deadZones;
});

// ---------------------------------------------------------------- controllers (life-size teleport + mode toggle)

const controllerModelFactory = new XRControllerModelFactory();
const teleportMarker = new THREE.Mesh(
  new THREE.RingGeometry(0.15, 0.2, 32),
  new THREE.MeshBasicMaterial({ color: COLORS.confirmed, transparent: true, opacity: 0.8, side: THREE.DoubleSide })
);
teleportMarker.rotation.x = -Math.PI / 2;
teleportMarker.visible = false;
scene3d.add(teleportMarker);

function setupController(index) {
  const controller = renderer.xr.getController(index);
  const ray = new THREE.Line(
    new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0, 0, 0), new THREE.Vector3(0, 0, -1)]),
    new THREE.LineBasicMaterial({ color: COLORS.frustum })
  );
  ray.scale.z = 3;
  controller.add(ray);
  rig.add(controller);

  const grip = renderer.xr.getControllerGrip(index);
  grip.add(controllerModelFactory.createControllerModel(grip));
  rig.add(grip);

  controller.addEventListener("selectstart", () => {
    if (mode !== MODE.LIFESIZE) return;
    const target = raycastFloor(controller);
    if (target) teleportTo(target);
  });
  controller.addEventListener("squeezestart", () => {
    mode = mode === MODE.DOLLHOUSE ? MODE.LIFESIZE : MODE.DOLLHOUSE;
    applyMode();
  });
  return controller;
}
setupController(0);
setupController(1);

const _raycaster = new THREE.Raycaster();
function raycastFloor(controller) {
  const origin = new THREE.Vector3();
  const dir = new THREE.Vector3(0, 0, -1);
  controller.getWorldPosition(origin);
  dir.applyQuaternion(controller.getWorldQuaternion(new THREE.Quaternion()));
  if (dir.y >= -0.01) return null; // pointing up/level -- no sane floor hit
  const t = -origin.y / dir.y;
  if (t <= 0 || t > 20) return null;
  return origin.clone().addScaledVector(dir, t);
}

function teleportTo(point) {
  // Move the rig (not the XR camera, which WebXR owns) so the
  // camera's world (x,z) lands on `point` -- height is left alone.
  const camWorld = new THREE.Vector3();
  camera.getWorldPosition(camWorld);
  rig.position.x += point.x - camWorld.x;
  rig.position.z += point.z - camWorld.z;
}

renderer.xr.addEventListener("sessionstart", () => {
  const session = renderer.xr.getSession();
  session.addEventListener("selectstart", () => {}); // placeholder for future hand input
});

// ---------------------------------------------------------------- render loop

let frameCount = 0, fpsWindowStart = performance.now();

function animate() {
  renderer.setAnimationLoop(render);
}

function render() {
  const targetT = client.serverNowMs() / 1000 - RENDER_DELAY_S;

  if (lastTracksMsg) {
    const activeIds = new Set(lastTracksMsg.tracks.map((t) => t.id));
    for (const [id, obj] of tracks) {
      obj.group.visible = activeIds.has(id) || (performance.now() - obj.lastSeenWall < 1000);
    }
  }
  for (const obj of tracks.values()) obj.update(targetT, layers);
  pruneStaleTracks();

  renderer.render(scene3d, camera);

  frameCount++;
  const now = performance.now();
  if (now - fpsWindowStart > 500) {
    const fps = (frameCount * 1000) / (now - fpsWindowStart);
    perfFps.textContent = fps.toFixed(0) + " fps";
    frameCount = 0;
    fpsWindowStart = now;
  }
}

animate();

window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});
