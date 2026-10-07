STS 3D Viewer – Step-by-step change guide
Oct 7, 2026 · @Aarav
Overview and prep
You will add a desktop 3D viewer at http://192.168.1.101:8765/viewer3d/ with 2 small edits, 3 new files and one downloaded library. Part B (optional, last section) adds real 3D skeleton joints and touches the tracking code, so do Steps 1 to 7 first and check them live before attempting it.
All paths are relative to /home/drdo/spatial_tracking_system_v1.0 on the Jetson. Tested here against the synthetic source with a fake room scan; not yet run on your RealSense.
Step
File
Action
1
scripts/run_present.py
add 1 line
2
web/common/vendor/three/jsm/controls/OrbitControls.js
download
3
web/viewer3d/index.html
new file
4
web/viewer3d/style.css
new file
5
web/viewer3d/app.js
new file
6
web/index.html
add 1 link
Before you start, back up and branch so you can undo everything:
cd /home/drdo/spatial_tracking_system_v1.0
git status            # if this is a git checkout
git checkout -b viewer3d
# no git? then: cp -r web web.bak && cp scripts/run_present.py scripts/run_present.py.bak
The viewer needs no change to the running backend except Step 1. Restart run_present.py after Step 1 so it takes effect.
Step 1: fix the map ID in scripts/run_present.py
One line fixes a real bug: LiveSource is created with map_id=None, so on the first frame the server overwrites the scene's real map ID (7814e6723421) with null, and no viewer can show which map it is registered to.
Find this block inside _build_live (about line 78):
    T = m4_pipeline.floor_transform
    server.scene.update_camera(T.R.tolist(), T.t.tolist(), T.camera_height_m, frame_name, T.map_id)
Insert one line between the two so it reads:
    T = m4_pipeline.floor_transform
    live_source.map_id = T.map_id  # was left as None, which overwrote the scene's real map_id on the first frame
    server.scene.update_camera(T.R.tolist(), T.t.tolist(), T.camera_height_m, frame_name, T.map_id)
Nothing else changes. The existing dashboard keeps working.
Step 2: add OrbitControls (the mouse rotate/pan/zoom library)
Your repo already vendors Three.js r160 and PLYLoader, but not OrbitControls. It must be the same r160 release. On the Jetson (needs internet once):
cd /home/drdo/spatial_tracking_system_v1.0
mkdir -p web/common/vendor/three/jsm/controls
curl -fsSL https://raw.githubusercontent.com/mrdoob/three.js/r160/examples/jsm/controls/OrbitControls.js \
  -o web/common/vendor/three/jsm/controls/OrbitControls.js
wc -c web/common/vendor/three/jsm/controls/OrbitControls.js   # expect about 29868 bytes
head -12 web/common/vendor/three/jsm/controls/OrbitControls.js | grep -c "from 'three'"   # expect 1
No internet on the Jetson? Run the same curl on your laptop and scp the file to that path. The viewer does not load anything from a CDN at runtime, so it still works on an offline LAN afterwards.
Steps 3 to 5: create the three viewer files
Create the folder and the three files below. Paste each block exactly; the file name is in the heading above it.
mkdir -p /home/drdo/spatial_tracking_system_v1.0/web/viewer3d
Step 3: web/viewer3d/index.html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>poi_present · 3D viewer</title>
  <link rel="stylesheet" href="../common/tokens.css" />
  <link rel="stylesheet" href="style.css" />
  <script type="importmap">
    {
      "imports": {
        "three": "../common/vendor/three/build/three.module.js",
        "three/addons/controls/OrbitControls.js": "../common/vendor/three/jsm/controls/OrbitControls.js",
        "three/addons/loaders/PLYLoader.js": "../common/vendor/three/jsm/loaders/PLYLoader.js"
      }
    }
  </script>
</head>
<body>
  <div id="stage"></div>

  <header class="hud">
    <span class="dot" id="conn-dot"></span>
    <span class="mono" id="conn-label">connecting</span>
    <span class="badge" id="frame-badge">--</span>
    <span class="badge mono" id="map-badge" title="map_id">no map</span>
    <span class="mono muted" id="hud-stats">-- fps</span>
  </header>

  <div class="banner" id="banner" hidden></div>

  <aside class="panel panel-left" id="panel-left">
    <section>
      <h2>View</h2>
      <div class="btn-row">
        <button class="btn" data-view="top" title="Key 1">Top</button>
        <button class="btn" data-view="iso" title="Key 2">Iso</button>
        <button class="btn" data-view="sensor" title="Key 3">Sensor</button>
      </div>
    </section>

    <section>
      <h2>Layers</h2>
      <label class="chk"><input type="checkbox" id="l-points" checked /> Room point cloud</label>
      <label class="chk"><input type="checkbox" id="l-grid" checked /> Floor grid</label>
      <label class="chk"><input type="checkbox" id="l-walk" /> Walkable area</label>
      <label class="chk"><input type="checkbox" id="l-sensor" checked /> Sensor + frustum</label>
      <label class="chk"><input type="checkbox" id="l-box" checked /> Person boxes</label>
      <label class="chk"><input type="checkbox" id="l-skel" checked /> Skeletons (3D)</label>
      <label class="chk"><input type="checkbox" id="l-trails" checked /> Trails <kbd>T</kbd></label>
      <label class="chk"><input type="checkbox" id="l-ellipse" checked /> Uncertainty</label>
      <label class="chk"><input type="checkbox" id="l-labels" checked /> Labels <kbd>L</kbd></label>
    </section>

    <section>
      <h2>Point cloud</h2>
      <label class="rng">Ceiling cut <span class="mono" id="v-clip">off</span>
        <input type="range" id="s-clip" min="0.3" max="3" step="0.05" value="3" />
      </label>
      <label class="rng">Point size <span class="mono" id="v-psize">0.035</span>
        <input type="range" id="s-psize" min="0.01" max="0.10" step="0.005" value="0.035" />
      </label>
      <div class="mono muted small" id="map-info">no map loaded</div>
    </section>

    <section class="hint small muted">
      Drag = orbit · right-drag = pan · wheel = zoom<br />
      Double-click a person to follow · <kbd>Esc</kbd> to release
    </section>
  </aside>

  <aside class="panel panel-right" id="panel-right">
    <section>
      <h2>People <span class="mono muted" id="people-count">0</span></h2>
      <div id="track-list" class="track-list"></div>
    </section>
    <section>
      <h2>Events</h2>
      <div id="event-log" class="event-log mono small"></div>
    </section>
  </aside>

  <div class="toast" id="toast" hidden></div>
  <div class="perf mono" id="perf">-- fps</div>

  <script type="module" src="app.js"></script>
</body>
</html>
Step 4: web/viewer3d/style.css
html, body { width: 100%; height: 100%; overflow: hidden; background: var(--bg); }

#stage { position: fixed; inset: 0; }
#stage canvas { display: block; width: 100%; height: 100%; touch-action: none; }

.muted { color: var(--text-muted); }
.small { font-size: 11px; }

kbd {
  font-family: var(--font-mono);
  font-size: 10px;
  padding: 0 4px;
  border: 1px solid var(--border-strong);
  border-radius: var(--radius-sm);
  color: var(--text-muted);
}

/* ---------------------------------------------------------------- HUD */
.hud {
  position: fixed;
  top: var(--space-3);
  left: 50%;
  transform: translateX(-50%);
  z-index: 10;
  display: flex;
  align-items: center;
  gap: var(--space-2);
  padding: 6px 14px;
  font-size: 12px;
  color: var(--text);
  background: color-mix(in srgb, var(--surface) 88%, transparent);
  border: 1px solid var(--border);
  border-radius: var(--radius-pill);
  backdrop-filter: blur(4px);
  white-space: nowrap;
}
.hud .dot { width: 7px; height: 7px; border-radius: 50%; background: var(--text-dim); }
.hud .dot.is-open { background: var(--accent-live); }
.hud .dot.is-connecting { background: var(--state-coasting); }
.hud .dot.is-stalled, .hud .dot.is-closed { background: var(--danger); }

.badge {
  font-family: var(--font-mono);
  font-size: 10.5px;
  padding: 1px 7px;
  border-radius: var(--radius-pill);
  border: 1px solid var(--border-strong);
  color: var(--text-muted);
}
.badge.is-room { color: var(--state-confirmed); }
.badge.is-local { color: var(--accent-provisional); }

.banner {
  position: fixed;
  top: 52px;
  left: 50%;
  transform: translateX(-50%);
  z-index: 10;
  padding: 5px 14px;
  font-size: 11.5px;
  font-weight: 500;
  letter-spacing: 0.02em;
  color: #1a1204;
  background: var(--accent-provisional);
  border-radius: var(--radius-sm);
}
.banner[hidden] { display: none; }
.banner.is-danger { background: var(--danger); color: #fff; }

/* ---------------------------------------------------------------- panels */
.panel {
  position: fixed;
  top: var(--space-3);
  z-index: 10;
  width: 220px;
  max-height: calc(100dvh - 2 * var(--space-3));
  overflow-y: auto;
  padding: var(--space-3);
  font-size: 12.5px;
  background: color-mix(in srgb, var(--surface) 90%, transparent);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  backdrop-filter: blur(4px);
}
.panel-left { left: var(--space-3); }
.panel-right { right: var(--space-3); width: 250px; }

.panel section + section {
  margin-top: var(--space-3);
  padding-top: var(--space-3);
  border-top: 1px solid var(--border);
}
.panel h2 {
  margin: 0 0 var(--space-2);
  font-size: 10.5px;
  font-weight: 600;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--text-muted);
}

.btn-row { display: flex; gap: var(--space-1); }
.btn {
  flex: 1;
  padding: 6px 0;
  font-size: 12px;
  color: var(--text-bright);
  background: var(--surface-raised);
  border: 1px solid var(--border-strong);
  border-radius: var(--radius-sm);
}
.btn:hover { border-color: var(--state-confirmed); }

.chk { display: flex; align-items: center; gap: 8px; padding: 2px 0; cursor: pointer; }
.chk input { accent-color: var(--state-confirmed); }

.rng { display: block; margin-bottom: var(--space-2); color: var(--text); }
.rng span { float: right; color: var(--text-muted); }
.rng input { width: 100%; margin-top: 4px; accent-color: var(--state-confirmed); }

/* ---------------------------------------------------------------- tracks */
.track-list { display: flex; flex-direction: column; gap: 2px; }
.track-row {
  display: grid;
  grid-template-columns: 10px 38px 1fr auto;
  align-items: center;
  gap: 8px;
  padding: 5px 6px;
  font-family: var(--font-mono);
  font-size: 11.5px;
  color: var(--text);
  border: 1px solid transparent;
  border-radius: var(--radius-sm);
  cursor: pointer;
}
.track-row:hover { background: var(--surface-raised); }
.track-row.is-selected { border-color: var(--state-confirmed); background: var(--surface-raised); }
.track-row .sw { width: 8px; height: 8px; border-radius: 2px; }
.track-row .meta { color: var(--text-muted); }
.track-empty { color: var(--text-muted); font-size: 12px; padding: 4px 0; }

.event-log { max-height: 130px; overflow: hidden; color: var(--text-muted); line-height: 1.55; }

/* ---------------------------------------------------------------- misc */
.toast {
  position: fixed;
  bottom: var(--space-5);
  left: 50%;
  transform: translateX(-50%);
  z-index: 11;
  padding: 8px 16px;
  font-size: 12.5px;
  color: var(--text-bright);
  background: var(--surface-raised);
  border: 1px solid var(--border-strong);
  border-radius: var(--radius-md);
}
.toast[hidden] { display: none; }

.perf {
  position: fixed;
  right: var(--space-3);
  bottom: var(--space-3);
  z-index: 10;
  font-size: 11px;
  color: var(--text-muted);
}

@media (max-width: 900px) {
  .panel-left, .panel-right { width: 180px; }
}
Step 5: web/viewer3d/app.js
The file is too long for one block, so it is in two parts. Paste Part A first, then paste Part B directly after it in the same file, with no gap or edits.
Part A (setup, environment, sensor, people, render clock):
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { PLYLoader } from "three/addons/loaders/PLYLoader.js";

import { PoiClient, wsUrl, httpUrl } from "../common/ws-client.js";
import { SceneClient } from "../common/scene-client.js";
import { validateTracksMessage, validateHealthMessage, warnOnce } from "../common/validate.js";

/*
 * poi_present 3D viewer (desktop, mouse-driven).
 *
 * Coordinate convention (contracts/anchor_bundle.md):
 *   room frame  : +X, +Y on the floor, +Z up, floor at z = 0, metres.
 *   three.js    : +Y up.
 * The ONE conversion is `roomRoot.rotation.x = -PI/2`, which maps
 * room (x, y, z) -> world (x, z, -y). EVERYTHING that is in the room
 * frame (point cloud, people, sensor, walkable grid) is a child of
 * roomRoot and uses raw room coordinates. Nothing else in this file
 * swaps axes.
 */

// ---------------------------------------------------------------- tunables

const CFG = {
  renderDelayS: 0.10,      // render this far behind "now" so we can interpolate
  extrapolateMaxS: 0.15,   // how far past the newest sample we will extrapolate
  sampleBufferMax: 180,    // ~6 s at 30 Hz
  trailSeconds: 5.0,
  trailMaxPoints: 170,
  hideAfterS: 1.0,         // not in the stream for this long -> hide
  removeAfterS: 4.0,       // ... and dispose after this long
  defaultHeightM: 1.72,
  boxLenM: 0.55,           // body footprint along the heading
  boxWidM: 0.40,
  yawMinSpeed: 0.25,       // m/s below which the box keeps its last heading
  skelMinJointConf: 0.2,
  frustumRangeM: 4.0,
};

// ---------------------------------------------------------------- tokens

const cssRoot = getComputedStyle(document.documentElement);
const cssVar = (name, fallback) => cssRoot.getPropertyValue(name).trim() || fallback;
const COLORS = {
  confirmed: new THREE.Color(cssVar("--state-confirmed", "#49e8c8")),
  coasting: new THREE.Color(cssVar("--state-coasting", "#eeb43e")),
  lost: new THREE.Color(cssVar("--state-lost", "#ff6a6a")),
  tentative: new THREE.Color(cssVar("--state-tentative", "#6b7785")),
  bg: new THREE.Color(cssVar("--bg", "#0a0e13")),
  grid: new THREE.Color(cssVar("--grid-line", "#1b232c")),
  gridMajor: new THREE.Color(cssVar("--border-strong", "#313e4c")),
  frustum: new THREE.Color(cssVar("--src-depth", "#5b9dff")),
  sensor: new THREE.Color(cssVar("--text", "#a9b6c2")),
  walk: new THREE.Color(cssVar("--state-confirmed", "#49e8c8")),
};

const $ = (id) => document.getElementById(id);

// ---------------------------------------------------------------- renderer / scene

const stage = $("stage");
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.localClippingEnabled = true; // per-material clipping (ceiling cut)
stage.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = COLORS.bg;

const camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.05, 300);
camera.position.set(6, 5, 6);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.screenSpacePanning = true;
controls.maxPolarAngle = Math.PI / 2 - 0.02; // never go below the floor
controls.minDistance = 0.3;
controls.maxDistance = 80;
controls.target.set(2, 0.9, 0);

scene.add(new THREE.HemisphereLight(0x9fb4c9, 0x0b0e12, 1.0));

// Room-frame root. See the convention note at the top of this file.
const roomRoot = new THREE.Group();
roomRoot.rotation.x = -Math.PI / 2;
scene.add(roomRoot);

roomRoot.add(new THREE.AxesHelper(0.5)); // X red, Y green, Z blue, at the room origin

// ---------------------------------------------------------------- state

const state = {
  layers: { points: true, grid: true, walk: false, sensor: true, trails: true, ellipse: true, labels: true, box: true, skel: true },
  pointSize: 0.035,
  clipZ: Infinity,
  followId: null,
  selectedId: null,
  frame: null,
  mapId: undefined,
  calib: null,
  connection: "connecting",
  zMax: 3,
  center: new THREE.Vector3(2, 0.9, 0),  // world coordinates
  radius: 5,
};

// ---------------------------------------------------------------- environment

const clipPlane = new THREE.Plane(new THREE.Vector3(0, -1, 0), 100); // keeps world y <= constant
const plyLoader = new PLYLoader();

const env = { points: null, grid: null, walk: null, key: "" };
const toast = $("toast");
function showToast(text) { toast.textContent = text; toast.hidden = false; }
function hideToast() { toast.hidden = true; }

function disposePoints() {
  if (!env.points) return;
  roomRoot.remove(env.points);
  env.points.geometry.dispose();
  env.points.material.dispose();
  env.points = null;
}

function loadPly(url) {
  return new Promise((resolve, reject) => {
    plyLoader.load(
      httpUrl(url),
      resolve,
      (ev) => {
        const mb = (ev.loaded / 1048576).toFixed(1);
        showToast(ev.lengthComputable
          ? `Loading room point cloud… ${Math.round((100 * ev.loaded) / ev.total)}% (${mb} MB)`
          : `Loading room point cloud… ${mb} MB`);
      },
      reject
    );
  });
}

function setSceneExtent(box /* in ROOM coordinates */) {
  const cx = (box.min.x + box.max.x) / 2;
  const cy = (box.min.y + box.max.y) / 2;
  const sx = box.max.x - box.min.x;
  const sy = box.max.y - box.min.y;
  state.center.set(cx, 0.9, -cy); // room -> world
  state.radius = THREE.MathUtils.clamp(0.5 * Math.hypot(sx, sy), 2, 25);
}

function rebuildGrid(box) {
  if (env.grid) { scene.remove(env.grid); env.grid.geometry.dispose(); env.grid.material.dispose(); }
  const size = Math.max(4, Math.ceil(Math.max(box.max.x - box.min.x, box.max.y - box.min.y) + 2));
  const g = new THREE.GridHelper(size, size, COLORS.gridMajor, COLORS.grid);
  g.position.set(state.center.x, -0.002, state.center.z);
  g.visible = state.layers.grid;
  scene.add(g);
  env.grid = g;
}

async function loadPointsAsset(asset) {
  try {
    const geometry = await loadPly(asset.url);
    disposePoints();
    const hasColor = geometry.hasAttribute("color");
    const mat = new THREE.PointsMaterial({
      size: state.pointSize,
      sizeAttenuation: true,
      vertexColors: hasColor,
      color: hasColor ? 0xffffff : 0x8fa0b2,
      clippingPlanes: [clipPlane],
    });
    const pts = new THREE.Points(geometry, mat);
    pts.frustumCulled = false;
    pts.visible = state.layers.points;
    roomRoot.add(pts);
    env.points = pts;

    geometry.computeBoundingBox();
    const bb = geometry.boundingBox;
    setSceneExtent(bb);
    rebuildGrid(bb);
    state.zMax = Math.max(1, Math.ceil(bb.max.z * 20) / 20);
    const clip = $("s-clip");
    clip.max = String(state.zMax);
    // Start with the ceiling cut slightly below the top of the scan so the top-down view is not
    // blocked by the ceiling; drag the slider to the far right for "off".
    const defaultCut = Math.max(1.0, state.zMax - 0.3);
    clip.value = String(defaultCut);
    applyClip(defaultCut);

    const n = geometry.getAttribute("position").count;
    $("map-info").textContent =
      `${(n / 1e6).toFixed(2)} M pts · z ${bb.min.z.toFixed(2)} … ${bb.max.z.toFixed(2)} m`;
    console.info("[viewer3d] point cloud (room frame) bbox", bb.min.toArray(), bb.max.toArray());
    if (bb.min.z < -0.6 || bb.min.z > 0.6) {
      console.warn("[viewer3d] floor does not sit near z=0 - is this PLY really in the ROOM frame?");
      $("map-info").textContent += " · CHECK FRAME";
    }
    flyToView("iso", 0);
  } catch (e) {
    console.warn("[viewer3d] failed to load point cloud:", e);
    $("map-info").textContent = "point cloud failed to load";
  } finally {
    hideToast();
  }
}

function buildWalkable(walkable) {
  if (env.walk) { roomRoot.remove(env.walk); env.walk.geometry.dispose(); env.walk.material.map.dispose(); env.walk.material.dispose(); env.walk = null; }
  if (!walkable || !walkable.grid || !walkable.grid.length) return;
  const rows = walkable.grid.length;
  const cols = walkable.grid[0].length;
  const res = walkable.resolution;
  const data = new Uint8Array(cols * rows * 4);
  const c = COLORS.walk;
  for (let r = 0; r < rows; r++) {
    for (let k = 0; k < cols; k++) {
      if (!walkable.grid[r][k]) continue;
      const i = (r * cols + k) * 4;
      data[i] = Math.round(c.r * 255); data[i + 1] = Math.round(c.g * 255); data[i + 2] = Math.round(c.b * 255); data[i + 3] = 70;
    }
  }
  const tex = new THREE.DataTexture(data, cols, rows, THREE.RGBAFormat);
  tex.magFilter = THREE.NearestFilter;
  tex.minFilter = THREE.NearestFilter;
  tex.needsUpdate = true; // row 0 = smallest y, which is what PlaneGeometry's v = 0 edge is
  const w = cols * res, h = rows * res;
  const mesh = new THREE.Mesh(
    new THREE.PlaneGeometry(w, h),
    new THREE.MeshBasicMaterial({ map: tex, transparent: true, depthWrite: false })
  );
  mesh.position.set(walkable.origin[0] + w / 2, walkable.origin[1] + h / 2, 0.004);
  mesh.visible = state.layers.walk;
  roomRoot.add(mesh);
  env.walk = mesh;
  if (!env.points) { // no cloud: frame the camera on the walkable area instead
    const box = new THREE.Box3(
      new THREE.Vector3(walkable.origin[0], walkable.origin[1], 0),
      new THREE.Vector3(walkable.origin[0] + w, walkable.origin[1] + h, 0)
    );
    setSceneExtent(box);
    rebuildGrid(box);
  }
}

// ---------------------------------------------------------------- sensor marker + frustum

const sensor = new THREE.Group(); // lives in the camera OPTICAL frame (x right, y down, z forward)
sensor.matrixAutoUpdate = false;
roomRoot.add(sensor);
const sensorBody = new THREE.Mesh(
  new THREE.BoxGeometry(0.09, 0.05, 0.05),
  new THREE.MeshBasicMaterial({ color: COLORS.sensor })
);
sensor.add(sensorBody);
const frustumLines = new THREE.LineSegments(
  new THREE.BufferGeometry(),
  new THREE.LineBasicMaterial({ color: COLORS.frustum, transparent: true, opacity: 0.55 })
);
sensor.add(frustumLines);
sensor.visible = false;
let sensorKnown = false;

function applySensorPose(doc) {
  const R = doc.camera && doc.camera.R;
  const t = doc.camera && doc.camera.t;
  if (!R || !t || R.length !== 3 || t.length !== 3) { sensorKnown = false; sensor.visible = false; return; }
  // room = R @ cam + t
  sensor.matrix.set(
    R[0][0], R[0][1], R[0][2], t[0],
    R[1][0], R[1][1], R[1][2], t[1],
    R[2][0], R[2][1], R[2][2], t[2],
    0, 0, 0, 1
  );
  sensor.matrixWorldNeedsUpdate = true;

  const fov = doc.fov_deg || [60, 45];
  const th = Math.tan((fov[0] * Math.PI) / 360) * CFG.frustumRangeM;
  const tv = Math.tan((fov[1] * Math.PI) / 360) * CFG.frustumRangeM;
  const z = CFG.frustumRangeM;
  const o = new THREE.Vector3(0, 0, 0);
  const c = [
    new THREE.Vector3(th, tv, z), new THREE.Vector3(-th, tv, z),
    new THREE.Vector3(-th, -tv, z), new THREE.Vector3(th, -tv, z),
  ];
  const pts = [];
  for (let i = 0; i < 4; i++) { pts.push(o, c[i]); pts.push(c[i], c[(i + 1) % 4]); }
  frustumLines.geometry.dispose();
  frustumLines.geometry = new THREE.BufferGeometry().setFromPoints(pts);
  sensorKnown = true;
  sensor.visible = state.layers.sensor;
  sensor.updateMatrixWorld(true);
}

// ---------------------------------------------------------------- people

const BOX_GEO = new THREE.BoxGeometry(1, 1, 1);
const CIRCLE_GEO = new THREE.CircleGeometry(1, 40);
const RING_GEO = (() => {
  const pts = [];
  for (let i = 0; i < 48; i++) {
    const a = (i / 48) * Math.PI * 2;
    pts.push(new THREE.Vector3(Math.cos(a), Math.sin(a), 0));
  }
  return new THREE.BufferGeometry().setFromPoints(pts);
})();

// COCO-17 order (poi_perception/contracts.py).
const COCO_BONES = [
  [0, 1], [0, 2], [1, 3], [2, 4],
  [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
  [5, 11], [6, 12], [11, 12],
  [11, 13], [13, 15], [12, 14], [14, 16],
];
const N_JOINTS = 17;

function makeLabel(text) {
  const canvas = document.createElement("canvas");
  canvas.width = 192; canvas.height = 72;
  const g = canvas.getContext("2d");
  g.fillStyle = "rgba(10,14,19,0.78)";
  g.beginPath();
  g.roundRect(4, 8, 184, 56, 12);
  g.fill();
  g.font = "600 36px 'IBM Plex Mono', ui-monospace, monospace";
  g.fillStyle = "#e9eef3";
  g.textAlign = "center";
  g.textBaseline = "middle";
  g.fillText(text, 96, 37);
  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, depthTest: false, transparent: true }));
  sprite.scale.set(0.5, 0.1875, 1);
  sprite.renderOrder = 20;
  return sprite;
}

const lerp = (a, b, f) => a + (b - a) * f;
function lerpAngle(a, b, f) {
  let d = ((b - a + Math.PI) % (2 * Math.PI)) - Math.PI;
  if (d < -Math.PI) d += 2 * Math.PI;
  return a + d * f;
}

class TrackView {
  constructor(id) {
    this.id = id;
    this.samples = [];
    this.lastSeen = performance.now();
    this.builtH = -1;
    this.yaw = 0;
    this.speed = 0;
    this.last = null; // last interpolated sample (for the HUD list / follow)

    this.group = new THREE.Group(); // origin = ground point, room frame
    roomRoot.add(this.group);

    this.body = new THREE.Group();
    this.group.add(this.body);

    this.solidMat = new THREE.LineBasicMaterial({ color: COLORS.tentative, transparent: true });
    this.dashMat = new THREE.LineDashedMaterial({ color: COLORS.tentative, transparent: true, dashSize: 0.07, gapSize: 0.05 });
    this.edges = new THREE.LineSegments(new THREE.BufferGeometry(), this.solidMat);
    this.body.add(this.edges);

    this.fillMat = new THREE.MeshBasicMaterial({ color: COLORS.tentative, transparent: true, opacity: 0.1, depthWrite: false });
    this.fill = new THREE.Mesh(BOX_GEO, this.fillMat);
    this.fill.userData.trackId = id;
    this.body.add(this.fill);

    this.label = makeLabel(`#${id}`);
    this.group.add(this.label);

    this.ellipse = new THREE.Group();
    this.ellipse.position.z = 0.01;
    this.ellipseFillMat = new THREE.MeshBasicMaterial({ color: COLORS.tentative, transparent: true, opacity: 0.16, depthWrite: false, side: THREE.DoubleSide });
    this.ellipseLineMat = new THREE.LineBasicMaterial({ color: COLORS.tentative, transparent: true, opacity: 0.9 });
    this.ellipse.add(new THREE.Mesh(CIRCLE_GEO, this.ellipseFillMat));
    this.ellipse.add(new THREE.LineLoop(RING_GEO, this.ellipseLineMat));
    this.group.add(this.ellipse);

    // Trail: room-frame positions, so it is a child of roomRoot, not of the moving group.
    this.trailPos = new Float32Array(CFG.trailMaxPoints * 3);
    this.trailCol = new Float32Array(CFG.trailMaxPoints * 3);
    this.trailGeom = new THREE.BufferGeometry();
    this.trailGeom.setAttribute("position", new THREE.BufferAttribute(this.trailPos, 3));
    this.trailGeom.setAttribute("color", new THREE.BufferAttribute(this.trailCol, 3));
    this.trailGeom.setDrawRange(0, 0);
    this.trailMat = new THREE.LineBasicMaterial({ vertexColors: true });
    this.trail = new THREE.Line(this.trailGeom, this.trailMat);
    this.trail.frustumCulled = false;
    roomRoot.add(this.trail);

    // Skeleton (only drawn when the backend sends `joints`; see Part B of the guide).
    this.skel = new THREE.Group();
    this.bonePos = new Float32Array(COCO_BONES.length * 2 * 3);
    this.boneGeom = new THREE.BufferGeometry();
    this.boneGeom.setAttribute("position", new THREE.BufferAttribute(this.bonePos, 3));
    this.boneMat = new THREE.LineBasicMaterial({ color: COLORS.tentative });
    this.bones = new THREE.LineSegments(this.boneGeom, this.boneMat);
    this.bones.frustumCulled = false;
    this.jointPos = new Float32Array(N_JOINTS * 3);
    this.jointGeom = new THREE.BufferGeometry();
    this.jointGeom.setAttribute("position", new THREE.BufferAttribute(this.jointPos, 3));
    this.jointMat = new THREE.PointsMaterial({ color: COLORS.tentative, size: 7, sizeAttenuation: false, depthTest: false });
    this.jointPts = new THREE.Points(this.jointGeom, this.jointMat);
    this.jointPts.frustumCulled = false;
    this.jointPts.renderOrder = 15;
    this.skel.add(this.bones, this.jointPts);
    this.skel.visible = false;
    roomRoot.add(this.skel);

    this.rebuildBody(CFG.defaultHeightM);
  }

  rebuildBody(h) {
    this.builtH = h;
    const g = new THREE.BoxGeometry(CFG.boxLenM, CFG.boxWidM, h);
    g.translate(0, 0, h / 2);
    const e = new THREE.EdgesGeometry(g);
    g.dispose();
    this.edges.geometry.dispose();
    this.edges.geometry = e;
    this.edges.computeLineDistances();
    this.fill.scale.set(CFG.boxLenM, CFG.boxWidM, h);
    this.fill.position.set(0, 0, h / 2);
    this.label.position.set(0, 0, h + 0.22);
  }

  pushSample(tr, t) {
    const last = this.samples[this.samples.length - 1];
    if (last && t <= last.t) return; // out of order / duplicate
    this.samples.push({
      t, p: tr.p, v: tr.v, state: tr.state, src: tr.src, conf: tr.conf,
      height: tr.height, cov: tr.cov_xy, joints: tr.joints || null,
    });
    if (this.samples.length > CFG.sampleBufferMax) this.samples.shift();
    this.lastSeen = performance.now();
  }

  /** Position/velocity interpolated at capture-time `tt`. */
  sampleAt(tt) {
    const s = this.samples;
    if (!s.length) return null;
    const first = s[0];
    if (tt <= first.t) return first;
    const last = s[s.length - 1];
    if (tt >= last.t) {
      const dt = Math.min(tt - last.t, CFG.extrapolateMaxS);
      return { ...last, p: [last.p[0] + last.v[0] * dt, last.p[1] + last.v[1] * dt, last.p[2]], joints: last.joints };
    }
    for (let i = s.length - 1; i > 0; i--) {
      if (s[i - 1].t <= tt) {
        const a = s[i - 1], b = s[i];
        const f = (tt - a.t) / Math.max(b.t - a.t, 1e-6);
        let joints = b.joints;
        if (a.joints && b.joints) {
          joints = b.joints.map((jb, k) => {
            const ja = a.joints[k];
            if (ja && jb) return [lerp(ja[0], jb[0], f), lerp(ja[1], jb[1], f), lerp(ja[2], jb[2], f), jb[3]];
            return jb || null;
          });
        }
        return {
          ...b,
          p: [lerp(a.p[0], b.p[0], f), lerp(a.p[1], b.p[1], f), lerp(a.p[2], b.p[2], f)],
          v: [lerp(a.v[0], b.v[0], f), lerp(a.v[1], b.v[1], f), lerp(a.v[2], b.v[2], f)],
          joints,
        };
      }
    }
    return first;
  }

  update(tt, nowMs, L, isSelected) {
    const s = this.sampleAt(tt);
    if (!s) return;
    this.last = s;

    const age = nowMs - this.lastSeen;
    const visible = age < CFG.hideAfterS * 1000;
    this.group.visible = visible;
    if (!visible) { this.trail.visible = false; this.skel.visible = false; return; }

    this.group.position.set(s.p[0], s.p[1], 0);

    const color = COLORS[s.state] || COLORS.tentative;
    const coasting = s.state === "coasting" || s.src === "predicted";
    let opacity = s.conf == null ? 1 : 0.4 + 0.6 * Math.min(1, Math.max(0, s.conf));
    if (s.state === "lost") opacity = Math.min(opacity, 0.35);
    else if (coasting) opacity = Math.min(opacity, 0.7);
    else if (s.state === "tentative") opacity = Math.min(opacity, 0.6);

    // body
    const h = s.height || CFG.defaultHeightM;
    if (Math.abs(h - this.builtH) > 0.03) this.rebuildBody(h);
    this.speed = Math.hypot(s.v[0], s.v[1]);
    if (this.speed > CFG.yawMinSpeed) this.yaw = lerpAngle(this.yaw, Math.atan2(s.v[1], s.v[0]), 0.12);
    this.body.rotation.z = this.yaw;

    const hasSkel = !!(s.joints && L.skel);
    this.body.visible = L.box;
    this.edges.material = coasting || s.state === "lost" ? this.dashMat : this.solidMat;
    this.edges.material.color.copy(color);
    this.edges.material.opacity = hasSkel ? opacity * 0.5 : opacity;
    this.fillMat.color.copy(color);
    this.fillMat.opacity = (isSelected ? 0.22 : 0.09) * (hasSkel ? 0.6 : 1);

    this.label.visible = L.labels;
    this.label.material.opacity = Math.max(0.5, opacity);

    // uncertainty ellipse (cov_xy = [var_x, cov_xy, var_y])
    if (L.ellipse && s.cov) {
      const [a, b, c] = s.cov;
      const trace = a + c, det = a * c - b * b;
      const disc = Math.sqrt(Math.max(0, (trace * trace) / 4 - det));
      const l1 = Math.max(trace / 2 + disc, 1e-6), l2 = Math.max(trace / 2 - disc, 1e-6);
      const angle = Math.abs(b) < 1e-12 ? (a >= c ? 0 : Math.PI / 2) : 0.5 * Math.atan2(2 * b, a - c);
      this.ellipse.scale.set(2 * Math.sqrt(l1), 2 * Math.sqrt(l2), 1); // ~2 sigma
      this.ellipse.rotation.z = angle;
      this.ellipseFillMat.color.copy(color);
      this.ellipseLineMat.color.copy(color);
      this.ellipse.visible = true;
    } else {
      this.ellipse.visible = false;
    }

    this._updateTrail(tt, L.trails, color, s);
    this._updateSkeleton(s, L.skel, color);
  }

  _updateTrail(tt, on, color, cur) {
    if (!on) { this.trail.visible = false; return; }
    const cutoff = tt - CFG.trailSeconds;
    let n = 0;
    for (const s of this.samples) {
      if (s.t < cutoff || s.t > tt) continue;
      if (n >= CFG.trailMaxPoints - 1) break;
      const fade = 1 - (tt - s.t) / CFG.trailSeconds;
      this.trailPos[n * 3] = s.p[0]; this.trailPos[n * 3 + 1] = s.p[1]; this.trailPos[n * 3 + 2] = 0.02;
      this.trailCol[n * 3] = lerp(COLORS.bg.r, color.r, fade);
      this.trailCol[n * 3 + 1] = lerp(COLORS.bg.g, color.g, fade);
      this.trailCol[n * 3 + 2] = lerp(COLORS.bg.b, color.b, fade);
      n++;
    }
    this.trailPos[n * 3] = cur.p[0]; this.trailPos[n * 3 + 1] = cur.p[1]; this.trailPos[n * 3 + 2] = 0.02;
    this.trailCol[n * 3] = color.r; this.trailCol[n * 3 + 1] = color.g; this.trailCol[n * 3 + 2] = color.b;
    n++;
    this.trailGeom.setDrawRange(0, n);
    this.trailGeom.attributes.position.needsUpdate = true;
    this.trailGeom.attributes.color.needsUpdate = true;
    this.trail.visible = n >= 2;
  }

  _updateSkeleton(s, on, color) {
    const j = s.joints;
    if (!on || !j) { this.skel.visible = false; return; }
    const ok = (k) => j[k] && j[k][3] >= CFG.skelMinJointConf;
    let nb = 0;
    for (const [a, b] of COCO_BONES) {
      if (!ok(a) || !ok(b)) continue;
      this.bonePos.set([j[a][0], j[a][1], j[a][2], j[b][0], j[b][1], j[b][2]], nb * 6);
      nb++;
    }
    let nj = 0;
    for (let k = 0; k < N_JOINTS; k++) {
      if (!ok(k)) continue;
      this.jointPos.set([j[k][0], j[k][1], j[k][2]], nj * 3);
      nj++;
    }
    this.boneGeom.setDrawRange(0, nb * 2);
    this.jointGeom.setDrawRange(0, nj);
    this.boneGeom.attributes.position.needsUpdate = true;
    this.jointGeom.attributes.position.needsUpdate = true;
    this.boneMat.color.copy(color);
    this.jointMat.color.copy(color);
    this.skel.visible = nb > 0 || nj > 0;
  }

  dispose() {
    roomRoot.remove(this.group, this.trail, this.skel);
    this.edges.geometry.dispose();
    this.solidMat.dispose(); this.dashMat.dispose(); this.fillMat.dispose();
    this.label.material.map.dispose(); this.label.material.dispose();
    this.ellipseFillMat.dispose(); this.ellipseLineMat.dispose();
    this.trailGeom.dispose(); this.trailMat.dispose();
    this.boneGeom.dispose(); this.boneMat.dispose();
    this.jointGeom.dispose(); this.jointMat.dispose();
  }
}

const tracks = new Map(); // id -> TrackView

function clearTracks() {
  for (const t of tracks.values()) t.dispose();
  tracks.clear();
  state.selectedId = null;
  state.followId = null;
}

// ---------------------------------------------------------------- render clock
// t_capture can be device time rather than epoch time (health.clock === "device"),
// and the browser's clock is never exactly the server's. So we do not trust absolute
// clocks: we measure (arrival time - t_capture) per message, take the smallest value
// seen recently (= the least-delayed message), and render a fixed CFG.renderDelayS
// behind that. Works with any constant clock offset.

const clock = { win: [], dSmooth: null };

function noteArrival(tCapture) {
  const now = performance.now();
  const d = now - tCapture * 1000;
  clock.win.push([now, d]);
  while (clock.win.length && now - clock.win[0][0] > 6000) clock.win.shift();
  let m = Infinity;
  for (const [, x] of clock.win) if (x < m) m = x;
  if (clock.dSmooth == null || Math.abs(m - clock.dSmooth) > 500) clock.dSmooth = m;
  else clock.dSmooth += Math.max(-1, Math.min(1, m - clock.dSmooth)); // slew <= 1 ms per message
}

function renderCaptureTime() {
  if (clock.dSmooth == null) return null;
  return (performance.now() - clock.dSmooth) / 1000 - CFG.renderDelayS;
}

function resetClock() { clock.win = []; clock.dSmooth = null; }
Part B (paste directly after Part A: connection, views, controls, track list, render loop):
// ---------------------------------------------------------------- connection

const client = new PoiClient({ url: wsUrl() });
const sceneClient = new SceneClient();

let latest = { msg: null, at: 0 };
let sceneInflight = false;
let sceneDoc = null;

async function refreshScene() {
  if (sceneInflight) return;
  sceneInflight = true;
  try { await sceneClient.fetch(); } catch (e) { console.warn("[viewer3d] /api/scene failed:", e); }
  finally { sceneInflight = false; }
}

sceneClient.addEventListener("scene", (e) => {
  sceneDoc = e.detail;
  applySensorPose(sceneDoc);
  const assets = sceneDoc.assets || [];
  const pointsAsset = assets.find((a) => a.kind === "points");
  const key = JSON.stringify([pointsAsset && pointsAsset.url, sceneDoc.map_id]);
  if (key !== env.key) {
    env.key = key;
    if (pointsAsset) loadPointsAsset(pointsAsset);
    else { disposePoints(); $("map-info").textContent = "no point cloud in this scene"; }
  }
  buildWalkable(sceneDoc.walkable);
  if (sceneDoc.map_id) setMapBadge(sceneDoc.map_id);
});

function setMapBadge(id) {
  const el = $("map-badge");
  el.textContent = id ? id : "no map";
}

client.addEventListener("status", (e) => {
  state.connection = e.detail.state;
  $("conn-dot").className = "dot is-" + e.detail.state;
  $("conn-label").textContent = { connecting: "connecting", open: "live", closed: "reconnecting…", stalled: "stream stalled" }[e.detail.state] || e.detail.state;
  updateBanner();
});

client.addEventListener("tracks", (e) => {
  const msg = e.detail;
  warnOnce(validateTracksMessage(msg));

  if (msg.frame !== state.frame || msg.map_id !== state.mapId) {
    const firstMsg = state.frame === null;
    state.frame = msg.frame;
    state.mapId = msg.map_id;
    if (!firstMsg) { clearTracks(); resetClock(); }
    const fb = $("frame-badge");
    fb.textContent = msg.frame;
    fb.className = "badge " + (msg.frame === "room" ? "is-room" : "is-local");
    setMapBadge(msg.map_id);
    refreshScene();
    updateBanner();
  }

  noteArrival(msg.t_capture);
  latest = { msg, at: performance.now() };
  for (const tr of msg.tracks) {
    let v = tracks.get(tr.id);
    if (!v) { v = new TrackView(tr.id); tracks.set(tr.id, v); }
    v.pushSample(tr, msg.t_capture);
  }
});

client.addEventListener("health", (e) => {
  const m = e.detail;
  warnOnce(validateHealthMessage(m));
  state.calib = m.calib;
  const lat = m.latency_ms == null || m.clock === "device" ? "latency n/a" : `${Math.round(m.latency_ms)} ms`;
  const temp = m.temp_c == null ? "" : ` · ${Math.round(m.temp_c)}°C`;
  $("hud-stats").textContent = `${m.fps.toFixed(1)} fps · ${lat}${temp}`;
  updateBanner();
});

client.addEventListener("event", (e) => {
  const m = e.detail;
  const log = $("event-log");
  const d = new Date();
  const hh = String(d.getHours()).padStart(2, "0"), mm = String(d.getMinutes()).padStart(2, "0"), ss = String(d.getSeconds()).padStart(2, "0");
  const row = document.createElement("div");
  row.textContent = `${hh}:${mm}:${ss}  ${m.event.replace("track_", "")} #${m.id}`;
  log.prepend(row);
  while (log.children.length > 8) log.lastChild.remove();
});

function updateBanner() {
  const b = $("banner");
  let text = null, danger = false;
  if (state.frame === "local") text = "LOCAL FRAME — positions are not registered to a room map";
  else if (state.frame === "room" && state.calib && state.calib !== "ok") { text = `CALIBRATION ${String(state.calib).toUpperCase()} — positions may be wrong`; danger = true; }
  b.hidden = !text;
  if (text) { b.textContent = text; b.className = "banner" + (danger ? " is-danger" : ""); }
}

client.connect();
refreshScene();

// ---------------------------------------------------------------- camera views

let fly = null;
controls.addEventListener("start", () => { fly = null; });

function viewPose(name) {
  const c = state.center, r = state.radius;
  if (name === "top") {
    return { pos: new THREE.Vector3(c.x, Math.max(6, r * 2.4), c.z + 0.001), target: new THREE.Vector3(c.x, 0, c.z) };
  }
  if (name === "sensor" && sensorKnown) {
    sensor.updateMatrixWorld(true);
    const p = new THREE.Vector3().setFromMatrixPosition(sensor.matrixWorld);
    const f = new THREE.Vector3(0, 0, 1).transformDirection(sensor.matrixWorld);
    return { pos: p, target: p.clone().addScaledVector(f, 2.5) };
  }
  const dir = new THREE.Vector3(0.8, 0.85, 1.0).normalize();
  return { pos: c.clone().addScaledVector(dir, r * 2.6), target: c.clone() };
}

function flyToView(name, ms = 700) {
  const v = viewPose(name);
  if (ms <= 0) { camera.position.copy(v.pos); controls.target.copy(v.target); controls.update(); return; }
  fly = { t0: performance.now(), ms, p0: camera.position.clone(), g0: controls.target.clone(), p1: v.pos, g1: v.target };
}

function stepFly(now) {
  if (!fly) return;
  const k = Math.min(1, (now - fly.t0) / fly.ms);
  const e = k < 0.5 ? 2 * k * k : 1 - Math.pow(-2 * k + 2, 2) / 2;
  camera.position.lerpVectors(fly.p0, fly.p1, e);
  controls.target.lerpVectors(fly.g0, fly.g1, e);
  if (k >= 1) fly = null;
}

// ---------------------------------------------------------------- UI wiring

document.querySelectorAll("[data-view]").forEach((b) => b.addEventListener("click", () => { state.followId = null; flyToView(b.dataset.view); }));

function bindLayer(id, key, apply) {
  const el = $(id);
  el.addEventListener("change", () => { state.layers[key] = el.checked; if (apply) apply(el.checked); });
}
bindLayer("l-points", "points", (v) => { if (env.points) env.points.visible = v; });
bindLayer("l-grid", "grid", (v) => { if (env.grid) env.grid.visible = v; });
bindLayer("l-walk", "walk", (v) => { if (env.walk) env.walk.visible = v; });
bindLayer("l-sensor", "sensor", (v) => { sensor.visible = v && sensorKnown; });
bindLayer("l-trails", "trails");
bindLayer("l-ellipse", "ellipse");
bindLayer("l-labels", "labels");
bindLayer("l-box", "box");
bindLayer("l-skel", "skel");

function applyClip(z) {
  state.clipZ = z;
  clipPlane.constant = z >= state.zMax - 1e-6 ? 1000 : z; // at the max = no cut
  $("v-clip").textContent = z >= state.zMax - 1e-6 ? "off" : `${z.toFixed(2)} m`;
}
$("s-clip").addEventListener("input", (e) => applyClip(parseFloat(e.target.value)));
$("s-psize").addEventListener("input", (e) => {
  state.pointSize = parseFloat(e.target.value);
  $("v-psize").textContent = state.pointSize.toFixed(3);
  if (env.points) env.points.material.size = state.pointSize;
});

function select(id, follow) {
  state.selectedId = id;
  state.followId = follow ? id : null;
}

$("track-list").addEventListener("click", (e) => {
  const row = e.target.closest("[data-id]");
  if (!row) return;
  const id = Number(row.dataset.id);
  if (state.selectedId === id && state.followId === id) select(null, false);
  else select(id, true);
});

const raycaster = new THREE.Raycaster();
renderer.domElement.addEventListener("dblclick", (e) => {
  const r = renderer.domElement.getBoundingClientRect();
  const ndc = new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
  raycaster.setFromCamera(ndc, camera);
  const pickables = [];
  for (const t of tracks.values()) if (t.group.visible) pickables.push(t.fill);
  const hit = raycaster.intersectObjects(pickables, false)[0];
  if (hit) select(hit.object.userData.trackId, true);
});

window.addEventListener("keydown", (e) => {
  if (e.target && e.target.tagName === "INPUT") return;
  const k = e.key.toLowerCase();
  if (k === "1") flyToView("top");
  else if (k === "2") flyToView("iso");
  else if (k === "3") flyToView("sensor");
  else if (k === "t") { $("l-trails").click(); }
  else if (k === "l") { $("l-labels").click(); }
  else if (k === "escape") select(null, false);
});

window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

// ---------------------------------------------------------------- track list (4 Hz)

const SW = { confirmed: "--state-confirmed", coasting: "--state-coasting", lost: "--state-lost", tentative: "--state-tentative" };

// Rows are created once per track and then updated in place. (Rebuilding innerHTML every
// tick would detach the row between mouse-down and mouse-up and swallow clicks.)
const rowEls = new Map(); // id -> { el, sw, id, state, info }

function renderTrackList() {
  const listEl = $("track-list");
  const seen = new Set();
  let n = 0;
  for (const t of [...tracks.values()].sort((a, b) => a.id - b.id)) {
    if (!t.group.visible || !t.last) continue;
    n++;
    seen.add(t.id);
    const s = t.last;
    let r = rowEls.get(t.id);
    if (!r) {
      const el = document.createElement("div");
      el.className = "track-row";
      el.dataset.id = String(t.id);
      el.innerHTML = '<span class="sw"></span><span class="rid"></span><span class="meta rstate"></span><span class="meta rinfo"></span>';
      r = { el, sw: el.children[0], id: el.children[1], state: el.children[2], info: el.children[3] };
      r.id.textContent = `#${t.id}`;
      rowEls.set(t.id, r);
      listEl.appendChild(el);
    }
    r.sw.style.background = `var(${SW[s.state] || SW.tentative})`;
    r.state.textContent = s.state + (s.joints ? " · 3D" : "");
    r.info.textContent = `${s.height ? s.height.toFixed(2) + "m" : "--"} · ${t.speed.toFixed(1)}m/s`;
    r.el.classList.toggle("is-selected", state.selectedId === t.id);
  }
  for (const [id, r] of rowEls) {
    if (!seen.has(id)) { r.el.remove(); rowEls.delete(id); }
  }
  const empty = listEl.querySelector(".track-empty");
  if (n === 0 && !empty) listEl.insertAdjacentHTML("beforeend", '<div class="track-empty">nobody in view</div>');
  else if (n > 0 && empty) empty.remove();
  $("people-count").textContent = String(n);
}
setInterval(renderTrackList, 250);

// ---------------------------------------------------------------- loop

let frames = 0, fpsT0 = performance.now();
const _follow = new THREE.Vector3();

function frame() {
  const now = performance.now();
  stepFly(now);

  const tt = renderCaptureTime();
  if (tt != null) {
    for (const [id, t] of tracks) {
      t.update(tt, now, state.layers, state.selectedId === id);
      if (now - t.lastSeen > CFG.removeAfterS * 1000) {
        t.dispose();
        tracks.delete(id);
        if (state.selectedId === id) select(null, false);
      }
    }
  }

  if (state.followId != null && !fly) {
    const t = tracks.get(state.followId);
    if (t && t.group.visible && t.last) {
      _follow.set(t.last.p[0], 0.9, -t.last.p[1]); // room -> world
      _follow.sub(controls.target).multiplyScalar(0.15);
      controls.target.add(_follow);
      camera.position.add(_follow);
    }
  }

  controls.update();
  renderer.render(scene, camera);

  frames++;
  if (now - fpsT0 > 500) {
    $("perf").textContent = `${((frames * 1000) / (now - fpsT0)).toFixed(0)} fps`;
    frames = 0; fpsT0 = now;
  }
}
renderer.setAnimationLoop(frame);

// Handy for the browser console / automated tests.
window.__sts3d = { THREE, scene, roomRoot, camera, controls, tracks, state, env, flyToView, CFG, client };
Check that the two parts joined correctly: node --check web/viewer3d/app.js should print nothing (if Node is installed), and the file should be 925 lines (wc -l). If the browser shows a blank page, open the developer console (F12): a 404 on OrbitControls.js means Step 2 was skipped or went to the wrong path.
Step 6: link the page from web/index.html
Optional but handy. In web/index.html, find the VR link line:
    <a class="link" href="vr/"><span>WebXR viewer</span><span class="go mono">VR →</span></a>
and add this line directly above it:
    <a class="link" href="viewer3d/"><span>3D viewer</span><span class="go mono">3D →</span></a>
The picker at http://192.168.1.101:8765/ then lists Dashboard, 3D viewer and WebXR viewer.
Step 7: run it and check it
Stop the running run_present.py (Ctrl+C) and start it again with your usual command, so Step 1 takes effect:
cd /home/drdo/spatial_tracking_system_v1.0
python3 scripts/run_present.py \
  --source live \
  --m3-config data/runtime/site_001/cam0/m3.json \
  --m4-config data/runtime/site_001/cam0/m4.json \
  --present-config data/runtime/site_001/cam0/present.json \
  --m4-source realsense
Open http://192.168.1.101:8765/viewer3d/ from your desktop browser. Keep the trailing slash: without it the page's relative file paths break.
What you should see
Check
Expected
Top bar
green dot and live, a green room badge, the map badge 7814e6723421, fps from the Jetson
Banner
none. A yellow LOCAL FRAME or red CALIBRATION … banner means positions are not trustworthy
Room
the scanned room appears upright (floor on the grid, ceiling above). A first load can download up to about 22 MB; a progress message shows
Left panel, map line
a point count and z 0.0 … ~2.6 m. CHECK FRAME appended means the floor is not near z = 0
Ceiling cut
starts about 0.3 m below the top of the scan. Drag it right to off, left to see into the room from above
People
a box on the floor point with an ID label, a ground ellipse for uncertainty and a 5 s trail. Dashed outline = coasting or predicted
Mouse
drag orbits, right-drag pans, wheel zooms, double-click a person follows them, Esc releases
Keys
1 top, 2 iso, 3 sensor view, T trails, L labels
Walk to a spot you can measure (for example 2 m from a wall) and check that the box stands there in the 3D room. This is the real test of the calibration; the viewer can only show what the tracker reports.
If something is wrong
Symptom
Likely cause and fix
Blank page, console shows 404 for OrbitControls.js
Step 2 file missing or in the wrong folder
Blank page, other 404s
missing trailing slash, or a file name typo in web/viewer3d/
401 Unauthorized
present.token is set; open …/viewer3d/?token=YOURTOKEN
No room, grid only
run curl -s http://192.168.1.101:8765/api/scene | python3 -m json.tool | head -30; assets should list /map/viewer/points.ply. If empty, present.json has no map_bundle_dir
Old room after re-calibrating
the server caches /map/ files for a day; hard refresh with Ctrl+Shift+R
Room lies on its side
the file is not in the room frame (CHECK FRAME shown); tell me the console bbox line
Walls hide everyone
lower the ceiling cut, switch Person boxes off to see only trails, or use the top view
stream stalled
the tracker is publishing nothing (nobody in view still sends empty frames); check the run_present.py terminal
Jittery boxes
normal at low fps from the camera; the viewer renders 0.1 s behind live to smooth them
The person boxes are derived from the tracker's floor point and estimated height, drawn about 0.55 m by 0.40 m. They are not measured 3D bounding boxes; the stream does not contain any. Real skeletons need Part B.
Part B (optional): real 3D skeleton joints
Do this only after Steps 1 to 7 work live. It makes M4 lift M3's 2D keypoints into room metres using the depth image, sends them as an optional joints field, and the viewer you already installed draws them (bones, joint dots, a 3D tag in the people list). Everything is off by default, so nothing changes until you switch it on in B8.
What to expect: good torso, shoulders and hips; shakier wrists and ankles, because a single RealSense often reads the wall behind a limb. A joint that fails the checks is sent as null and its bones are simply not drawn.
Design choice worth knowing: the joints travel as a plain attribute on the track record, deliberately not a dataclass field, so your JSONL track logs and the golden test files are unchanged. Tested here: the full unit suite (174 tests, 7 of them new) and 69 integration tests pass, and lifted joints matched known positions within 3 cm in a synthetic scene. Not tested on a real camera.
B1: poi_localization/config.py (3 edits)
Edit 1: directly above the @dataclass line that sits above class GateConfig, add this whole block (it starts with its own @dataclass and ends with two blank lines):
@dataclass
class JointsConfig:
    """Optional 3D skeleton for the viewers: lift M3's 2D COCO-17 keypoints into the
    floor/room frame using the aligned depth image. Off by default; costs a few
    patch-median lookups per detection per frame."""

    enabled: bool = False
    keypoint_conf_thresh: float = 0.3
    patch_radius_px: int = 3
    # A joint whose depth differs from the torso's by more than this is almost certainly
    # sitting on the wall behind the person (silhouette-edge depth) -> dropped (null).
    max_torso_depth_dev_m: float = 0.9
    min_z_m: float = -0.15
    max_z_m: float = 2.4
    min_valid_joints: int = 6


So the result reads @dataclass / class JointsConfig … / two blank lines / @dataclass / class GateConfig:.
Edit 2: in class M4Config, add one line after the height: field:
    height: HeightConfig = field(default_factory=HeightConfig)
    joints: JointsConfig = field(default_factory=JointsConfig)
    gates: GateConfig = field(default_factory=GateConfig)
Edit 3: in M4Config.load, add one line after the height= argument:
            height=HeightConfig(**raw.get("height", {})),
            joints=JointsConfig(**raw.get("joints", {})),
            gates=GateConfig(**raw.get("gates", {})),
Older m4.json files without a joints key still load.
B2: new file poi_localization/measurement/joints.py
"""
Optional 3D skeleton: lift M3's 2D COCO-17 keypoints to the output frame.

Same recipe the height estimator already uses for the head keypoint -- median of a small
depth patch around the keypoint, deproject with the intrinsics, transform with the same
FloorFrameTransform as every other measurement -- applied to all 17 keypoints, plus the
plausibility gates a single RealSense needs:

  * confidence gate        : the 2D keypoint must be trusted by M3,
  * depth validity         : the patch must contain valid (non-zero) depth,
  * torso-depth consistency: wrists/ankles at a silhouette edge often read the wall
                             behind the person; a joint more than `max_torso_depth_dev_m`
                             from the torso depth is dropped,
  * vertical sanity        : z must be between `min_z_m` and `max_z_m`.

A rejected joint is None (the viewer simply skips the bones touching it). If fewer than
`min_valid_joints` survive, the whole skeleton is None for this detection.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from poi_localization import geometry
from poi_localization.config import JointsConfig
from poi_localization.frames.types import FloorFrameTransform
from poi_perception.contracts import COCO_KEYPOINT_NAMES, Intrinsics


def _patch_median_depth_m(depth_raw: np.ndarray, u: float, v: float, intr: Intrinsics, radius_px: int) -> Optional[float]:
    h, w = depth_raw.shape[:2]
    ui, vi = int(round(u)), int(round(v))
    u0, u1 = max(0, ui - radius_px), min(w, ui + radius_px + 1)
    v0, v1 = max(0, vi - radius_px), min(h, vi + radius_px + 1)
    if u0 >= u1 or v0 >= v1:
        return None
    patch = depth_raw[v0:v1, u0:u1]
    valid = patch[patch > 0]
    if valid.size == 0:
        return None
    return float(np.median(valid)) * intr.depth_scale


def lift_joints(
    depth_raw: Optional[np.ndarray],
    keypoints: Dict[str, Tuple[float, float, float]],
    intr: Intrinsics,
    floor_transform: FloorFrameTransform,
    cfg: JointsConfig,
    torso_depth_m: Optional[float] = None,
) -> Optional[List[Optional[List[float]]]]:
    """Returns 17 entries in COCO order, each [x, y, z, conf] (output frame, metres) or None;
    or None when too few joints survive."""
    if depth_raw is None or not keypoints:
        return None
    out: List[Optional[List[float]]] = []
    n_ok = 0
    for name in COCO_KEYPOINT_NAMES:
        kp = keypoints.get(name)
        if kp is None or kp[2] < cfg.keypoint_conf_thresh:
            out.append(None)
            continue
        u, v, conf = kp
        d = _patch_median_depth_m(depth_raw, u, v, intr, cfg.patch_radius_px)
        if d is None or d <= 0.0:
            out.append(None)
            continue
        if torso_depth_m is not None and abs(d - torso_depth_m) > cfg.max_torso_depth_dev_m:
            out.append(None)
            continue
        p = floor_transform.apply_point(geometry.deproject_pixel(u, v, d, intr))
        z = float(p[2])
        if not (cfg.min_z_m <= z <= cfg.max_z_m):
            out.append(None)
            continue
        out.append([float(p[0]), float(p[1]), z, float(conf)])
        n_ok += 1
    if n_ok < cfg.min_valid_joints:
        return None
    return out
B3: poi_localization/tracking/track_manager.py (6 small edits)
The joints ride along with the matched detection, the same way the bounding box already does.
1. In class DetectionMeasurements, after height_sample_m: Optional[float] add:
    # Optional 3D skeleton (17 x [x, y, z, conf] or None), already in the output frame.
    joints_room: Optional[list] = None
2. In class WorldTrackOutput, after bbox_px: Optional[Tuple[float, float, float, float]] add:
    joints_room: Optional[list] = None
3. In class _Track.__init__, after self.last_det_conf: float = 0.0 add:
        self.last_joints: Optional[list] = None
4. In _filter_walkable, in the return DetectionMeasurements( call at the end, add a last argument after height_sample_m=det.height_sample_m,:
            joints_room=det.joints_room,
5. In update, there are two places that say track.last_det_conf = det.det_conf. After each of them add a line with the same indentation as the line above it:
track.last_joints = det.joints_room
(The first is inside the loop for matched tracks, 20 spaces of indent; the second is in the new-track branch, 12 spaces.)
6. In _to_output, in the WorldTrackOutput( call, after bbox_px=track.last_bbox, add:
            # only the frame this track was actually measured in: never show a stale pose while coasting
            joints_room=track.last_joints if track.time_since_update_s < 1e-9 else None,
B4: poi_localization/runtime/m4_pipeline.py (4 edits)
1. Add an import after the depth_measurement import line:
from poi_localization.measurement.joints import lift_joints
2. In _build_detection_measurements, between the height_sample = estimate_instantaneous_height(...) call and return DetectionMeasurements(, add:
        joints_room = None
        if self.cfg.joints.enabled:
            joints_room = lift_joints(
                frame.depth, det.keypoints, frame.intr, self.floor_transform, self.cfg.joints,
                torso_depth_m=fallback_depth_m,
            )
3. In that same return DetectionMeasurements( call, after height_sample_m=height_sample, add:
            joints_room=joints_room,
4. In process, directly after the records = [self._to_phase_record(o, frame.t) for o in outputs] line, add:
        # 3D skeleton rides along as a plain attribute, deliberately NOT a dataclass field, so the
        # JSONL track logs and the golden files stay byte-identical. poi_present's adapter reads it.
        for rec, o in zip(records, outputs):
            if o.joints_room is not None:
                rec.joints = o.joints_room
B5: poi_present/schema.py (class TrackWire)
After the bbox: field add the new field, then replace the single line def to_dict(self) -> Dict[str, Any]: with the three pieces shown. The existing dictionary body stays exactly as it is, under the renamed _base_dict.
Before:
    age_s: float
    bbox: Optional[Tuple[float, float, float, float]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
After:
    age_s: float
    bbox: Optional[Tuple[float, float, float, float]]
    # Optional 3D skeleton: 17 COCO entries, each [x, y, z, conf] in the frame named by the message,
    # or None for a joint that failed the depth gates. Absent from the JSON unless M4 produced one.
    joints: Optional[List[Optional[List[float]]]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = self._base_dict()
        if self.joints is not None:
            d["joints"] = [None if j is None else [_round(j[0], 3), _round(j[1], 3), _round(j[2], 3), _round(j[3], 2)]
                           for j in self.joints]
        return d

    def _base_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
List and Optional are already imported in that file. Messages without joints stay byte-identical, so the golden example tests keep passing.
B6: poi_present/adapter.py (function record_to_wire, 2 edits)
1. After the line bbox = d.get("bbox")  # absent entirely on Phase A -> None, correct add:
    joints = d.get("joints")  # optional 3D skeleton, only present when M4's joints.enabled is on
2. In the return TrackWire( call, after the bbox=None if bbox is None else tuple(bbox), line add:
        joints=None if joints is None else [None if j is None else list(j) for j in joints],
B7: contracts/schemas/poi_tracks.schema.json (recommended)
Inside properties.tracks.items.properties, after the bbox property (add a comma after its closing brace), add:
"joints": {
  "type": ["array", "null"],
  "minItems": 17,
  "maxItems": 17,
  "description": "optional 3D skeleton, COCO-17 order; each joint [x,y,z,conf] in the message frame, or null",
  "items": {
    "type": ["array", "null"],
    "minItems": 4,
    "maxItems": 4,
    "items": { "type": "number" }
  }
}
B8: switch it on
Quickest, for a test run: open data/runtime/site_001/cam0/m4.json and add this as a new top-level key (next to "height"):
  "joints": { "enabled": true },
Note that python -m sts configs, check and accept regenerate m4.json and would erase that edit. To make it permanent, set it in configs/site.json instead:
"localization": { "phase": "B", "overrides": { "joints": { "enabled": true } } },
and then run python -m sts configs --phase B.
Restart run_present.py. In the viewer the people list should now show confirmed · 3D and a skeleton should appear inside the box. If it does not, check in this order:
1. The terminal shows no Python error at startup (a typo in B1 to B4 usually shows up there).
2. In the browser console, run [...__sts3d.tracks.values()][0].samples.at(-1).joints. null means the backend sent none; an array means it is a viewer-side toggle.
3. Stand closer. Keypoints need M3 confidence above 0.3 and at least 6 of 17 joints must pass the depth checks. You can loosen keypoint_conf_thresh, max_torso_depth_dev_m and min_valid_joints under joints in m4.json.
B9: add the test file (recommended)
Create tests/unit/test_joints.py, then run python -m pytest tests/unit -q. All unit tests should pass; the new file has 7.
"""
3D skeleton lifting (poi_localization.measurement.joints) and its trip to the wire.

Setup: camera 2.3 m up, pitched 25 deg down, looking along floor +X. A synthetic person stands 3.5 m
in front of it. The depth image is a wall at 6 m with the true depth painted around every joint.
Keypoints are projected from known 3D joints, so the lifted joints must come back (almost) exactly.
"""
import numpy as np
import pytest

from poi_localization import geometry
from poi_localization.config import JointsConfig, M4Config
from poi_localization.frames.types import FloorFrameTransform
from poi_localization.measurement.joints import lift_joints
from poi_perception.contracts import COCO_KEYPOINT_NAMES, Intrinsics
from poi_present.adapter import record_to_wire
from poi_present.schema import tracks_message

INTR = Intrinsics(fx=606.75, fy=606.57, cx=320.19, cy=237.06, width=640, height=480, depth_scale=0.001, baseline=0.05)
FT = FloorFrameTransform.from_height_and_yaw(2.3, 25.0, 0.0)  # pitched 25 deg down, faces floor +X


def _project(p_floor):
    """floor point -> (u, v, depth_m) for the camera above."""
    p_cam = FT.R.T @ (np.asarray(p_floor, float) - FT.t)
    z = p_cam[2]
    return INTR.fx * p_cam[0] / z + INTR.cx, INTR.fy * p_cam[1] / z + INTR.cy, z


TORSO_D = _project((3.5, 0.0, 1.2))[2]  # torso depth the pipeline would pass in


def _scene():
    # joints of a person standing at floor (3.5, 0.0), arms hanging
    truth = {n: None for n in COCO_KEYPOINT_NAMES}
    truth.update(
        nose=(3.5, 0.0, 1.62), left_shoulder=(3.5, 0.2, 1.4), right_shoulder=(3.5, -0.2, 1.4),
        left_elbow=(3.5, 0.25, 1.1), right_elbow=(3.5, -0.25, 1.1),
        left_wrist=(3.5, 0.25, 0.85), right_wrist=(3.5, -0.25, 0.85),
        left_hip=(3.5, 0.12, 0.9), right_hip=(3.5, -0.12, 0.9),
        left_knee=(3.5, 0.12, 0.5), right_knee=(3.5, -0.12, 0.5),
        left_ankle=(3.5, 0.12, 0.08), right_ankle=(3.5, -0.12, 0.08),
    )
    depth = np.full((480, 640), 6000, np.uint16)  # wall at 6 m
    kps = {}
    for n, p in truth.items():
        if p is None:
            continue
        u, v, d = _project(p)
        assert 8 < u < 632 and 8 < v < 472, f"test setup: {n} is outside the image"
        depth[int(v) - 6:int(v) + 7, int(u) - 6:int(u) + 7] = int(round(d * 1000))
        kps[n] = (u, v, 0.9)
    return truth, kps, depth


def test_lifts_every_visible_joint_back_to_its_true_position():
    truth, kps, depth = _scene()
    out = lift_joints(depth, kps, INTR, FT, JointsConfig(enabled=True), torso_depth_m=TORSO_D)
    assert out is not None and len(out) == 17
    for name, joint in zip(COCO_KEYPOINT_NAMES, out):
        if truth[name] is None:
            assert joint is None
        else:
            assert joint[:3] == pytest.approx(truth[name], abs=0.03), name
            assert joint[3] == pytest.approx(0.9)


def test_joint_on_the_wall_behind_the_person_is_dropped():
    truth, kps, depth = _scene()
    u, v, _ = _project(truth["left_wrist"])
    depth[int(v) - 6:int(v) + 7, int(u) - 6:int(u) + 7] = 6000  # silhouette-edge depth reads the wall
    out = lift_joints(depth, kps, INTR, FT, JointsConfig(enabled=True), torso_depth_m=TORSO_D)
    assert out[COCO_KEYPOINT_NAMES.index("left_wrist")] is None
    assert out[COCO_KEYPOINT_NAMES.index("right_wrist")] is not None


def test_low_confidence_and_below_floor_are_dropped_and_too_few_joints_gives_none():
    truth, kps, depth = _scene()
    kps["nose"] = (*kps["nose"][:2], 0.1)
    out = lift_joints(depth, kps, INTR, FT, JointsConfig(enabled=True), torso_depth_m=TORSO_D)
    assert out[0] is None
    only_two = {k: kps[k] for k in ("left_shoulder", "right_shoulder")}
    assert lift_joints(depth, only_two, INTR, FT, JointsConfig(enabled=True), torso_depth_m=TORSO_D) is None
    assert lift_joints(None, kps, INTR, FT, JointsConfig(enabled=True)) is None


def test_disabled_by_default_and_config_roundtrip():
    assert M4Config().joints.enabled is False
    cfg = M4Config.default()
    cfg.joints.enabled = True
    from dataclasses import asdict
    import json, tempfile, pathlib
    p = pathlib.Path(tempfile.mkdtemp()) / "m4.json"
    cfg.save(p)
    assert M4Config.load(p).joints.enabled is True
    # an m4.json written before this feature existed (no "joints" key) still loads
    raw = asdict(M4Config()); raw.pop("joints")
    p.write_text(json.dumps(raw))
    assert M4Config.load(p).joints.enabled is False


class _Rec:  # stands in for a WorldTrackPhaseB record with the joints side-channel attached
    pass


def test_joints_reach_the_wire_only_when_present():
    base = dict(id=1, state="confirmed", p=(1, 2, 0), v=(0, 0, 0), cov_xy=(0.01, 0, 0.01), height=1.7,
                conf=0.9, src="fused", age_s=1.0, bbox=(1, 2, 3, 4))
    r = _Rec(); r.__dict__.update(base)
    assert "joints" not in tracks_message(map_id=None, cam_id="c", frame="room", seq=1, t_capture=0, t_publish=0,
                                           tracks=[record_to_wire(r)])["tracks"][0]
    r.joints = [[1.0, 2.0, 1.5, 0.9]] + [None] * 16
    j = tracks_message(map_id=None, cam_id="c", frame="room", seq=1, t_capture=0, t_publish=0,
                       tracks=[record_to_wire(r)])["tracks"][0]["joints"]
    assert len(j) == 17 and j[0] == [1.0, 2.0, 1.5, 0.9] and j[1] is None


def test_schema_accepts_joints():
    jsonschema = pytest.importorskip("jsonschema")
    import json, pathlib
    schema = json.loads((pathlib.Path(__file__).resolve().parents[2] / "contracts/schemas/poi_tracks.schema.json").read_text())
    msg = tracks_message(map_id=None, cam_id="c", frame="room", seq=1, t_capture=0, t_publish=0, tracks=[])
    msg["tracks"] = [{"id": 1, "state": "confirmed", "p": [0, 0, 0], "v": [0, 0, 0], "cov_xy": [1, 0, 1], "src": "fused",
                      "age_s": 1.0, "joints": [[1, 2, 3, 0.5]] + [None] * 16}]
    jsonschema.validate(msg, schema)


def test_track_manager_carries_joints_and_drops_them_while_coasting():
    from poi_localization.measurement.types import Measurement
    from poi_localization.tracking.track_manager import DetectionMeasurements, TrackManager

    cfg = M4Config.default()
    cfg.track.tentative_confirm_hits = 2
    cfg.track.coasting_budget_s = 1.0
    tm = TrackManager(cfg)
    dt = 1.0 / 30.0
    skel = [[1.0, 0.0, 1.5, 0.9]] + [None] * 16

    def det(x, joints):
        m = Measurement(position_xy=(x, 0.0), cov_xy=(0.01, 0.0, 0.01), src="depth")
        return DetectionMeasurements(track_id_2d=1, det_conf=0.9, bbox_px=(0, 0, 60, 170), depth_measurement=m,
                                     raycast_measurement=None, height_sample_m=1.7, joints_room=joints)

    t, out = 0.0, []
    for i in range(3):
        out, _ = tm.update(t, dt, [det(1.0 + 0.01 * i, skel)])
        t += dt
    assert len(out) == 1 and out[0].joints_room == skel
    out, _ = tm.update(t, dt, [])  # missed frame -> coasting, prediction only
    assert len(out) == 1 and out[0].state == "coasting" and out[0].joints_room is None
To undo Part B completely: set "enabled": false (or remove the joints key). Nothing else needs reverting, because with joints off the new code paths never run.
