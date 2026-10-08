import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { PLYLoader } from "three/addons/loaders/PLYLoader.js";

import { PoiClient, wsUrl, httpUrl } from "../common/ws-client.js";
import { SceneClient } from "../common/scene-client.js";
import { validateTracksMessage, validateHealthMessage, warnOnce } from "../common/validate.js";

/*
 * poi_present 3D viewer (desktop, mouse-driven) -- SKELETON edition.
 *
 * Coordinate convention (contracts/anchor_bundle.md):
 *   room frame  : +X, +Y on the floor, +Z up, floor at z = 0, metres.
 *   three.js    : +Y up.
 * The ONE conversion is `roomRoot.rotation.x = -PI/2`, which maps
 * room (x, y, z) -> world (x, z, -y). EVERYTHING that is in the room
 * frame (point cloud, people, sensor, walkable grid) is a child of
 * roomRoot and uses raw room coordinates. Nothing else in this file
 * swaps axes.
 *
 * People are drawn as 3D skeletons that match the YOLOv11 pose model used by Module 3: the same
 * COCO-17 limb list as Ultralytics (including the ear-to-shoulder links; there is no neck joint) and
 * the same colours: green face, orange shoulder line + arms, magenta torso sides + hip line, blue legs.
 *
 * Pose data on the wire (see Part B of the change guide): each track may carry an optional
 *     joints: [ [x, y, z, conf] | null,  ... x17 ]      (COCO-17 order, room frame, metres)
 * produced by M4 lifting M3's 2D keypoints through the depth image. COCO-17 has: nose, eyes,
 * ears, shoulders, elbows, wrists, hips, knees, ankles. It has NO finger or face-mesh points,
 * so the hands end at the wrist and the face is nose + eyes + ears.
 *
 * Rendering cost model (no per-frame allocation):
 *   - every track owns two THREE.InstancedMesh objects (17 joint spheres, 19 limb cylinders) that
 *     share ONE sphere geometry and ONE cylinder geometry across all tracks;
 *   - per frame only the instance matrices / colours are rewritten in place (scratch objects
 *     are module-level), and `count` is set to the number of joints/limbs actually drawn, so a
 *     null / low-confidence joint simply is not drawn;
 *   - TrackView objects are pooled: when a person leaves, their meshes are parked and re-used
 *     by the next person instead of being disposed and re-created.
 *
 * Weak / software GPUs: the viewer probes the GPU once at start-up. On a software renderer (llvmpipe, SwiftShader ...)
 * it switches to a LITE profile automatically (smaller point budget, no antialiasing, 1x pixel ratio, 24 fps cap).
 * URL switches (combine freely):  ?lite=1 | ?lite=0   ?points=200000   ?aa=0   ?dpr=1   ?fps=30
 *
 * A track with NO usable pose (M4's `joints` option is off, the person is coasting, or too few joints pass
 * the depth checks) is drawn as a dim, translucent TEMPLATE figure with the same skeleton, sized to the
 * tracker's height estimate and turned to face the walking direction. That is NOT measured pose: it only
 * shows position, height and heading. It is tagged "est" in the people list and drawn ghost-like so it
 * cannot be mistaken for a real pose.
 */

// ---------------------------------------------------------------- tunables

const CFG = {
  renderDelayS: 0.10,      // render this far behind "now" so we can interpolate
  extrapolateMaxS: 0.15,   // how far past the newest sample we will extrapolate
  sampleBufferMax: 180,    // ~6 s at 30 Hz
  trailSeconds: 5.0,
  trailMaxPoints: 170,
  hideAfterS: 1.0,         // not in the stream for this long -> hide
  removeAfterS: 4.0,       // ... and park / dispose after this long
  defaultHeightM: 1.72,
  skelMinJointConf: 0.2,   // joints below this confidence are not drawn
  skelMinDrawJoints: 3,    // fewer valid joints than this -> treat as "no pose"
  skelHoldS: 0.35,         // keep showing the last pose this long if joints briefly drop out
  skelMaxBoneLenM: 1.0,    // a "limb" longer than this is bad depth data -> not drawn
  skelMaxOffsetM: 1.5,     // pose centre further than this from the track -> ignore the pose
  poseHintAfterS: 6,       // tracked people but no pose data for this long -> show a hint
  trackPoolMax: 24,        // idle TrackViews kept for re-use
  frustumRangeM: 4.0,
  skelColorMode: "yolo",   // "yolo" = Ultralytics pose colours, "state" = colour by track state
  ghostOpacity: 0.65,      // opacity multiplier for the estimated (template) figure
  yawMinSpeed: 0.25,       // m/s: below this the template figure keeps its last heading
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
  white: new THREE.Color(0xffffff),
};

const $ = (id) => document.getElementById(id);

// ---------------------------------------------------------------- runtime profile (weak / software GPUs)

const QUERY = new URLSearchParams(location.search);

let fatalText = null; // a message that must stay on screen (GPU lost, WebGL unavailable ...)
function showFatal(text) {
  fatalText = text;
  const b = $("banner");
  b.textContent = text;
  b.className = "banner is-danger";
  b.hidden = false;
}
function clearFatal() { fatalText = null; updateBanner(); }

/** Which GPU the browser really uses ("" if it hides it), read from a throw-away context. */
function probeGpu() {
  try {
    const c = document.createElement("canvas");
    const gl = c.getContext("webgl2") || c.getContext("webgl");
    if (!gl) return { ok: false, name: "" };
    const ext = gl.getExtension("WEBGL_debug_renderer_info");
    const name = ext ? String(gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) || "") : "";
    const lose = gl.getExtension("WEBGL_lose_context");
    if (lose) lose.loseContext();
    return { ok: true, name };
  } catch (e) {
    return { ok: false, name: "" };
  }
}
const GPU = probeGpu();
const SOFTWARE_GL = /llvmpipe|softpipe|swrast|swiftshader|software|basic render/i.test(GPU.name);
// ?lite=1 forces the light profile, ?lite=0 the full one; otherwise it follows the GPU probe.
const LITE = QUERY.has("lite") ? QUERY.get("lite") !== "0" : SOFTWARE_GL;
const numParam = (key, dflt) => {
  const v = Number(QUERY.get(key));
  return QUERY.has(key) && Number.isFinite(v) ? v : dflt;
};
const PROFILE = {
  lite: LITE,
  maxPoints: Math.max(10000, numParam("points", LITE ? 120000 : 500000)),  // point-cloud budget (the file may hold 1.5 M)
  antialias: QUERY.has("aa") ? QUERY.get("aa") !== "0" : !LITE,
  dpr: Math.min(window.devicePixelRatio || 1, numParam("dpr", LITE ? 1 : 2)),
  maxFps: numParam("fps", LITE ? 24 : 0),                                  // 0 = every display frame
  maxPointPx: LITE ? 6 : 14,                                               // cap on one point's on-screen size
};
console.info("[viewer3d] GPU:", GPU.name || "(hidden by the browser)", "| profile:", PROFILE);

// ---------------------------------------------------------------- renderer / scene

const stage = $("stage");
let renderer;
try {
  renderer = new THREE.WebGLRenderer({ antialias: PROFILE.antialias, alpha: false, powerPreference: "high-performance" });
} catch (err) {
  showFatal("WebGL could not start in this browser (" + (err && err.message ? err.message : err) + "). Update the graphics driver, check about:support, or try ?lite=1.");
  throw err;
}
renderer.setPixelRatio(PROFILE.dpr);
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.localClippingEnabled = true; // per-material clipping (ceiling cut)
stage.appendChild(renderer.domElement);

// If the GPU resets, keep the page alive and let the browser restore the context (three.js re-uploads everything).
let contextLost = false;
renderer.domElement.addEventListener("webglcontextlost", (ev) => {
  ev.preventDefault();
  contextLost = true;
  showFatal("GPU context lost - waiting for the browser to restore it. If this repeats, reload with ?lite=1");
  console.warn("[viewer3d] WebGL context lost");
}, false);
renderer.domElement.addEventListener("webglcontextrestored", () => {
  contextLost = false;
  clearFatal();
  console.info("[viewer3d] WebGL context restored");
}, false);

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
// The skeleton limbs are lit (so they read as 3D tubes); every other object in the scene uses an
// unlit material, so this light only affects the people.
const keyLight = new THREE.DirectionalLight(0xffffff, 1.1);
keyLight.position.set(3, 6, 4);
scene.add(keyLight);

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

// PLY value types -> DataView getter name and byte size.
const PLY_TYPES = {
  char: [1, "getInt8"], int8: [1, "getInt8"], uchar: [1, "getUint8"], uint8: [1, "getUint8"],
  short: [2, "getInt16"], int16: [2, "getInt16"], ushort: [2, "getUint16"], uint16: [2, "getUint16"],
  int: [4, "getInt32"], int32: [4, "getInt32"], uint: [4, "getUint32"], uint32: [4, "getUint32"],
  float: [4, "getFloat32"], float32: [4, "getFloat32"], double: [8, "getFloat64"], float64: [8, "getFloat64"],
};

/** Download into ONE buffer (sized from Content-Length when known), with a progress toast. */
async function fetchBytes(url, label) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`GET ${url} -> ${res.status}`);
  if (!res.body || !res.body.getReader) return new Uint8Array(await res.arrayBuffer());
  const total = Number(res.headers.get("content-length")) || 0;
  const reader = res.body.getReader();
  let buf = new Uint8Array(total || 1 << 20), got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (got + value.length > buf.length) {
      const bigger = new Uint8Array(Math.max(buf.length * 2, got + value.length));
      bigger.set(buf.subarray(0, got));
      buf = bigger;
    }
    buf.set(value, got);
    got += value.length;
    showToast(`${label}… ${total ? Math.round((100 * got) / total) + "% " : ""}(${(got / 1048576).toFixed(1)} MB)`);
  }
  return buf.subarray(0, got);
}

/** Header of a vertex-first PLY, or null if it is not something the fast reader understands. */
function parsePlyHeader(bytes) {
  const head = new TextDecoder("latin1").decode(bytes.subarray(0, Math.min(bytes.length, 16384)));
  const end = head.indexOf("end_header");
  if (!head.startsWith("ply") || end < 0) return null;
  const nl = head.indexOf("\n", end);
  if (nl < 0) return null;
  let format = "", count = 0, elements = 0, inVertex = false;
  const props = [];
  for (const line of head.slice(0, end).split(/\r?\n/)) {
    const w = line.trim().split(/\s+/);
    if (w[0] === "format") format = w[1];
    else if (w[0] === "element") {
      elements++;
      inVertex = w[1] === "vertex";
      if (inVertex) { if (elements !== 1) return null; count = Number(w[2]); }
    } else if (w[0] === "property" && inVertex) {
      if (w[1] === "list") return null;
      props.push({ type: w[1], name: w[2] });
    }
  }
  return { format, count, props, dataStart: nl + 1 };
}

/** Keep at most `maxPoints` evenly spaced points of any geometry (used only for files the fast reader cannot parse). */
function decimateGeometry(g, maxPoints) {
  const n = g.getAttribute("position").count;
  if (n <= maxPoints) return g;
  const step = n / maxPoints, out = new THREE.BufferGeometry();
  for (const name of ["position", "color"]) {
    const a = g.getAttribute(name);
    if (!a) continue;
    const arr = new Float32Array(maxPoints * 3);
    for (let k = 0; k < maxPoints; k++) {
      const i = Math.floor(k * step);
      arr[k * 3] = a.getX(i); arr[k * 3 + 1] = a.getY(i); arr[k * 3 + 2] = a.getZ(i);
    }
    out.setAttribute(name, new THREE.BufferAttribute(arr, 3));
  }
  g.dispose();
  out.userData.sourcePoints = n;
  return out;
}

/**
 * Load the room point cloud WITHOUT materialising all of it. three.js' PLYLoader creates one object and several
 * plain-JS numbers per point (hundreds of MB for 1.5 M points); this reader pulls only the points it will draw
 * (an even stride through the file) straight into typed arrays. Anything unusual falls back to PLYLoader + decimation.
 */
async function loadPlyDecimated(url, maxPoints) {
  const bytes = await fetchBytes(httpUrl(url), "Loading room point cloud");
  const fallback = () => {
    const ab = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
    return decimateGeometry(plyLoader.parse(ab), maxPoints);
  };
  const h = parsePlyHeader(bytes);
  if (!h || !(h.format === "binary_little_endian" || h.format === "binary_big_endian") || !(h.count > 0)) return fallback();
  const little = h.format === "binary_little_endian";

  let stride = 0;
  const at = {};
  for (const p of h.props) {
    const ty = PLY_TYPES[p.type];
    if (!ty) return fallback();
    at[p.name] = { off: stride, get: ty[1], isFloat: ty[1] === "getFloat32" || ty[1] === "getFloat64" };
    stride += ty[0];
  }
  if (!at.x || !at.y || !at.z) return fallback();
  if (bytes.length < h.dataStart + h.count * stride) throw new Error("PLY file is truncated");

  const hasColor = !!(at.red && at.green && at.blue);
  const keep = Math.min(h.count, maxPoints), step = h.count / keep;
  const pos = new Float32Array(keep * 3);
  // Same colour handling as three's PLYLoader: file colours are sRGB, the shader wants linear.
  const col = hasColor ? new Float32Array(keep * 3) : null;
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const SRGB_LUT = new Float32Array(256);
  for (let i = 0; i < 256; i++) { const c = i / 255; SRGB_LUT[i] = c < 0.04045 ? c * 0.0773993808 : Math.pow(c * 0.9478672986 + 0.0521327014, 2.4); }
  const c8 = (a, base) => {
    const v = dv[a.get](base + a.off, little);
    return SRGB_LUT[a.isFloat ? Math.max(0, Math.min(255, Math.round(v * 255))) : Math.max(0, Math.min(255, v | 0))];
  };
  for (let k = 0; k < keep; k++) {
    const base = h.dataStart + Math.floor(k * step) * stride;
    pos[k * 3] = dv[at.x.get](base + at.x.off, little);
    pos[k * 3 + 1] = dv[at.y.get](base + at.y.off, little);
    pos[k * 3 + 2] = dv[at.z.get](base + at.z.off, little);
    if (col) { col[k * 3] = c8(at.red, base); col[k * 3 + 1] = c8(at.green, base); col[k * 3 + 2] = c8(at.blue, base); }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  if (col) g.setAttribute("color", new THREE.BufferAttribute(col, 3));
  g.userData.sourcePoints = h.count;
  return g;
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
    const geometry = await loadPlyDecimated(asset.url, PROFILE.maxPoints);
    disposePoints();
    const hasColor = geometry.hasAttribute("color");
    const mat = new THREE.PointsMaterial({
      size: state.pointSize,
      sizeAttenuation: true,
      vertexColors: hasColor,
      color: hasColor ? 0xffffff : 0x8fa0b2,
      clippingPlanes: [clipPlane],
    });
    // Cap the on-screen size of one point. With size attenuation, points right next to the camera would otherwise grow
    // into huge squares; millions of those overdraw the whole screen, which is what stalls (or crashes) weak GPUs.
    mat.onBeforeCompile = (shader) => {
      shader.vertexShader = shader.vertexShader.replace(
        "#include <worldpos_vertex>",
        `#include <worldpos_vertex>\n\tgl_PointSize = min( gl_PointSize, ${PROFILE.maxPointPx.toFixed(1)} );`
      );
    };
    mat.customProgramCacheKey = () => "clamped-points-" + PROFILE.maxPointPx;
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
    const srcN = geometry.userData.sourcePoints || n;
    $("map-info").textContent =
      `${(n / 1e6).toFixed(2)} M pts${srcN > n ? " of " + (srcN / 1e6).toFixed(2) + " M" : ""} · z ${bb.min.z.toFixed(2)} … ${bb.max.z.toFixed(2)} m${PROFILE.lite ? " · lite" : ""}`;
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

// ---------------------------------------------------------------- skeleton definition

// COCO-17 order (poi_perception/contracts.py).
const N_JOINTS = 17;
const J_STRIDE = 4; // x, y, z, conf  (conf < 0 means "missing")
const J_NAMES = [
  "nose", "left_eye", "right_eye", "left_ear", "right_ear",
  "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
  "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
];

// Limbs as joint-index pairs, in the SAME order as Ultralytics' YOLO pose skeleton (0-based COCO-17).
const COCO_BONES = [
  [15, 13], [13, 11], [16, 14], [14, 12],          // legs
  [11, 12], [5, 11], [6, 12],                      // hip line + torso sides
  [5, 6], [5, 7], [6, 8], [7, 9], [8, 10],         // shoulder line + arms (end at the wrists: the model has no fingers)
  [1, 2], [0, 1], [0, 2], [1, 3], [2, 4],          // face: eye bridge, nose-eyes, eyes-ears
  [3, 5], [4, 6],                                  // ears -> shoulders (this model has no neck joint)
];
const N_BONE_SLOTS = COCO_BONES.length;

// Body parts, used for colour grouping (left side is drawn a little darker than right).
const PART_FACE = 0, PART_TORSO = 1, PART_ARM_L = 2, PART_ARM_R = 3, PART_LEG_L = 4, PART_LEG_R = 5;
const PART_COUNT = 6;
const JOINT_PART = [
  PART_FACE, PART_FACE, PART_FACE, PART_FACE, PART_FACE,
  PART_TORSO, PART_TORSO,
  PART_ARM_L, PART_ARM_R, PART_ARM_L, PART_ARM_R,
  PART_TORSO, PART_TORSO,
  PART_LEG_L, PART_LEG_R, PART_LEG_L, PART_LEG_R,
];
const BONE_PART = [
  PART_LEG_L, PART_LEG_L, PART_LEG_R, PART_LEG_R,
  PART_TORSO, PART_TORSO, PART_TORSO,
  PART_TORSO, PART_ARM_L, PART_ARM_R, PART_ARM_L, PART_ARM_R,
  PART_FACE, PART_FACE, PART_FACE, PART_FACE, PART_FACE,
  PART_FACE, PART_FACE,
];
// Sphere radius per joint (metres) and limb radius per part.
const JOINT_R = [
  0.022, 0.014, 0.014, 0.016, 0.016,
  0.034, 0.034, 0.028, 0.028, 0.024, 0.024,
  0.034, 0.034, 0.030, 0.030, 0.026, 0.026,
];
const PART_BONE_R = [0.006, 0.016, 0.013, 0.013, 0.015, 0.015];
const BONE_R = BONE_PART.map((p) => PART_BONE_R[p]);

// Ultralytics' pose colours (RGB). Each limb / keypoint uses one palette entry, exactly as in the YOLO plots.
const YOLO_PALETTE = { 0: [255, 128, 0], 7: [255, 51, 255], 9: [51, 153, 255], 16: [0, 255, 0] };
const YOLO_LIMB_IDX = [9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16];
const YOLO_KPT_IDX = [16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9];
const YOLO_COLOR = {};
for (const key of Object.keys(YOLO_PALETTE)) {
  const [r, g, b] = YOLO_PALETTE[key];
  YOLO_COLOR[key] = new THREE.Color().setRGB(r / 255, g / 255, b / 255, THREE.SRGBColorSpace);
}
const JOINT_YOLO_COLOR = YOLO_KPT_IDX.map((i) => YOLO_COLOR[i]);
const BONE_YOLO_COLOR = YOLO_LIMB_IDX.map((i) => YOLO_COLOR[i]);

// Neutral standing figure used when a track has no pose: [forward, left, height] as fractions of body height.
const TEMPLATE = [
  [0.028, 0.000, 0.915], [0.022, 0.016, 0.935], [0.022, -0.016, 0.935], [0.000, 0.040, 0.925], [0.000, -0.040, 0.925],
  [0.000, 0.110, 0.820], [0.000, -0.110, 0.820],
  [0.000, 0.140, 0.640], [0.000, -0.140, 0.640], [0.000, 0.150, 0.470], [0.000, -0.150, 0.470],
  [0.000, 0.055, 0.530], [0.000, -0.055, 0.530],
  [0.000, 0.060, 0.285], [0.000, -0.060, 0.285], [0.000, 0.060, 0.040], [0.000, -0.060, 0.040],
];

// ---------------------------------------------------------------- shared geometry + scratch objects

const SPHERE_GEO = new THREE.SphereGeometry(1, PROFILE.lite ? 8 : 14, PROFILE.lite ? 6 : 10);                 // unit sphere, scaled per instance
const BONE_GEO = new THREE.CylinderGeometry(1, 1, 1, PROFILE.lite ? 6 : 10, 1, false);     // unit cylinder along +Y, scaled per instance
const CIRCLE_GEO = new THREE.CircleGeometry(1, 40);
const RING_GEO = (() => {
  const pts = [];
  for (let i = 0; i < 48; i++) {
    const a = (i / 48) * Math.PI * 2;
    pts.push(new THREE.Vector3(Math.cos(a), Math.sin(a), 0));
  }
  return new THREE.BufferGeometry().setFromPoints(pts);
})();

// Invisible, but ray-castable, standing cylinder so people can still be picked with a double-click.
const PICK_GEO = (() => {
  const g = new THREE.CylinderGeometry(0.35, 0.35, 1, 10, 1, false);
  g.rotateX(Math.PI / 2);  // axis Y -> Z (room up)
  g.translate(0, 0, 0.5);  // base on the floor, 1 m tall (scaled to the person's height)
  return g;
})();
const PICK_MAT = new THREE.MeshBasicMaterial({ visible: false });

const _UP = new THREE.Vector3(0, 1, 0);
const _p1 = new THREE.Vector3();
const _p2 = new THREE.Vector3();
const _dir = new THREE.Vector3();
const _mid = new THREE.Vector3();
const _scl = new THREE.Vector3();
const _q = new THREE.Quaternion();
const _qId = new THREE.Quaternion();
const _mat = new THREE.Matrix4();

/** Wire joints -> packed Float32Array(17 * 4), conf = -1 for missing. Null if nothing usable. */
function normJoints(raw) {
  if (!Array.isArray(raw)) return null;
  const out = new Float32Array(N_JOINTS * J_STRIDE);
  let n = 0;
  for (let k = 0; k < N_JOINTS; k++) {
    const j = raw[k];
    const o = k * J_STRIDE;
    if (Array.isArray(j) && j.length >= 3 && Number.isFinite(j[0]) && Number.isFinite(j[1]) && Number.isFinite(j[2])) {
      out[o] = j[0];
      out[o + 1] = j[1];
      out[o + 2] = j[2];
      out[o + 3] = Number.isFinite(j[3]) ? Math.max(0, j[3]) : 1;
      n++;
    } else {
      out[o + 3] = -1;
    }
  }
  return n > 0 ? out : null;
}

function makeLabel() {
  const canvas = document.createElement("canvas");
  canvas.width = 192; canvas.height = 72;
  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, depthTest: false, transparent: true }));
  sprite.scale.set(0.5, 0.1875, 1);
  sprite.renderOrder = 20;
  sprite.userData.canvas = canvas;
  return sprite;
}

/** Re-draw a label's text into its existing canvas texture (labels are re-used by pooled tracks). */
function paintLabel(sprite, text) {
  const canvas = sprite.userData.canvas;
  const g = canvas.getContext("2d");
  g.clearRect(0, 0, canvas.width, canvas.height);
  g.fillStyle = "rgba(10,14,19,0.78)";
  g.beginPath();
  if (g.roundRect) g.roundRect(4, 8, 184, 56, 12); else g.rect(4, 8, 184, 56);
  g.fill();
  g.font = "600 36px 'IBM Plex Mono', ui-monospace, monospace";
  g.fillStyle = "#e9eef3";
  g.textAlign = "center";
  g.textBaseline = "middle";
  g.fillText(text, 96, 37);
  sprite.material.map.needsUpdate = true;
}

const lerp = (a, b, f) => a + (b - a) * f;
function lerpAngle(a, b, f) {
  let d = ((b - a + Math.PI) % (2 * Math.PI)) - Math.PI;
  if (d < -Math.PI) d += 2 * Math.PI;
  return a + d * f;
}

/** One person: pooled skeleton meshes (measured pose, or an estimated template) + label + uncertainty ellipse + trail. */
class TrackView {
  constructor(id) {
    this.id = id;
    this.samples = [];
    this.lastSeen = performance.now();
    this.builtH = -1;
    this.speed = 0;
    this.last = null;        // last interpolated sample (for the HUD list / follow)
    this.lastPose = null;    // newest sample that carried joints
    this.hasPose = false;    // true when a drawable pose exists for the current frame
    this.colorKey = "";
    this.yaw = 0;            // heading used by the template figure
    this.isTemplate = false; // true while the estimated (template) figure is what is drawn
    this.ghostColor = new THREE.Color();
    this._ba = null;         // sample bracket found by sampleAt()
    this._bb = null;
    this._bf = 0;
    this.jBuf = new Float32Array(N_JOINTS * J_STRIDE); // joints resolved for the current frame
    this.partColors = [];
    for (let i = 0; i < PART_COUNT; i++) this.partColors.push(new THREE.Color());

    this.group = new THREE.Group(); // origin = ground point, room frame
    roomRoot.add(this.group);

    // --- invisible pick proxy (double-click selection)
    this.pick = new THREE.Mesh(PICK_GEO, PICK_MAT);
    this.pick.userData.trackId = id;
    this.group.add(this.pick);

    this.label = makeLabel();
    paintLabel(this.label, `#${id}`);
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

    // --- skeleton: 17 joint spheres + 17 limb cylinders, instanced and updated in place.
    // Joints/limbs are in absolute room coordinates, so this lives under roomRoot, not the moving group.
    this.skel = new THREE.Group();
    this.jointMat = new THREE.MeshLambertMaterial({ color: 0xffffff, transparent: true });
    this.boneMat = new THREE.MeshLambertMaterial({ color: 0xffffff, transparent: true });
    this.joints = new THREE.InstancedMesh(SPHERE_GEO, this.jointMat, N_JOINTS);
    this.bones = new THREE.InstancedMesh(BONE_GEO, this.boneMat, N_BONE_SLOTS);
    for (const m of [this.joints, this.bones]) {
      m.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
      for (let i = 0; i < m.instanceMatrix.count; i++) m.setColorAt(i, COLORS.tentative); // allocates instanceColor once
      m.instanceColor.setUsage(THREE.DynamicDrawUsage);
      m.frustumCulled = false; // instances move; the cached bounding sphere would go stale
      m.count = 0;
    }
    this.skel.add(this.joints, this.bones);
    this.skel.visible = false;
    roomRoot.add(this.skel);

  }

  /** Re-use this view for a different track id (called when taking one out of the pool). */
  reset(id) {
    this.id = id;
    this.samples.length = 0;
    this.lastSeen = performance.now();
    this.builtH = -1;
    this.speed = 0;
    this.last = null;
    this.lastPose = null;
    this.hasPose = false;
    this.colorKey = "";
    this.yaw = 0;
    this.isTemplate = false;
    this._ba = this._bb = null;
    this._bf = 0;
    this.pick.userData.trackId = id;
    paintLabel(this.label, `#${id}`);
    this.trailGeom.setDrawRange(0, 0);
    this.trail.visible = false;
    this.skel.visible = false;
    this.joints.count = 0;
    this.bones.count = 0;
    this.group.visible = true;
  }

  /** Hide everything and drop per-person data; the meshes stay allocated for the next person. */
  park() {
    this.group.visible = false;
    this.trail.visible = false;
    this.skel.visible = false;
    this.joints.count = 0;
    this.bones.count = 0;
    this.samples.length = 0;
    this.last = null;
    this.lastPose = null;
    this.hasPose = false;
    this.isTemplate = false;
    this._ba = this._bb = null;
  }

  pushSample(tr, t) {
    const last = this.samples[this.samples.length - 1];
    if (last && t <= last.t) return; // out of order / duplicate
    const joints = normJoints(tr.joints);
    const smp = {
      t, p: tr.p, v: tr.v, state: tr.state, src: tr.src, conf: tr.conf,
      height: tr.height, cov: tr.cov_xy, joints,
    };
    this.samples.push(smp);
    if (joints) this.lastPose = smp;
    if (this.samples.length > CFG.sampleBufferMax) this.samples.shift();
    this.lastSeen = performance.now();
  }

  /** Position/velocity interpolated at capture-time `tt`. Also records the sample bracket for the pose. */
  sampleAt(tt) {
    const s = this.samples;
    if (!s.length) return null;
    const first = s[0];
    if (tt <= first.t) { this._ba = this._bb = first; this._bf = 0; return first; }
    const last = s[s.length - 1];
    if (tt >= last.t) {
      const dt = Math.min(tt - last.t, CFG.extrapolateMaxS);
      this._ba = this._bb = last; this._bf = 0;
      return { ...last, p: [last.p[0] + last.v[0] * dt, last.p[1] + last.v[1] * dt, last.p[2]] };
    }
    for (let i = s.length - 1; i > 0; i--) {
      if (s[i - 1].t <= tt) {
        const a = s[i - 1], b = s[i];
        const f = (tt - a.t) / Math.max(b.t - a.t, 1e-6);
        this._ba = a; this._bb = b; this._bf = f;
        return {
          ...b,
          p: [lerp(a.p[0], b.p[0], f), lerp(a.p[1], b.p[1], f), lerp(a.p[2], b.p[2], f)],
          v: [lerp(a.v[0], b.v[0], f), lerp(a.v[1], b.v[1], f), lerp(a.v[2], b.v[2], f)],
        };
      }
    }
    this._ba = this._bb = first; this._bf = 0;
    return first;
  }

  /**
   * Fill this.jBuf with the pose for the current frame (interpolated between the two samples that
   * bracket render time, or the newest/last-known pose carried along with the track for up to
   * CFG.skelHoldS if joints briefly drop out). Returns the number of drawable joints, or 0 for "no pose".
   */
  _resolvePose(tt, s) {
    const A = this._ba, B = this._bb, buf = this.jBuf;
    if (!A || !B) return 0;
    const ja = A.joints, jb = B.joints;

    if (ja && jb && A !== B) {
      const f = this._bf;
      for (let k = 0; k < N_JOINTS; k++) {
        const o = k * J_STRIDE;
        const ca = ja[o + 3], cb = jb[o + 3];
        if (ca >= 0 && cb >= 0) {
          buf[o] = ja[o] + (jb[o] - ja[o]) * f;
          buf[o + 1] = ja[o + 1] + (jb[o + 1] - ja[o + 1]) * f;
          buf[o + 2] = ja[o + 2] + (jb[o + 2] - ja[o + 2]) * f;
          buf[o + 3] = cb;
        } else if (cb >= 0) {
          buf[o] = jb[o]; buf[o + 1] = jb[o + 1]; buf[o + 2] = jb[o + 2]; buf[o + 3] = cb;
        } else if (ca >= 0) {
          buf[o] = ja[o]; buf[o + 1] = ja[o + 1]; buf[o + 2] = ja[o + 2]; buf[o + 3] = ca;
        } else {
          buf[o + 3] = -1;
        }
      }
    } else {
      let src = null;
      if (jb && tt - B.t <= CFG.skelHoldS) src = B;
      else if (ja && tt - A.t <= CFG.skelHoldS) src = A;
      else if (this.lastPose && tt - this.lastPose.t <= CFG.skelHoldS) src = this.lastPose;
      if (!src) return 0;
      // carry the pose along with the (interpolated / extrapolated) floor position
      const jj = src.joints;
      const dx = s.p[0] - src.p[0], dy = s.p[1] - src.p[1];
      for (let k = 0; k < N_JOINTS; k++) {
        const o = k * J_STRIDE;
        if (jj[o + 3] >= 0) {
          buf[o] = jj[o] + dx; buf[o + 1] = jj[o + 1] + dy; buf[o + 2] = jj[o + 2]; buf[o + 3] = jj[o + 3];
        } else {
          buf[o + 3] = -1;
        }
      }
    }

    // How many joints are drawable, and is the pose plausibly attached to this track?
    const thr = CFG.skelMinJointConf;
    let n = 0, sx = 0, sy = 0;
    for (let k = 0; k < N_JOINTS; k++) {
      const o = k * J_STRIDE;
      if (buf[o + 3] >= thr) { n++; sx += buf[o]; sy += buf[o + 1]; }
    }
    if (n < CFG.skelMinDrawJoints) return 0;
    if (Math.hypot(sx / n - s.p[0], sy / n - s.p[1]) > CFG.skelMaxOffsetM) return 0;
    return n;
  }

  _setPartColors(base, key) {
    this.colorKey = key;
    const pc = this.partColors;
    pc[PART_FACE].copy(base).lerp(COLORS.white, 0.65);
    pc[PART_TORSO].copy(base);
    pc[PART_ARM_L].copy(base).lerp(COLORS.white, 0.2);
    pc[PART_LEG_L].copy(base).lerp(COLORS.white, 0.2);
    pc[PART_ARM_R].copy(base).lerp(COLORS.white, 0.5);
    pc[PART_LEG_R].copy(base).lerp(COLORS.white, 0.5);
  }

  _jointColor(k, ghost) {
    if (ghost) return this.ghostColor;
    return CFG.skelColorMode === "state" ? this.partColors[JOINT_PART[k]] : JOINT_YOLO_COLOR[k];
  }

  _boneColor(b, ghost) {
    if (ghost) return this.ghostColor;
    return CFG.skelColorMode === "state" ? this.partColors[BONE_PART[b]] : BONE_YOLO_COLOR[b];
  }

  /** No measured pose: stand a neutral figure on the tracker's floor point, scaled to its height, facing its heading. */
  _fillTemplate(s, h) {
    const buf = this.jBuf, c = Math.cos(this.yaw), sn = Math.sin(this.yaw);
    for (let k = 0; k < N_JOINTS; k++) {
      const o = k * J_STRIDE, tp = TEMPLATE[k];
      const f = tp[0] * h, l = tp[1] * h;
      buf[o] = s.p[0] + f * c - l * sn;
      buf[o + 1] = s.p[1] + f * sn + l * c;
      buf[o + 2] = tp[2] * h;
      buf[o + 3] = 1;
    }
  }

  /** Place limb instance `i` between _p1 and _p2. False (nothing written) for a degenerate or implausibly long limb. */
  _putBone(i, radius, color) {
    _dir.subVectors(_p2, _p1);
    const len = _dir.length();
    if (len < 1e-3 || len > CFG.skelMaxBoneLenM) return false;
    _dir.multiplyScalar(1 / len);
    _q.setFromUnitVectors(_UP, _dir);
    _mid.addVectors(_p1, _p2).multiplyScalar(0.5);
    _scl.set(radius, len, radius);
    _mat.compose(_mid, _q, _scl);
    this.bones.setMatrixAt(i, _mat);
    this.bones.setColorAt(i, color);
    return true;
  }

  /** Write the resolved pose into the instanced meshes. Returns the z of the highest drawn joint. */
  _drawPose(isSelected, ghost) {
    const buf = this.jBuf, thr = CFG.skelMinJointConf;
    const grow = isSelected ? 1.35 : 1.0;
    const jm = this.joints, bm = this.bones;

    let nj = 0, topZ = -Infinity;
    for (let k = 0; k < N_JOINTS; k++) {
      const o = k * J_STRIDE;
      if (!(buf[o + 3] >= thr)) continue; // also skips missing (-1) and NaN
      const r = JOINT_R[k] * grow;
      _p1.set(buf[o], buf[o + 1], buf[o + 2]);
      _scl.set(r, r, r);
      _mat.compose(_p1, _qId, _scl);
      jm.setMatrixAt(nj, _mat);
      jm.setColorAt(nj, this._jointColor(k, ghost));
      if (buf[o + 2] > topZ) topZ = buf[o + 2];
      nj++;
    }
    jm.count = nj;
    jm.instanceMatrix.needsUpdate = true;
    jm.instanceColor.needsUpdate = true;

    let nb = 0;
    for (let b = 0; b < COCO_BONES.length; b++) {
      const a = COCO_BONES[b][0] * J_STRIDE, c = COCO_BONES[b][1] * J_STRIDE;
      if (!(buf[a + 3] >= thr) || !(buf[c + 3] >= thr)) continue; // a null joint just drops its limbs
      _p1.set(buf[a], buf[a + 1], buf[a + 2]);
      _p2.set(buf[c], buf[c + 1], buf[c + 2]);
      if (this._putBone(nb, BONE_R[b] * grow, this._boneColor(b, ghost))) nb++;
    }
    bm.count = nb;
    bm.instanceMatrix.needsUpdate = true;
    bm.instanceColor.needsUpdate = true;

    return Number.isFinite(topZ) ? topZ : CFG.defaultHeightM;
  }

  update(tt, nowMs, L, isSelected) {
    const s = this.sampleAt(tt);
    if (!s) return;
    this.last = s;

    const age = nowMs - this.lastSeen;
    const visible = age < CFG.hideAfterS * 1000;
    this.group.visible = visible;
    if (!visible) { this.trail.visible = false; this.skel.visible = false; this.hasPose = false; return; }

    this.group.position.set(s.p[0], s.p[1], 0);

    const color = COLORS[s.state] || COLORS.tentative;
    const coasting = s.state === "coasting" || s.src === "predicted";
    let opacity = s.conf == null ? 1 : 0.4 + 0.6 * Math.min(1, Math.max(0, s.conf));
    if (s.state === "lost") opacity = Math.min(opacity, 0.35);
    else if (coasting) opacity = Math.min(opacity, 0.7);
    else if (s.state === "tentative") opacity = Math.min(opacity, 0.6);

    const h = s.height || CFG.defaultHeightM;
    this.speed = Math.hypot(s.v[0], s.v[1]);
    this.pick.scale.z = h;

    // ---- skeleton: the measured pose if there is one, else a dim template figure ("est")
    if (this.speed > CFG.yawMinSpeed) this.yaw = lerpAngle(this.yaw, Math.atan2(s.v[1], s.v[0]), 0.12);
    this.hasPose = this._resolvePose(tt, s) > 0;
    this.isTemplate = !this.hasPose && L.box && L.skel;
    if (this.colorKey !== s.state) this._setPartColors(color, s.state);
    const drawn = L.skel && (this.hasPose || this.isTemplate);
    let topZ = h;
    if (drawn) {
      if (this.isTemplate) {
        this._fillTemplate(s, h);
        this.ghostColor.copy(color).lerp(COLORS.bg, 0.2);
      }
      const a = this.isTemplate ? opacity * CFG.ghostOpacity : opacity;
      this.jointMat.opacity = a;
      this.boneMat.opacity = a;
      topZ = this._drawPose(isSelected, this.isTemplate);
    }
    this.skel.visible = drawn;

    this.label.position.set(0, 0, topZ + 0.22);
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
      this.ellipseFillMat.opacity = isSelected ? 0.3 : 0.16;
      this.ellipseLineMat.color.copy(color);
      this.ellipse.visible = true;
    } else {
      this.ellipse.visible = false;
    }

    this._updateTrail(tt, L.trails, color, s);
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

  /** Free GPU resources. The shared geometries (sphere, cylinder, pick, circle, ring) are NOT disposed. */
  dispose() {
    roomRoot.remove(this.group, this.trail, this.skel);
    this.label.material.map.dispose(); this.label.material.dispose();
    this.ellipseFillMat.dispose(); this.ellipseLineMat.dispose();
    this.trailGeom.dispose(); this.trailMat.dispose();
    this.joints.dispose(); this.bones.dispose();
    this.jointMat.dispose(); this.boneMat.dispose();
  }
}

// ---------------------------------------------------------------- track registry + pool

const tracks = new Map(); // id -> TrackView
const trackPool = [];     // parked TrackViews waiting for the next person

function acquireTrack(id) {
  const v = trackPool.pop();
  if (v) { v.reset(id); return v; }
  return new TrackView(id);
}

function releaseTrack(v) {
  v.park();
  if (trackPool.length < CFG.trackPoolMax) trackPool.push(v);
  else v.dispose();
}

function clearTracks() {
  for (const t of tracks.values()) releaseTrack(t);
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
    if (!v) { v = acquireTrack(tr.id); tracks.set(tr.id, v); }
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
  if (fatalText) { text = fatalText; danger = true; }
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
bindLayer("l-box", "box");   // no longer boxes: toggles the estimated template figure shown for people without pose data
{ // relabel the checkbox in place (index.html stays untouched)
  const lab = $("l-box").parentElement;
  if (lab) for (const n of lab.childNodes) if (n.nodeType === 3 && n.textContent.trim()) n.textContent = " Estimated figure (no pose)";
}
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
const _ndc = new THREE.Vector2();
const _pickables = [];
renderer.domElement.addEventListener("dblclick", (e) => {
  const r = renderer.domElement.getBoundingClientRect();
  _ndc.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
  raycaster.setFromCamera(_ndc, camera);
  _pickables.length = 0;
  for (const t of tracks.values()) if (t.group.visible) _pickables.push(t.pick);
  const hit = raycaster.intersectObjects(_pickables, false)[0];
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

// If people are being tracked but no pose data ever arrives, say why (the backend `joints` option is off by default).
const poseHint = { since: 0, shown: false };
function updatePoseHint(nPeople, nPosed) {
  const now = performance.now();
  if (nPeople === 0 || nPosed > 0 || !state.layers.skel) {
    poseHint.since = 0;
    if (poseHint.shown) { poseHint.shown = false; hideToast(); }
    return;
  }
  if (!poseHint.since) poseHint.since = now;
  if (!poseHint.shown && toast.hidden && now - poseHint.since > CFG.poseHintAfterS * 1000) {
    poseHint.shown = true;
    showToast("No 3D pose data is arriving, so dim estimated figures are shown. Enable \"joints\" in m4.json (Part B of the guide).");
    console.info("[viewer3d] no `joints` in the track stream - showing estimated figures only");
    setTimeout(() => { if (poseHint.shown) hideToast(); }, 12000);
  }
}

function renderTrackList() {
  const listEl = $("track-list");
  const seen = new Set();
  let n = 0, nPosed = 0;
  for (const t of [...tracks.values()].sort((a, b) => a.id - b.id)) {
    if (!t.group.visible || !t.last) continue;
    n++;
    if (t.hasPose) nPosed++;
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
    r.state.textContent = s.state + (t.hasPose ? " · 3D" : t.isTemplate ? " · est" : "");
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
  updatePoseHint(n, nPosed);
}
setInterval(renderTrackList, 250);

// ---------------------------------------------------------------- loop

let frames = 0, fpsT0 = performance.now();
const _follow = new THREE.Vector3();

let lastRenderMs = 0, renderErrors = 0;

function frame() {
  const now = performance.now();
  if (contextLost) return; // the browser is restoring the GPU context; nothing to draw into
  if (PROFILE.maxFps > 0 && now - lastRenderMs < 1000 / PROFILE.maxFps - 2) return;
  lastRenderMs = now;
  stepFly(now);

  const tt = renderCaptureTime();
  if (tt != null) {
    for (const [id, t] of tracks) {
      t.update(tt, now, state.layers, state.selectedId === id);
      if (now - t.lastSeen > CFG.removeAfterS * 1000) {
        releaseTrack(t);   // parked for re-use, not disposed
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
  try {
    renderer.render(scene, camera);
    renderErrors = 0;
  } catch (e) {
    if (++renderErrors === 1) console.error("[viewer3d] render failed:", e);
    if (renderErrors > 30) {
      renderer.setAnimationLoop(null);
      showFatal("Rendering stopped after repeated GPU errors (" + (e && e.message ? e.message : e) + "). Reload with ?lite=1");
    }
    return;
  }

  frames++;
  if (now - fpsT0 > 500) {
    $("perf").textContent = `${((frames * 1000) / (now - fpsT0)).toFixed(0)} fps`;
    frames = 0; fpsT0 = now;
  }
}
renderer.setAnimationLoop(frame);

// Handy for the browser console / automated tests.
window.__sts3d = {
  THREE, scene, roomRoot, camera, controls, tracks, trackPool, state, env, flyToView, CFG, client, PROFILE, GPU,
  acquireTrack, releaseTrack, J_NAMES, COCO_BONES, renderer,
};
