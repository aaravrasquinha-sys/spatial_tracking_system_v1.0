import { PoiClient, wsUrl } from "../common/ws-client.js";
import { SceneClient } from "../common/scene-client.js";
import { validateTracksMessage, validateHealthMessage, warnOnce } from "../common/validate.js";

// -------------------------------------------------------------- state

const state = {
  connStatus: "connecting",
  scene: null,
  latestTracks: null,       // last `tracks` message
  trails: new Map(),        // id -> [{t, x, y}]
  events: [],                // recent event rows, newest first
  frozen: false,
  groundTruth: null,         // parsed floor-marker JSON, or null
  layers: { trails: true, ellipses: true, markers: false },
};

const TRAIL_SECONDS = 4.0;
const MAX_EVENT_ROWS = 60;

// -------------------------------------------------------------- DOM

const $ = (id) => document.getElementById(id);
const el = {
  connDot: $("conn-dot"), connLabel: $("conn-label"),
  camId: $("cam-id"), frameBadge: $("frame-badge"), provisionalBanner: $("provisional-banner"),
  statFps: $("stat-fps"), statLatency: $("stat-latency"), statClients: $("stat-clients"), statTemp: $("stat-temp"),
  trackList: $("track-list"), trackCount: $("track-count"), eventLog: $("event-log"),
  canvas: $("radar"), stageOverlay: $("stage-overlay"),
  toggleTrails: $("toggle-trails"), toggleEllipses: $("toggle-ellipses"), toggleMarkers: $("toggle-markers"),
  btnFreeze: $("btn-freeze"), btnLoadMarkers: $("btn-load-markers"), gtFile: $("gt-file"),
};

// -------------------------------------------------------------- connection

const client = new PoiClient({ url: wsUrl() });
const scene = new SceneClient();

client.addEventListener("status", (e) => {
  state.connStatus = e.detail.state;
  renderConnStatus();
});

client.addEventListener("tracks", (e) => {
  const msg = e.detail;
  warnOnce(validateTracksMessage(msg));
  scene.noticeTracksMessage(msg).catch(() => {});
  if (state.frozen) return;
  state.latestTracks = msg;
  updateTrails(msg);
  updateFrameBadge(msg);
});

client.addEventListener("event", (e) => {
  pushEvent(e.detail);
});

client.addEventListener("health", (e) => {
  const msg = e.detail;
  warnOnce(validateHealthMessage(msg));
  renderHealth(msg);
});

scene.addEventListener("scene", (e) => {
  state.scene = e.detail;
});

client.connect();
scene.fetch().catch((err) => console.warn("initial /api/scene fetch failed:", err));

// -------------------------------------------------------------- connection UI

function renderConnStatus() {
  el.connDot.className = "dot is-" + state.connStatus;
  el.connLabel.textContent = {
    connecting: "connecting", open: "live", closed: "reconnecting…", stalled: "stream stalled",
  }[state.connStatus] || state.connStatus;
  el.stageOverlay.hidden = state.connStatus !== "stalled" && state.connStatus !== "closed";
}

function updateFrameBadge(msg) {
  const isRoom = msg.frame === "room";
  el.frameBadge.textContent = msg.frame;
  el.frameBadge.className = "frame-badge " + (isRoom ? "is-room" : "is-local");
  el.provisionalBanner.hidden = isRoom;
  el.camId.textContent = msg.cam_id;
}

function renderHealth(msg) {
  el.statFps.textContent = msg.fps.toFixed(1) + " fps";
  el.statFps.className = msg.fps < 15 ? "bad" : msg.fps < 24 ? "warn" : "";

  if (msg.latency_ms == null) {
    el.statLatency.textContent = "clock unsynced";
    el.statLatency.className = "warn";
  } else {
    el.statLatency.textContent = msg.latency_ms.toFixed(0) + " ms";
    el.statLatency.className = msg.latency_ms > 150 ? "warn" : "";
  }

  el.statClients.textContent = msg.clients + (msg.clients === 1 ? " client" : " clients");
  el.statTemp.textContent = msg.temp_c == null ? "-- \u00b0C" : msg.temp_c.toFixed(0) + " \u00b0C";
  el.statTemp.className = msg.temp_c != null && msg.temp_c > 75 ? "bad" : "";
}

// -------------------------------------------------------------- trails + events

function updateTrails(msg) {
  const nowT = msg.t_capture;
  const seen = new Set();
  for (const tr of msg.tracks) {
    seen.add(tr.id);
    let trail = state.trails.get(tr.id);
    if (!trail) { trail = []; state.trails.set(tr.id, trail); }
    trail.push({ t: nowT, x: tr.p[0], y: tr.p[1] });
    const cutoff = nowT - TRAIL_SECONDS;
    while (trail.length && trail[0].t < cutoff) trail.shift();
  }
  for (const [id, trail] of state.trails) {
    if (!seen.has(id) && (trail.length === 0 || nowT - trail[trail.length - 1].t > TRAIL_SECONDS)) {
      state.trails.delete(id);
    }
  }
}

function pushEvent(ev) {
  state.events.push(ev);
  if (state.events.length > MAX_EVENT_ROWS) state.events.shift();
  renderEvents();
}

function renderEvents() {
  el.eventLog.innerHTML = "";
  for (const ev of state.events.slice(-MAX_EVENT_ROWS)) {
    const row = document.createElement("div");
    row.className = "event-row ev-" + ev.event;
    const time = new Date(ev.t * 1000).toLocaleTimeString([], { hour12: false });
    row.innerHTML = '<span class="t">' + time + "</span>" + ev.event.replace("track_", "") + " #" + ev.id;
    el.eventLog.appendChild(row);
  }
}

// -------------------------------------------------------------- track list panel

function renderTrackList() {
  const tracks = state.latestTracks ? state.latestTracks.tracks : [];
  el.trackCount.textContent = String(tracks.length);
  if (tracks.length === 0) {
    el.trackList.innerHTML = '<p class="empty-hint">No one in frame.</p>';
    return;
  }
  el.trackList.innerHTML = "";
  for (const tr of tracks) {
    const card = document.createElement("div");
    card.className = "track-card state-" + tr.state;
    card.innerHTML =
      '<div class="track-card-head">' +
        '<span class="id">#' + tr.id + '</span>' +
        '<span class="state">' + tr.state + '</span>' +
      '</div>' +
      '<div class="track-card-body">' +
        '<span class="pos">' + tr.p[0].toFixed(2) + ', ' + tr.p[1].toFixed(2) + '</span>' +
        '<span><span class="src-dot src-' + tr.src + '"></span>' + tr.src + '</span>' +
      '</div>';
    el.trackList.appendChild(card);
  }
}

// -------------------------------------------------------------- toolbar

el.toggleTrails.addEventListener("change", () => (state.layers.trails = el.toggleTrails.checked));
el.toggleEllipses.addEventListener("change", () => (state.layers.ellipses = el.toggleEllipses.checked));
el.toggleMarkers.addEventListener("change", () => (state.layers.markers = el.toggleMarkers.checked));

el.btnFreeze.addEventListener("click", () => {
  state.frozen = !state.frozen;
  el.btnFreeze.classList.toggle("is-active", state.frozen);
  el.btnFreeze.textContent = state.frozen ? "Frozen (resume)" : "Freeze";
});

el.btnLoadMarkers.addEventListener("click", () => el.gtFile.click());
el.gtFile.addEventListener("change", async () => {
  const file = el.gtFile.files[0];
  if (!file) return;
  try {
    const parsed = JSON.parse(await file.text());
    state.groundTruth = normalizeMarkers(parsed);
    state.layers.markers = true;
    el.toggleMarkers.checked = true;
  } catch (e) {
    console.warn("could not parse floor-marker file:", e);
  }
});

function normalizeMarkers(parsed) {
  const list = Array.isArray(parsed) ? parsed : parsed.markers || parsed.points || [];
  return list
    .map((m) => {
      if (Array.isArray(m.position)) return { x: m.position[0], y: m.position[1], name: m.name || m.id || "" };
      if ("x" in m && "y" in m) return { x: m.x, y: m.y, name: m.name || m.id || "" };
      return null;
    })
    .filter(Boolean);
}

// -------------------------------------------------------------- radar canvas

const canvas = el.canvas;
const ctx = canvas.getContext("2d");
let dpr = Math.max(1, window.devicePixelRatio || 1);

function resizeCanvas() {
  const rect = canvas.parentElement.getBoundingClientRect();
  dpr = Math.max(1, window.devicePixelRatio || 1);
  canvas.width = Math.round(rect.width * dpr);
  canvas.height = Math.round(rect.height * dpr);
  canvas.style.width = rect.width + "px";
  canvas.style.height = rect.height + "px";
}
window.addEventListener("resize", resizeCanvas);
resizeCanvas();

let userZoom = 1.0;
let userOffset = { x: 0, y: 0 };
let dragging = null;

canvas.addEventListener("wheel", (e) => {
  e.preventDefault();
  const factor = Math.exp(-e.deltaY * 0.001);
  userZoom = Math.min(8, Math.max(0.25, userZoom * factor));
}, { passive: false });

canvas.addEventListener("pointerdown", (e) => {
  dragging = { x: e.clientX, y: e.clientY, ox: userOffset.x, oy: userOffset.y };
  canvas.setPointerCapture(e.pointerId);
});
canvas.addEventListener("pointermove", (e) => {
  if (!dragging) return;
  userOffset.x = dragging.ox + (e.clientX - dragging.x);
  userOffset.y = dragging.oy + (e.clientY - dragging.y);
});
canvas.addEventListener("pointerup", () => (dragging = null));
canvas.addEventListener("dblclick", () => { userZoom = 1; userOffset = { x: 0, y: 0 }; });

function computeAutoFit() {
  const cam = state.scene ? state.scene.camera.t : [0, 0, 2.3];
  let minX = cam[0] - 1, maxX = cam[0] + 1, minY = cam[1] - 1, maxY = cam[1] + 1;
  const tracks = state.latestTracks ? state.latestTracks.tracks : [];
  for (const tr of tracks) {
    minX = Math.min(minX, tr.p[0]); maxX = Math.max(maxX, tr.p[0]);
    minY = Math.min(minY, tr.p[1]); maxY = Math.max(maxY, tr.p[1]);
  }
  const padding = 1.2;
  const cx = (minX + maxX) / 2, cy = (minY + maxY) / 2;
  const halfSpan = Math.max((maxX - minX) / 2 + padding, (maxY - minY) / 2 + padding, 2.5);
  return { cx: cx, cy: cy, halfSpan: halfSpan };
}

function worldToScreen(x, y, fit, W, H, pxPerM) {
  const sx = W / 2 + (x - fit.cx) * pxPerM + userOffset.x * dpr;
  const sy = H / 2 - (y - fit.cy) * pxPerM + userOffset.y * dpr;
  return [sx, sy];
}

function drawGrid(fit, W, H, pxPerM) {
  ctx.strokeStyle = getVar("--grid-line");
  ctx.lineWidth = 1;
  const step = 1;
  const aspect = W / H > 1 ? W / H : 1;
  const left = fit.cx - fit.halfSpan * aspect - 2;
  const right = fit.cx + fit.halfSpan * aspect + 2;
  const bottom = fit.cy - fit.halfSpan - 2;
  const top = fit.cy + fit.halfSpan + 2;

  ctx.font = (10 * dpr) + 'px "IBM Plex Mono", monospace';
  ctx.fillStyle = getVar("--text-dim");

  for (let gx = Math.floor(left / step) * step; gx <= right; gx += step) {
    const p0 = worldToScreen(gx, 0, fit, W, H, pxPerM);
    ctx.beginPath(); ctx.moveTo(p0[0], 0); ctx.lineTo(p0[0], H); ctx.stroke();
    if (Math.round(gx) % 2 === 0) ctx.fillText(gx + "m", p0[0] + 3, H - 6);
  }
  for (let gy = Math.floor(bottom / step) * step; gy <= top; gy += step) {
    const p0 = worldToScreen(0, gy, fit, W, H, pxPerM);
    ctx.beginPath(); ctx.moveTo(0, p0[1]); ctx.lineTo(W, p0[1]); ctx.stroke();
  }
}

function drawCamera(fit, W, H, pxPerM) {
  if (!state.scene) return;
  const cam = state.scene.camera.t;
  const c0 = worldToScreen(cam[0], cam[1], fit, W, H, pxPerM);
  const sx = c0[0], sy = c0[1];
  const fovH = (state.scene.fov_deg && state.scene.fov_deg[0]) || 60;
  const halfAngle = (fovH / 2) * (Math.PI / 180);
  const range = fit.halfSpan * 2.4;

  ctx.save();
  ctx.beginPath();
  ctx.moveTo(sx, sy);
  const steps = 24;
  for (let i = 0; i <= steps; i++) {
    const a = -halfAngle + (2 * halfAngle) * (i / steps);
    const wx = cam[0] + Math.cos(a) * range;
    const wy = cam[1] + Math.sin(a) * range;
    const p = worldToScreen(wx, wy, fit, W, H, pxPerM);
    ctx.lineTo(p[0], p[1]);
  }
  ctx.closePath();
  ctx.fillStyle = "rgba(91, 157, 255, 0.06)";
  ctx.fill();
  ctx.strokeStyle = "rgba(91, 157, 255, 0.35)";
  ctx.lineWidth = 1;
  ctx.stroke();
  ctx.restore();

  ctx.save();
  ctx.fillStyle = getVar("--text");
  ctx.strokeStyle = getVar("--bg");
  ctx.lineWidth = 2 * dpr;
  ctx.beginPath();
  ctx.moveTo(sx, sy - 7 * dpr);
  ctx.lineTo(sx - 6 * dpr, sy + 5 * dpr);
  ctx.lineTo(sx + 6 * dpr, sy + 5 * dpr);
  ctx.closePath();
  ctx.stroke();
  ctx.fill();
  ctx.restore();
}

function drawMarkers(fit, W, H, pxPerM) {
  if (!state.groundTruth) return;
  ctx.save();
  ctx.strokeStyle = getVar("--text-muted");
  ctx.lineWidth = 1.5 * dpr;
  for (const m of state.groundTruth) {
    const p = worldToScreen(m.x, m.y, fit, W, H, pxPerM);
    const r = 6 * dpr;
    ctx.beginPath();
    ctx.moveTo(p[0] - r, p[1] - r); ctx.lineTo(p[0] + r, p[1] + r);
    ctx.moveTo(p[0] + r, p[1] - r); ctx.lineTo(p[0] - r, p[1] + r);
    ctx.stroke();
  }
  ctx.restore();
}

function stateColorVar(s) {
  return { confirmed: "--state-confirmed", coasting: "--state-coasting", lost: "--state-lost", tentative: "--state-tentative" }[s] || "--state-tentative";
}

function drawTrail(id, fit, W, H, pxPerM) {
  const trail = state.trails.get(id);
  if (!trail || trail.length < 2) return;
  const nowT = trail[trail.length - 1].t;
  const baseColor = getVar("--state-confirmed");
  ctx.save();
  for (let i = 1; i < trail.length; i++) {
    const a = trail[i - 1], b = trail[i];
    const age = nowT - b.t;
    const alpha = Math.max(0, 1 - age / TRAIL_SECONDS) * 0.5;
    const pa = worldToScreen(a.x, a.y, fit, W, H, pxPerM);
    const pb = worldToScreen(b.x, b.y, fit, W, H, pxPerM);
    ctx.strokeStyle = "color-mix(in srgb, " + baseColor + " " + Math.round(alpha * 100) + "%, transparent)";
    ctx.lineWidth = 2 * dpr;
    ctx.beginPath(); ctx.moveTo(pa[0], pa[1]); ctx.lineTo(pb[0], pb[1]); ctx.stroke();
  }
  ctx.restore();
}

function drawEllipse(tr, fit, W, H, pxPerM) {
  const a = tr.cov_xy[0], b = tr.cov_xy[1], c = tr.cov_xy[2];
  const trace = a + c, det = a * c - b * b;
  const disc = Math.sqrt(Math.max(0, trace * trace / 4 - det));
  const l1 = trace / 2 + disc, l2 = trace / 2 - disc;
  const angle = Math.abs(b) < 1e-12 ? 0 : Math.atan2(l1 - a, b);
  const k = 2;
  const rx = k * Math.sqrt(Math.max(l1, 0)) * pxPerM;
  const ry = k * Math.sqrt(Math.max(l2, 0)) * pxPerM;
  const p0 = worldToScreen(tr.p[0], tr.p[1], fit, W, H, pxPerM);

  ctx.save();
  ctx.translate(p0[0], p0[1]);
  ctx.rotate(-angle);
  ctx.beginPath();
  ctx.ellipse(0, 0, Math.max(rx, 1), Math.max(ry, 1), 0, 0, Math.PI * 2);
  const color = getVar(stateColorVar(tr.state));
  ctx.strokeStyle = "color-mix(in srgb, " + color + " 55%, transparent)";
  ctx.lineWidth = 1.5 * dpr;
  if (tr.state === "coasting" || tr.src === "predicted") ctx.setLineDash([4 * dpr, 3 * dpr]);
  ctx.stroke();
  ctx.restore();
}

function drawTrack(tr, fit, W, H, pxPerM) {
  const p0 = worldToScreen(tr.p[0], tr.p[1], fit, W, H, pxPerM);
  const sx = p0[0], sy = p0[1];
  const color = getVar(stateColorVar(tr.state));
  const r = 6 * dpr;

  ctx.save();
  if (tr.state === "lost") ctx.globalAlpha = 0.45;
  if (tr.state === "tentative") ctx.globalAlpha = 0.55;

  ctx.beginPath();
  ctx.arc(sx, sy, r, 0, Math.PI * 2);
  if (tr.state === "coasting") {
    ctx.strokeStyle = color; ctx.lineWidth = 2 * dpr; ctx.setLineDash([3 * dpr, 3 * dpr]); ctx.stroke();
  } else {
    ctx.fillStyle = color; ctx.fill();
  }
  ctx.setLineDash([]);

  const vx = tr.v[0], vy = tr.v[1];
  const speed = Math.hypot(vx, vy);
  if (speed > 0.05) {
    const pe = worldToScreen(tr.p[0] + vx * 0.5, tr.p[1] + vy * 0.5, fit, W, H, pxPerM);
    ctx.strokeStyle = color; ctx.lineWidth = 1.5 * dpr;
    ctx.beginPath(); ctx.moveTo(sx, sy); ctx.lineTo(pe[0], pe[1]); ctx.stroke();
  }

  const srcVarMap = { depth: "--src-depth", raycast: "--src-raycast", fused: "--src-fused", predicted: "--src-predicted" };
  const srcColor = getVar(srcVarMap[tr.src] || "--src-fused");
  ctx.beginPath(); ctx.arc(sx + r + 4 * dpr, sy - r - 2 * dpr, 2.5 * dpr, 0, Math.PI * 2);
  ctx.fillStyle = srcColor; ctx.fill();

  ctx.globalAlpha = Math.min(ctx.globalAlpha, 1) * 0.95;
  ctx.font = (11 * dpr) + 'px "IBM Plex Mono", monospace';
  ctx.fillStyle = getVar("--text-bright");
  ctx.fillText("#" + tr.id, sx + r + 8 * dpr, sy + 4 * dpr);
  ctx.restore();
}

const cssVarCache = new Map();
function getVar(name) {
  if (!cssVarCache.has(name)) {
    cssVarCache.set(name, getComputedStyle(document.documentElement).getPropertyValue(name).trim());
  }
  return cssVarCache.get(name);
}

function render() {
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = getVar("--bg");
  ctx.fillRect(0, 0, W, H);

  const fit = computeAutoFit();
  const spanPx = Math.min(W, H);
  const pxPerM = (spanPx / (fit.halfSpan * 2)) * userZoom;

  drawGrid(fit, W, H, pxPerM);
  if (state.layers.markers) drawMarkers(fit, W, H, pxPerM);
  drawCamera(fit, W, H, pxPerM);

  const tracks = state.latestTracks ? state.latestTracks.tracks : [];
  if (state.layers.trails) for (const tr of tracks) drawTrail(tr.id, fit, W, H, pxPerM);
  for (const tr of tracks) {
    if (state.layers.ellipses) drawEllipse(tr, fit, W, H, pxPerM);
    drawTrack(tr, fit, W, H, pxPerM);
  }

  renderTrackList();
  requestAnimationFrame(render);
}

requestAnimationFrame(render);
renderConnStatus();
