/**
 * Shared WebSocket client for both the dashboard and the VR viewer.
 * Owns reconnection (with backoff), ping/pong clock-offset tracking
 * (SCHEMA.md's "pong" section), and stream-stalled detection
 * (SCHEMA.md's "stream stalled vs nobody here" section) -- both
 * viewers listen to the same events so that logic lives in one place.
 *
 * Usage:
 *   const client = new PoiClient({ url: wsUrl() });
 *   client.on("tracks", (msg) => ...);
 *   client.on("event", (msg) => ...);
 *   client.on("health", (msg) => ...);
 *   client.on("status", ({ state }) => ...);   // "connecting"|"open"|"closed"|"stalled"
 *   client.connect();
 */

export function wsUrl(path = "/ws") {
  const scheme = location.protocol === "https:" ? "wss" : "ws";
  const token = new URLSearchParams(location.search).get("token");
  const q = token ? `?token=${encodeURIComponent(token)}` : "";
  return `${scheme}://${location.host}${path}${q}`;
}

export function httpUrl(path) {
  const token = new URLSearchParams(location.search).get("token");
  const q = token ? `?token=${encodeURIComponent(token)}` : "";
  return `${path}${q}`;
}

const STALL_THRESHOLD_MS = 700; // a few 30Hz frame periods -- SCHEMA.md's definition
const PING_INTERVAL_MS = 2000;
const RECONNECT_BASE_MS = 500;
const RECONNECT_MAX_MS = 8000;

export class PoiClient extends EventTarget {
  constructor({ url }) {
    super();
    this.url = url;
    this.ws = null;
    this._reconnectDelay = RECONNECT_BASE_MS;
    this._lastFrameAt = 0;
    this._stallTimer = null;
    this._pingTimer = null;
    this._stopped = false;

    // Clock offset estimation: offset such that serverTime ~=
    // performance-independent Date.now() + offset. Keeps the sample
    // with the lowest observed round-trip time.
    this.clockOffsetMs = 0;
    this._bestRtt = Infinity;
  }

  connect() {
    this._stopped = false;
    this._open();
  }

  close() {
    this._stopped = true;
    clearTimeout(this._stallTimer);
    clearInterval(this._pingTimer);
    if (this.ws) this.ws.close();
  }

  send(obj) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(obj));
    }
  }

  _emit(type, detail) {
    this.dispatchEvent(new CustomEvent(type, { detail }));
  }

  _open() {
    if (this._stopped) return;
    this._emit("status", { state: "connecting" });
    let ws;
    try {
      ws = new WebSocket(this.url);
    } catch (e) {
      this._scheduleReconnect();
      return;
    }
    this.ws = ws;

    ws.onopen = () => {
      this._reconnectDelay = RECONNECT_BASE_MS;
      this._emit("status", { state: "open" });
      this._armStallWatch();
      this._pingTimer = setInterval(() => this._ping(), PING_INTERVAL_MS);
      this._ping();
    };

    ws.onmessage = (ev) => {
      let msg;
      try {
        msg = JSON.parse(ev.data);
      } catch {
        return;
      }
      if (msg.type === "tracks") {
        this._lastFrameAt = performance.now();
        this._armStallWatch();
        this._emit("tracks", msg);
      } else if (msg.type === "event") {
        this._emit("event", msg);
      } else if (msg.type === "health") {
        this._emit("health", msg);
      } else if (msg.type === "pong") {
        this._handlePong(msg);
      }
    };

    ws.onclose = () => {
      clearInterval(this._pingTimer);
      clearTimeout(this._stallTimer);
      this._emit("status", { state: "closed" });
      this._scheduleReconnect();
    };

    ws.onerror = () => {
      /* onclose follows; nothing extra to do here */
    };
  }

  _scheduleReconnect() {
    if (this._stopped) return;
    setTimeout(() => this._open(), this._reconnectDelay);
    this._reconnectDelay = Math.min(this._reconnectDelay * 1.7, RECONNECT_MAX_MS);
  }

  _armStallWatch() {
    clearTimeout(this._stallTimer);
    this._stallTimer = setTimeout(() => {
      this._emit("status", { state: "stalled" });
    }, STALL_THRESHOLD_MS);
  }

  _ping() {
    const client_t = Date.now() / 1000;
    this._pingSentAt = performance.now();
    this.send({ type: "ping", client_t });
  }

  _handlePong(msg) {
    const rtt = performance.now() - (this._pingSentAt || performance.now());
    if (rtt < this._bestRtt) {
      this._bestRtt = rtt;
      const serverEpochMs = msg.server_t * 1000;
      const nowEpochMs = Date.now();
      // Best guess at server time "now", correcting for one-way
      // latency (~rtt/2), minus our own clock -> offset to add to
      // Date.now() to approximate the server's clock.
      this.clockOffsetMs = serverEpochMs + rtt / 2 - nowEpochMs;
      this._emit("clock", { offsetMs: this.clockOffsetMs, rttMs: rtt });
    }
  }

  /** Best estimate of the server's current wall-clock time, ms since epoch. */
  serverNowMs() {
    return Date.now() + this.clockOffsetMs;
  }
}
