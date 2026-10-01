/**
 * GET /api/scene -- a one-shot document, not part of the ~30Hz
 * stream (SCHEMA.md). Fetched once at startup and re-fetched
 * whenever a `tracks` message reports a different frame/map_id than
 * last seen (a daemon restart in the other phase).
 */
import { httpUrl } from "./ws-client.js";

export class SceneClient extends EventTarget {
  constructor() {
    super();
    this.doc = null;
    this._lastFrame = null;
    this._lastMapId = undefined;
  }

  async fetch() {
    const res = await fetch(httpUrl("/api/scene"), { cache: "no-store" });
    if (!res.ok) throw new Error(`GET /api/scene -> ${res.status}`);
    this.doc = await res.json();
    this._lastFrame = this.doc.frame;
    this._lastMapId = this.doc.map_id;
    this.dispatchEvent(new CustomEvent("scene", { detail: this.doc }));
    return this.doc;
  }

  /** Call on every `tracks` message; re-fetches only when frame/map_id changed. */
  async noticeTracksMessage(msg) {
    if (msg.frame !== this._lastFrame || msg.map_id !== this._lastMapId) {
      await this.fetch();
    }
  }
}
