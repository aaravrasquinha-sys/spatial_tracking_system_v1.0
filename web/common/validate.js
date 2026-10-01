/**
 * A deliberately small runtime check, not a full JSON-schema
 * validator: confirms the fields SCHEMA.md promises are present with
 * the right JS type, so a server-side field rename shows up as a
 * loud console warning + a status badge instead of a silently blank
 * dashboard. See poi_present/schema.py and
 * tests/test_poi_present/test_schema_examples.py for the
 * Python-side half of this same contract.
 */

const TRACK_FIELDS = {
  id: "number", state: "string", p: "array", v: "array",
  cov_xy: "array", src: "string", age_s: "number",
};

function typeOf(v) {
  if (Array.isArray(v)) return "array";
  if (v === null) return "null";
  return typeof v;
}

function checkFields(obj, spec, path, problems) {
  for (const [key, expected] of Object.entries(spec)) {
    if (!(key in obj)) {
      problems.push(`${path}.${key} missing`);
      continue;
    }
    const actual = typeOf(obj[key]);
    if (actual !== expected) {
      problems.push(`${path}.${key} expected ${expected}, got ${actual}`);
    }
  }
}

export function validateTracksMessage(msg) {
  const problems = [];
  checkFields(msg, {
    type: "string", schema: "string", cam_id: "string", frame: "string",
    seq: "number", t_capture: "number", t_publish: "number", tracks: "array",
  }, "tracks", problems);
  if (msg.frame !== undefined && !["local", "room"].includes(msg.frame)) {
    problems.push(`tracks.frame unexpected value ${msg.frame}`);
  }
  (msg.tracks || []).forEach((t, i) => checkFields(t, TRACK_FIELDS, `tracks.tracks[${i}]`, problems));
  return problems;
}

export function validateHealthMessage(msg) {
  const problems = [];
  checkFields(msg, {
    type: "string", fps: "number", calib: "string", drop_rate: "number", clients: "number", clock: "string",
  }, "health", problems);
  return problems;
}

let _warnedOnce = new Set();

/** Logs each distinct problem once per page load, so a persistent
 * mismatch doesn't spam the console at 30Hz. */
export function warnOnce(problems) {
  for (const p of problems) {
    if (!_warnedOnce.has(p)) {
      _warnedOnce.add(p);
      console.warn(`[poi_present schema] ${p}`);
    }
  }
  return problems.length > 0;
}
