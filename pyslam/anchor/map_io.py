"""
Read module 1's dense/points.ply (and, if present, manifest.json for the
map_id). M1's own writer (pyslam.mapping.export.write_ply) emits a
fixed binary little-endian layout; this reader also tolerates ascii and
extra properties so a hand-exported or externally-cleaned cloud works.
"""
from __future__ import annotations
import json
import os
from typing import Optional, Tuple
import numpy as np

from pyslam.core.log import get_logger

log = get_logger("anchor.map_io")

_PLY_TYPES = {"char": "i1", "uchar": "u1", "short": "<i2", "ushort": "<u2",
              "int": "<i4", "uint": "<u4", "float": "<f4", "double": "<f8",
              "int8": "i1", "uint8": "u1", "int16": "<i2", "uint16": "<u2",
              "int32": "<i4", "uint32": "<u4", "float32": "<f4", "float64": "<f8"}


def read_ply(path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """-> (pts (N,3) float64, colors (N,3) uint8 or None)."""
    with open(path, "rb") as f:
        header = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"{path}: truncated PLY header")
            s = line.decode("ascii", errors="replace").strip()
            header.append(s)
            if s == "end_header":
                break
        fmt = next((h.split()[1] for h in header if h.startswith("format")), None)
        n = 0
        props = []
        in_vertex = False
        for h in header:
            p = h.split()
            if h.startswith("element"):
                in_vertex = (p[1] == "vertex")
                if in_vertex:
                    n = int(p[2])
            elif h.startswith("property") and in_vertex:
                if p[1] == "list":
                    raise ValueError("list properties in the vertex element are not supported")
                props.append((p[2], _PLY_TYPES[p[1]]))
        if fmt == "binary_little_endian":
            dt = np.dtype([(nm, ty) for nm, ty in props])
            arr = np.frombuffer(f.read(n * dt.itemsize), dtype=dt, count=n)
            get = lambda k: arr[k]
            names = set(arr.dtype.names)
        elif fmt == "ascii":
            data = np.loadtxt(f, max_rows=n)
            data = data.reshape(n, -1)
            idx = {nm: i for i, (nm, _) in enumerate(props)}
            get = lambda k: data[:, idx[k]]
            names = set(idx)
        else:
            raise ValueError(f"unsupported PLY format {fmt!r}")
    pts = np.stack([np.asarray(get(k), dtype=np.float64) for k in ("x", "y", "z")], axis=1)
    colors = None
    for trip in (("red", "green", "blue"), ("r", "g", "b")):
        if set(trip) <= names:
            colors = np.stack([np.asarray(get(k)) for k in trip], axis=1).astype(np.uint8)
            break
    return pts, colors


def resolve_map_id(map_dir: str, content_files=("dense/points.ply", "capture_profile.json")) -> dict:
    """Use the manifest's map_id if the bundle was locked (lockstep mode);
    otherwise compute it with M1's own compute_map_id so the two agree
    if the same bundle is locked later. Returns {map_id, source, recomputed, mismatch}."""
    from pyslam.mapping.lock import compute_map_id
    recomputed = compute_map_id(map_dir, list(content_files))
    man = os.path.join(map_dir, "manifest.json")
    if os.path.exists(man):
        mid = json.load(open(man)).get("map_id")
        if mid:
            mismatch = (mid != recomputed)
            if mismatch:
                log.warning(f"manifest map_id {mid} != recomputed {recomputed} over {content_files}; "
                            f"using the manifest's (the bundle may have been edited after locking)")
            return {"map_id": mid, "source": "manifest.json", "recomputed": recomputed, "mismatch": mismatch}
    return {"map_id": recomputed, "source": "computed(dense/points.ply+capture_profile.json)",
            "recomputed": recomputed, "mismatch": False}


def load_map_bundle(map_dir: str) -> dict:
    ply = os.path.join(map_dir, "dense", "points.ply")
    if not os.path.exists(ply):
        raise FileNotFoundError(f"{ply} not found -- is {map_dir} a module 1 map bundle?")
    pts, colors = read_ply(ply)
    prof = os.path.join(map_dir, "capture_profile.json")
    prof_sha = None
    if os.path.exists(prof):
        from pyslam.live.capture_profile import CaptureProfile
        prof_sha = CaptureProfile.load(prof).content_hash()
    ident = resolve_map_id(map_dir)
    log.info(f"Loaded map {ident['map_id']} ({ident['source']}): {pts.shape[0]} points")
    return {"pts_map": pts, "colors": colors, "map_id": ident["map_id"], "map_id_info": ident,
            "capture_profile_sha": prof_sha, "map_dir": map_dir}
