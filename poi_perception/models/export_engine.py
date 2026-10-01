"""
.pt -> ONNX -> TensorRT engine, run on the Orin itself (Section 3:
"vision models this size export directly on-device" -- unlike Module 1's
x86-hosted quantization tools, this one small enough to just build here).

Always records a manifest.json next to the engine with the JetPack
version, TensorRT version, and the exact export command used (Section 3:
"Lock and record... When JetPack is upgraded, re-export -- don't carry an
old engine forward and assume it still loads"). models/trt_engine.py
reads this manifest at load time and warns loudly if it's missing.

Usage:
    python -m poi_perception.models.export_engine \\
        --weights yolo11n-pose.pt --out models/engines/yolo11n_pose_fp16.engine \\
        --width 640 --height 480 --precision fp16

INT8 is intentionally not offered here -- Section 3: "INT8 is not worth
pursuing for this model class on this hardware: the accuracy loss is
disproportionate to the throughput gain at the n size." Stay at FP16
unless a real measured bottleneck says otherwise.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _read_jetpack_version() -> Optional[str]:
    """Best-effort JetPack identification, per Section 3's manifest
    requirement. Returns None off-Jetson (e.g. during a dev-machine dry
    run) rather than guessing."""
    p = Path("/etc/nv_tegra_release")
    if p.exists():
        return p.read_text().strip()
    return None


def _read_tensorrt_version() -> Optional[str]:
    try:
        import tensorrt as trt

        return trt.__version__
    except ImportError:
        return None


def export_to_onnx(weights_path: str, onnx_path: str, img_size: tuple) -> str:
    """.pt -> ONNX via ultralytics. Returns the exact command string used,
    for the manifest."""
    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise RuntimeError("ultralytics is required for the .pt -> ONNX step (`pip install ultralytics`).") from e

    model = YOLO(weights_path)
    # imgsz for ultralytics export takes (h, w); Section 1: don't force a
    # square 640x640 unless the exporter requires it -- pass the real
    # (height, width) through and let export fail loudly if it can't
    # honor a non-square size, rather than silently letterboxing.
    result_path = model.export(format="onnx", imgsz=list(img_size), opset=17, simplify=True)
    result_path = Path(result_path)
    if str(result_path) != onnx_path:
        Path(onnx_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(result_path), onnx_path)
    cmd = f"YOLO('{weights_path}').export(format='onnx', imgsz={list(img_size)}, opset=17, simplify=True)"
    return cmd


def build_trt_engine(onnx_path: str, engine_path: str, precision: str, workspace_mb: int) -> str:
    """ONNX -> TensorRT engine via trtexec. Returns the exact shell
    command used, for the manifest. trtexec is the standard, well-tested
    path for this on Jetson; wrapping the low-level TensorRT builder API
    ourselves would just re-implement it with more surface for bugs."""
    if shutil.which("trtexec") is None:
        raise RuntimeError(
            "trtexec not found on PATH. It ships with JetPack's TensorRT "
            "install (typically /usr/src/tensorrt/bin/trtexec) -- add it to "
            "PATH or pass its full path via --trtexec-bin."
        )
    Path(engine_path).parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--memPoolSize=workspace:{workspace_mb}",
    ]
    if precision == "fp16":
        cmd.append("--fp16")
    elif precision != "fp32":
        raise ValueError(f"Unsupported precision '{precision}' -- only fp32/fp16 (see module docstring re: INT8).")

    log_path = Path(engine_path).with_suffix(".trtexec.log")
    with open(log_path, "w") as logf:
        proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise RuntimeError(f"trtexec failed (exit {proc.returncode}); see {log_path}")
    return " ".join(cmd)


def write_manifest(
    engine_path: str,
    weights_path: str,
    onnx_cmd: str,
    trt_cmd: str,
    precision: str,
    img_size: tuple,
) -> str:
    manifest = {
        "engine_file": Path(engine_path).name,
        "weights_source": weights_path,
        "precision": precision,
        "img_size_hw": list(img_size),
        "jetpack_version": _read_jetpack_version(),
        "tensorrt_version": _read_tensorrt_version(),
        "onnx_export_command": onnx_cmd,
        "trtexec_command": trt_cmd,
        "engine_sha256_16": _sha256_of_file(Path(engine_path)) if Path(engine_path).exists() else None,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    manifest_path = Path(engine_path).with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return str(manifest_path)


def main(argv: Optional[list] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True, help="path to .pt checkpoint (COCO-pretrained is fine, Section 3)")
    ap.add_argument("--out", required=True, help="output .engine path")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--precision", choices=["fp16", "fp32"], default="fp16")
    ap.add_argument("--workspace-mb", type=int, default=2048)
    ap.add_argument("--onnx-out", default=None, help="defaults to <out>.onnx alongside the engine")
    ap.add_argument("--skip-onnx", action="store_true", help="onnx already exists at --onnx-out, skip re-exporting it")
    args = ap.parse_args(argv)

    onnx_path = args.onnx_out or str(Path(args.out).with_suffix(".onnx"))

    if args.skip_onnx:
        if not Path(onnx_path).exists():
            print(f"--skip-onnx set but {onnx_path} does not exist", file=sys.stderr)
            sys.exit(1)
        onnx_cmd = "(skipped -- reused existing ONNX file)"
    else:
        onnx_cmd = export_to_onnx(args.weights, onnx_path, (args.height, args.width))
        print(f"ONNX exported: {onnx_path}")

    trt_cmd = build_trt_engine(onnx_path, args.out, args.precision, args.workspace_mb)
    print(f"TensorRT engine built: {args.out}")

    manifest_path = write_manifest(
        args.out, args.weights, onnx_cmd, trt_cmd, args.precision, (args.height, args.width)
    )
    print(f"Manifest written: {manifest_path}")


if __name__ == "__main__":
    main()
