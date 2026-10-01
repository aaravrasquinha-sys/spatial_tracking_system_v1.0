"""`sts doctor` -- preflight. Fails loudly on the environment problems that would
otherwise show up as a confusing crash (or worse, a silently wrong result) later.

Levels: PASS / WARN / FAIL / SKIP. Exit code 1 if any FAIL. Nothing here changes
the system. Hardware checks (`--camera`) take the camera lock and open the D435i briefly.
"""
from __future__ import annotations

import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

from sts.paths import REPO_ROOT
from sts.site import SiteConfig


@dataclass
class Result:
    name: str
    level: str          # PASS | WARN | FAIL | SKIP
    detail: str = ""
    fix: str = ""


class Doctor:
    def __init__(self, site: Optional[SiteConfig], cam_id: Optional[str] = None, camera: bool = False):
        self.site, self.camera = site, camera
        self.cam_id = cam_id
        self.results: List[Result] = []

    def add(self, name, level, detail="", fix=""):
        self.results.append(Result(name, level, detail, fix))

    # ------------------------------------------------------------------ groups
    def check_python_and_paths(self):
        v = sys.version_info
        self.add("python >= 3.10", "PASS" if v >= (3, 10) else "FAIL", platform.python_version(),
                 "JetPack 6 ships 3.10" if v < (3, 10) else "")
        for pkg in ("pyslam", "poi_perception", "poi_localization", "poi_present", "sts"):
            try:
                m = importlib.import_module(pkg)
                f = Path(m.__file__).resolve()
                inside = REPO_ROOT in f.parents
                self.add(f"import {pkg} resolves inside this repo", "PASS" if inside else "FAIL", str(f),
                         "" if inside else "a stale editable install shadows this repo. `pip uninstall poi-perception "
                                            "spatial-tracking-system` then `pip install -e .` from the repo root")
            except Exception as e:
                self.add(f"import {pkg}", "FAIL", f"{type(e).__name__}: {e}", "pip install -e . from the repo root")
        # duplicate distributions providing the same top-level package (the PRESENT.md warning)
        try:
            from importlib import metadata
            owners = {}
            for d in metadata.distributions():
                tl = (d.read_text("top_level.txt") or "").split()
                for t in tl:
                    if t in ("poi_perception", "poi_localization", "poi_present", "pyslam"):
                        owners.setdefault(t, set()).add(d.metadata["Name"])
            dup = {k: sorted(v) for k, v in owners.items() if len(v) > 1}
            self.add("no duplicate installs of a first-party package", "FAIL" if dup else "PASS",
                     json.dumps(dup) if dup else "", "pip uninstall the extras" if dup else "")
        except Exception as e:
            self.add("duplicate-install scan", "SKIP", str(e))

    def check_native(self):
        import numpy
        nv = numpy.__version__
        self.add("numpy", "PASS" if nv.startswith("1.26") else "WARN", nv,
                 "the Orin venv is built around numpy 1.26.x (opencv-python-headless 4.8 + source-built gtsam). "
                 "pip install -c requirements/constraints.txt ..." if not nv.startswith("1.26") else "")
        for mod, need in (("scipy", True), ("cv2", True), ("websockets", True), ("psutil", True)):
            try:
                m = importlib.import_module(mod)
                self.add(f"import {mod}", "PASS", getattr(m, "__version__", ""))
            except Exception as e:
                self.add(f"import {mod}", "FAIL", str(e), f"pip install -r requirements/runtime.txt")
        try:
            import websockets
            major = int(websockets.__version__.split(".")[0])
            self.add("websockets in [14, 18)", "PASS" if 14 <= major < 18 else "FAIL", websockets.__version__,
                     "pip install 'websockets>=14,<18'" if not 14 <= major < 18 else "")
        except Exception:
            pass
        for mod, purpose in (("gtsam", "M1 iSAM2/batch backend (M1 falls back to the native pose graph)"),
                             ("sklearn", "M1/M2 (scikit-learn)")):
            try:
                m = importlib.import_module(mod)
                self.add(f"import {mod}", "PASS", getattr(m, "__version__", ""))
            except Exception as e:
                self.add(f"import {mod}", "WARN", str(e), purpose)
        try:
            import pyrealsense2 as rs
            f = getattr(rs, "__file__", "?")
            self.add("pyrealsense2", "PASS", f"{getattr(rs, '__version__', '?')} at {f}",
                     "")
            if "site-packages" in str(f) and "dist-packages" not in str(f):
                self.add("pyrealsense2 is the source-built RSUSB/CUDA build", "WARN",
                         f"loaded from {f}: if this is the pip wheel it shadows the build scripts/setup_orin.sh makes "
                         f"(RSUSB backend + CUDA align)", "check `rs-enumerate-devices` output for 'RSUSB', or reinstall per setup_orin.sh")
        except Exception as e:
            self.add("pyrealsense2", "FAIL", str(e), "scripts/setup_orin.sh librealsense")
        try:
            import tensorrt as trt
            self.add("tensorrt", "PASS", trt.__version__)
        except Exception as e:
            self.add("tensorrt", "FAIL", str(e), "ships with JetPack; do not pip install on the Orin")
        try:
            import pycuda.autoinit  # noqa: F401
            self.add("pycuda context", "PASS")
        except Exception as e:
            self.add("pycuda context", "FAIL", f"{type(e).__name__}: {e}", "needed by poi_perception.models.trt_engine")

    def check_jetson(self):
        model = Path("/proc/device-tree/model")
        if not model.exists():
            self.add("running on a Jetson", "WARN", "not a Jetson (dev machine?)")
            return
        self.add("running on a Jetson", "PASS", model.read_bytes().rstrip(b"\0").decode(errors="replace"))
        for tool, args, label in (("nvpmodel", ["-q"], "power mode"), ("jetson_clocks", ["--show"], "clocks")):
            exe = shutil.which(tool) or (f"/usr/sbin/{tool}" if Path(f"/usr/sbin/{tool}").exists() else None)
            if not exe:
                self.add(label, "SKIP", f"{tool} not found")
                continue
            try:
                r = subprocess.run([exe, *args], capture_output=True, text=True, timeout=10)
                txt = (r.stdout or r.stderr).strip().replace("\n", " | ")[:200]
                self.add(label, "PASS" if r.returncode == 0 else "WARN", txt,
                         "`scripts/setup_orin.sh power apply` to lock a fixed mode + clocks (thermal-throttle risk on long runs)")
            except Exception as e:
                self.add(label, "SKIP", str(e))

    def check_resources(self):
        import psutil
        vm = psutil.virtual_memory()
        self.add("memory available", "PASS" if vm.available > 2.5e9 else "WARN", f"{vm.available/1e9:.1f} GB free of {vm.total/1e9:.1f} GB",
                 "mapping (M1) wants most of the 8 GB; close other workloads" if vm.available <= 2.5e9 else "")
        root = self.site.data_root() if self.site else REPO_ROOT
        root.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(root).free
        self.add("disk free under data/", "PASS" if free > 20e9 else "WARN", f"{free/1e9:.0f} GB",
                 "bags are ~1-2 GB/min at 640x480; prune data/recordings" if free <= 20e9 else "")

    def check_site(self):
        if self.site is None:
            self.add("site.json", "FAIL", "not found", "python -m sts init")
            return
        s = self.site
        self.add("site.json loads", "PASS", f"site={s.site} cameras={[c.cam_id for c in s.cameras]} phase={s.localization.phase}")
        prof = s.capture_profile_path()
        self.add("capture profile exists", "PASS" if prof.exists() else "FAIL", str(prof))
        if s.perception.backend == "ultralytics":
            self.add("perception.backend", "WARN", "ultralytics (PyTorch) is for export/dev only",
                     "the runtime must use backend 'trt'; the pip torch on this Orin is not the JetPack build")
        if len(s.enabled_cameras()) != 1:
            self.add("exactly one enabled camera", "WARN", f"{len(s.enabled_cameras())} enabled; `sts run` supports one")
        cam = s.camera(self.cam_id) if (self.cam_id or len(s.enabled_cameras()) == 1) else None
        if cam is None:
            return
        eng = s.engine_path()
        if s.perception.backend == "trt":
            self.add("TensorRT engine file", "PASS" if eng.exists() else "FAIL", str(eng),
                     "sts engine  (builds it on THIS Orin; engines are device-specific)" if not eng.exists() else "")
            man = eng.with_suffix(".manifest.json")
            if eng.exists():
                if not man.exists():
                    self.add("engine manifest", "WARN", f"{man.name} missing", "re-export with sts engine")
                else:
                    m = json.loads(man.read_text())
                    hw = m.get("img_size_hw")
                    ok = hw == [cam.height, cam.width]
                    self.add("engine input size == camera size", "PASS" if ok else "FAIL",
                             f"engine {hw} vs camera {[cam.height, cam.width]}",
                             "re-export with --width/--height matching the camera" if not ok else "")
                    tegra = Path("/etc/nv_tegra_release")
                    if tegra.exists() and m.get("jetpack_version"):
                        same = m["jetpack_version"].strip() == tegra.read_text().strip()
                        self.add("engine built on this JetPack", "PASS" if same else "FAIL",
                                 "" if same else "JetPack changed since export",
                                 "re-export: an old engine may fail to load or silently misbehave" if not same else "")
        # config generation round-trip (catches bad overrides)
        try:
            from sts.configgen import write_configs
            write_configs(s, cam.cam_id)
            self.add("generated M3/M4/M5 configs load through the modules' own loaders", "PASS")
        except Exception as e:
            self.add("generated configs", "FAIL", str(e))
        # data products
        self.add("map bundle", "PASS" if (s.map_dir() / "dense" / "points.ply").exists() else "SKIP", str(s.map_dir()),
                 "sts map")
        cal = s.calibration_path(cam.cam_id)
        self.add("accepted calibration", "PASS" if cal.exists() else "SKIP", str(cal),
                 "sts anchor calibrate (Phase A runs without it)")

    def check_camera(self):
        if not self.camera:
            self.add("camera hardware checks", "SKIP", "pass --camera to enable (takes the camera lock)")
            return
        from sts.camera_lock import CameraBusy, CameraLock
        s = self.site
        cam = s.camera(self.cam_id)
        try:
            lock = CameraLock(s.data_root() / "locks", "default", "doctor").acquire()
        except CameraBusy as e:
            self.add("camera lock", "FAIL", str(e))
            return
        try:
            import pyrealsense2 as rs
            ctx = rs.context()
            devs = list(ctx.query_devices())
            if not devs:
                self.add("D435i detected", "FAIL", "no RealSense device", "check cable/udev rules (setup_orin.sh librealsense)")
                return
            d = devs[0]
            if cam.serial:
                match = [x for x in devs if x.get_info(rs.camera_info.serial_number) == cam.serial]
                if not match:
                    self.add("camera serial", "FAIL", f"{cam.serial} not among {[x.get_info(rs.camera_info.serial_number) for x in devs]}")
                    return
                d = match[0]
            name = d.get_info(rs.camera_info.name)
            usb = d.get_info(rs.camera_info.usb_type_descriptor) if d.supports(rs.camera_info.usb_type_descriptor) else "?"
            self.add("D435i detected", "PASS", f"{name} serial={d.get_info(rs.camera_info.serial_number)} "
                     f"fw={d.get_info(rs.camera_info.firmware_version)}")
            self.add("USB 3.x link", "PASS" if str(usb).startswith("3") else "FAIL", f"usb={usb}",
                     "a passive long cable drops the D435i to USB2; co-locate the Orin with the camera" if not str(usb).startswith("3") else "")
            has_imu = any("Motion" in s_.get_info(rs.camera_info.name) for s_ in d.query_sensors())
            self.add("IMU present", "PASS" if has_imu else "FAIL", "", "M2 needs the accelerometer for its roll/pitch cross-check")
            # intrinsics + global time via a brief pipeline
            p = rs.pipeline(); c = rs.config()
            c.enable_device(d.get_info(rs.camera_info.serial_number))
            c.enable_stream(rs.stream.color, cam.width, cam.height, rs.format.rgb8, cam.fps)
            c.enable_stream(rs.stream.depth, cam.width, cam.height, rs.format.z16, cam.fps)
            prof = p.start(c)
            try:
                i = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
                diff = max(abs(i.fx - cam.expected_fx), abs(i.fy - cam.expected_fy), abs(i.ppx - cam.expected_cx), abs(i.ppy - cam.expected_cy))
                self.add("device intrinsics vs site.json", "PASS" if diff <= cam.intrinsics_tol_px else "FAIL",
                         f"device fx={i.fx:.2f} fy={i.fy:.2f} cx={i.ppx:.2f} cy={i.ppy:.2f}; max diff {diff:.2f}px",
                         "update cameras[].expected_* in site.json AND run_live_map.py's hardcoded fork-mode intrinsics "
                         "(606.75/606.57/320.19/237.06) if they differ" if diff > cam.intrinsics_tol_px else "")
                ds = prof.get_device().first_depth_sensor()
                gt = ds.supports(rs.option.global_time_enabled) and ds.get_option(rs.option.global_time_enabled) == 1
                self.add("depth sensor global time", "PASS" if gt else "WARN", "on" if gt else "off (the runtime source enables it; informational)")
            finally:
                p.stop()
        except Exception as e:
            self.add("camera checks", "FAIL", f"{type(e).__name__}: {e}")
        finally:
            lock.release()

    def run(self) -> List[Result]:
        self.check_python_and_paths()
        self.check_native()
        self.check_jetson()
        self.check_resources()
        self.check_site()
        self.check_camera()
        return self.results

    def render(self) -> str:
        glyph = {"PASS": "PASS", "WARN": "WARN", "FAIL": "FAIL", "SKIP": "skip"}
        lines = []
        for r in self.results:
            lines.append(f"[{glyph[r.level]}] {r.name}" + (f": {r.detail}" if r.detail else ""))
            if r.fix and r.level in ("FAIL", "WARN"):
                lines.append(f"        -> {r.fix}")
        n = {k: sum(1 for r in self.results if r.level == k) for k in ("PASS", "WARN", "FAIL", "SKIP")}
        lines.append(f"\n{n['PASS']} pass, {n['WARN']} warn, {n['FAIL']} FAIL, {n['SKIP']} skipped")
        return "\n".join(lines)

    @property
    def ok(self) -> bool:
        return not any(r.level == "FAIL" for r in self.results)
