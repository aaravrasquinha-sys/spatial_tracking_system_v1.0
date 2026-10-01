"""
Run BEFORE touching the camera:  python -m pyslam.tools.env_probe [--realsense]

Checks the actual installed versions and symbol availability against
what the codebase assumes, and (with --realsense) queries the D435i
device itself for FW, intrinsics, and IMU presence.
"""
from __future__ import annotations
import sys
import json
import argparse

sys.path.insert(0, ".")


def parse_nvpmodel_conf(text: str) -> list:
    """WP-K4 (Phase A4): parse /etc/nvpmodel.conf into [{"id": int, "name": str}].
    Pure function (no device needed) so it is gate-testable -- see
    tests/gates/test_ga.py. The file declares each power mode as a line
    like `< POWER_MODEL ID=2 NAME=MAXN_SUPER >`; mode IDs and names differ
    between Jetson models AND between JetPack releases (on the Orin Nano
    dev kit under JetPack 6.2+, ID 0 is the 15W mode, NOT MAXN -- which is
    why the README's old hard-coded `nvpmodel -m 0` left clocks at
    15W-mode values), so the ID must be looked up from the NAME on the
    actual unit, never assumed."""
    import re
    modes = []
    for m in re.finditer(r"<\s*POWER_MODEL\s+ID\s*=\s*(\d+)\s+NAME\s*=\s*(.+?)\s*>", text):
        modes.append({"id": int(m.group(1)), "name": m.group(2).strip()})
    return modes


def parse_nvpmodel_query(text: str) -> dict:
    """`nvpmodel -q` prints e.g. "NV Power Mode: 15W\n0" -> {"name": "15W", "id": 0}."""
    import re
    name = None
    mid = None
    for line in text.splitlines():
        line = line.strip()
        m = re.match(r"NV Power Mode:\s*(.+)$", line)
        if m:
            name = m.group(1).strip()
        elif re.fullmatch(r"\d+", line):
            mid = int(line)
    return {"name": name, "id": mid}


def power_mode_advice(current_name, modes: list) -> dict:
    """Which mode should benchmarking/live runs use? Picks a MAXN* mode,
    preferring one whose name says SUPER (the higher-clock variant JetPack
    6.2 added for Orin Nano/NX) over plain MAXN. Returns
    {"is_max_mode", "recommended": {"id","name"} | None, "note"}.
    Never applies anything -- scripts/setup_orin.sh's `power apply` stage
    does that, and only when explicitly asked (max clocks cost heat)."""
    maxn = [m for m in modes if "MAXN" in m["name"].upper()]
    maxn.sort(key=lambda m: (0 if "SUPER" in m["name"].upper() else 1, m["id"]))
    rec = maxn[0] if maxn else None
    cur = (current_name or "").upper()
    is_max = "MAXN" in cur
    if not modes:
        note = ("could not read power modes from /etc/nvpmodel.conf; run `sudo nvpmodel -p --verbose` "
                "and pick the highest-performance mode by name")
    elif is_max:
        note = f"already in a MAXN mode ({current_name}); run `sudo jetson_clocks` to also lock clocks"
    elif rec is None:
        note = f"current mode {current_name!r}; no MAXN* mode found in nvpmodel.conf ({[m['name'] for m in modes]})"
    else:
        note = (f"current mode is {current_name!r}, NOT a MAXN mode: timings measured now are not "
                f"comparable to a max-clock run. Switch with `./scripts/setup_orin.sh power apply` "
                f"(= sudo nvpmodel -m {rec['id']}  # {rec['name']}, then sudo jetson_clocks)")
    return {"is_max_mode": is_max, "recommended": rec, "note": note}


def probe_jetson() -> dict:
    """WP-J0 (Orin port, Tier 0): report Jetson-specific facts env_probe
    never needed to know about before this port -- model/RAM (which
    ceiling applies to WM sizing), L4T/JetPack version (which GTSAM/
    OpenCV/librealsense build instructions apply), CUDA/cuDNN/TensorRT
    versions, whether OpenCV was built with CUDA support (needed before
    any Tier-3 GPU-offload work is worth attempting), current power mode
    and clock-lock state (nvpmodel/jetson_clocks -- results measured
    without these set are not comparable to results measured with them),
    and librealsense's IMU enumeration path. Every field degrades to
    None/False with a 'reason' rather than raising -- this must run
    cleanly on non-Jetson hardware too (e.g. this codebase's own x86 dev/
    CI environment), where it should simply report is_jetson=False.
    """
    import os
    import re
    import subprocess

    out: dict = {"is_jetson": False}

    model = None
    try:
        with open("/proc/device-tree/model", "rb") as f:
            model = f.read().split(b"\x00")[0].decode("utf-8", "ignore").strip()
    except Exception:
        pass
    if not model or "jetson" not in model.lower():
        out["reason"] = "no /proc/device-tree/model, or it doesn't mention 'jetson' -- not running on a Jetson"
        return out
    out["is_jetson"] = True
    out["model"] = model

    # RAM: 8GB vs 4GB Orin Nano changes the WM-size ceiling directly (see
    # the port plan's memory note -- each resident WM node keeps full
    # rgb+depth until evicted, ~1.5MB/node, shared with CUDA on this SoC).
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    out["ram_gb"] = round(kb / (1024 * 1024), 1)
                    break
    except Exception as e:
        out["ram_gb_error"] = str(e)

    # L4T / JetPack version.
    try:
        with open("/etc/nv_tegra_release") as f:
            out["l4t_release_raw"] = f.readline().strip()
    except Exception as e:
        out["l4t_release_raw"] = None
        out["l4t_release_error"] = str(e)
    try:
        r = subprocess.run(["dpkg-query", "-W", "-f=${Version}", "nvidia-l4t-core"],
                            capture_output=True, text=True, timeout=5)
        if r.returncode == 0 and r.stdout.strip():
            out["nvidia_l4t_core_version"] = r.stdout.strip()
    except Exception:
        pass
    try:
        r = subprocess.run(["dpkg-query", "-W", "-f=${Version}", "nvidia-jetpack"],
                            capture_output=True, text=True, timeout=5)
        if r.returncode == 0 and r.stdout.strip():
            out["jetpack_version"] = r.stdout.strip()
    except Exception:
        pass

    # CUDA toolkit version, independent of whether any Python package
    # picks it up -- GTSAM/OpenCV build scripts need this, not pip.
    try:
        r = subprocess.run(["nvcc", "--version"], capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            m = re.search(r"release (\d+\.\d+)", r.stdout)
            out["cuda_version"] = m.group(1) if m else r.stdout.strip().splitlines()[-1]
    except Exception:
        out["cuda_version"] = None
        out["cuda_note"] = "nvcc not found on PATH -- check /usr/local/cuda*/bin is on PATH"

    # OpenCV CUDA support -- decides whether the CUDA ORB / CUDA Hamming
    # matcher options in the port plan's Tier 3 are even reachable on
    # this install, or whether OpenCV needs a CUDA-enabled rebuild first.
    try:
        import cv2
        out["opencv_version"] = cv2.__version__
        try:
            out["opencv_cuda_device_count"] = cv2.cuda.getCudaEnabledDeviceCount()
        except Exception:
            out["opencv_cuda_device_count"] = 0
        out["opencv_has_cuda"] = out["opencv_cuda_device_count"] > 0
    except Exception as e:
        out["opencv_error"] = str(e)

    # Power mode + clock lock. Timing numbers collected WITHOUT these set
    # are not comparable to numbers collected with them -- see the port
    # plan's Tier-0 bring-up note.
    try:
        r = subprocess.run(["nvpmodel", "-q"], capture_output=True, text=True, timeout=5)
        out["nvpmodel_query"] = r.stdout.strip() if r.returncode == 0 else r.stderr.strip()
    except Exception as e:
        out["nvpmodel_query"] = None
        out["nvpmodel_error"] = str(e)
    # WP-K4: list this unit's own power modes and say which one to use --
    # see parse_nvpmodel_conf's docstring for why mode 0 must not be assumed.
    try:
        with open("/etc/nvpmodel.conf") as f:
            out["nvpmodel_modes"] = parse_nvpmodel_conf(f.read())
    except Exception as e:
        out["nvpmodel_modes"] = []
        out["nvpmodel_conf_error"] = str(e)
    cur = parse_nvpmodel_query(out.get("nvpmodel_query") or "")
    out["power_mode_current"] = cur
    out["power_advice"] = power_mode_advice(cur.get("name"), out["nvpmodel_modes"])
    try:
        r = subprocess.run(["jetson_clocks", "--show"], capture_output=True, text=True, timeout=5)
        out["jetson_clocks_show"] = (r.stdout or r.stderr).strip() if r.returncode == 0 else r.stderr.strip()
    except Exception as e:
        out["jetson_clocks_show"] = None
        out["jetson_clocks_error"] = str(e)

    return out


def probe_python_env() -> dict:
    import numpy, cv2, scipy, sklearn
    out = {
        "python": sys.version.split()[0],
        "numpy": numpy.__version__,
        "opencv": cv2.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
    }
    # symbols this codebase relies on
    symbols = ["ORB_create", "BFMatcher", "solvePnPRansac", "Rodrigues", "projectPoints"]
    out["opencv_symbols"] = {s: hasattr(cv2, s) for s in symbols}
    out["opencv_usac_magsac"] = hasattr(cv2, "USAC_MAGSAC")
    return out


def probe_gtsam() -> dict:
    try:
        import gtsam
        # minimal functional check, not just import
        g = gtsam.NonlinearFactorGraph()
        v = gtsam.Values()
        v.insert(0, gtsam.Pose3())
        return {"available": True, "functional_check": "ok"}
    except ImportError as e:
        return {"available": False, "reason": str(e)}
    except Exception as e:
        return {"available": True, "functional_check": f"FAILED: {e}"}


def probe_pyrealsense2() -> dict:
    try:
        import pyrealsense2 as rs
    except ImportError as e:
        return {"available": False, "reason": str(e)}
    return {"available": True, "version": getattr(rs, "__version__", "unknown")}


def probe_realsense_device() -> dict:
    try:
        import pyrealsense2 as rs
    except ImportError:
        return {"error": "pyrealsense2 not importable"}

    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) == 0:
        return {"error": "no RealSense device found -- check USB connection/permissions"}

    dev = devices[0]
    info = {
        "name": dev.get_info(rs.camera_info.name),
        "serial_number": dev.get_info(rs.camera_info.serial_number),
        "firmware_version": dev.get_info(rs.camera_info.firmware_version),
        "usb_type": dev.get_info(rs.camera_info.usb_type_descriptor) if dev.supports(rs.camera_info.usb_type_descriptor) else "unknown",
    }

    # IMU presence, ACTUAL RATES (not assumed), and part inference.
    # Phase 0's realsense.py hardcoded accel=250Hz/gyro=200Hz, which are
    # right for a BMI055 but not necessarily for a BMI085 (per your
    # brief's part number) or a later hw revision. Report what the
    # device itself advertises so a wrong assumption shows up here
    # rather than silently failing a stream request during a recording
    # session -- see the Phase 1 plan's finding F7.
    accel_rates, gyro_rates = set(), set()
    for s in dev.query_sensors():
        for p in s.get_stream_profiles():
            if p.stream_type() == rs.stream.accel:
                accel_rates.add(p.fps())
            elif p.stream_type() == rs.stream.gyro:
                gyro_rates.add(p.fps())
    info["imu_streams_available"] = sorted(
        {str(t) for t in (rs.stream.accel, rs.stream.gyro) if
         (accel_rates if t == rs.stream.accel else gyro_rates)}
    )
    info["accel_rates_hz_available"] = sorted(accel_rates)
    info["gyro_rates_hz_available"] = sorted(gyro_rates)

    from pyslam.sensors.realsense import infer_imu_part
    part_guess = infer_imu_part(accel_rates, gyro_rates)
    info["imu_part_guess"] = part_guess if part_guess else ["unknown -- rates don't cleanly match BMI055 or BMI085"]
    info["imu_calibration_warning"] = (
        "D435i ships with IMU intrinsics UNCALIBRATED from factory. "
        "If accel/gyro readings look biased or roll/pitch drift quickly, "
        "run Intel's IMU calibration tool before enabling any IMU-based "
        "fusion (not used by the Phase-0/1 pipeline yet, but recorded)."
    )

    # WP-J3: does a single sensor expose BOTH accel and gyro? That's the
    # precondition RealSenseSource(imu_capture_mode="callback") needs --
    # report it here so a "falling back to synced" warning during an
    # actual run isn't the first time this is discovered.
    has_combined_motion_sensor = False
    for s in dev.query_sensors():
        stream_types = {p.stream_type() for p in s.get_stream_profiles()}
        if rs.stream.accel in stream_types and rs.stream.gyro in stream_types:
            has_combined_motion_sensor = True
            break
    info["imu_callback_mode_supported"] = has_combined_motion_sensor

    # try a short pipeline start to confirm streaming actually works
    try:
        pipeline = rs.pipeline()
        cfg = rs.config()
        # WP-J3: test rgb8 first, since that's now RealSenseSource's
        # default request (falls back to bgr8 there if this fails) --
        # report which one actually works rather than assuming.
        color_format_used = "rgb8"
        try:
            cfg.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
            cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            profile = pipeline.start(cfg)
        except RuntimeError:
            pipeline = rs.pipeline()
            cfg = rs.config()
            cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            profile = pipeline.start(cfg)
            color_format_used = "bgr8 (rgb8 rejected by device/firmware)"
        info["color_format_used"] = color_format_used
        color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        rs_intr = color_stream.get_intrinsics()
        depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        info["measured_intrinsics"] = {
            "fx": rs_intr.fx, "fy": rs_intr.fy, "cx": rs_intr.ppx, "cy": rs_intr.ppy,
            "width": rs_intr.width, "height": rs_intr.height, "depth_scale": depth_scale,
        }
        expected = dict(fx=606.75, fy=606.57, cx=320.19, cy=237.06, width=640, height=480,
                         depth_scale=0.0010000000474974513)
        mismatches = []
        for k, v in expected.items():
            got = info["measured_intrinsics"].get(k)
            if got is None:
                continue
            if isinstance(v, float):
                if abs(got - v) > 0.5:
                    mismatches.append(f"{k}: expected~{v}, got {got}")
            elif got != v:
                mismatches.append(f"{k}: expected {v}, got {got}")
        info["intrinsics_match_expected"] = (len(mismatches) == 0)
        if mismatches:
            info["intrinsics_mismatches"] = mismatches
        pipeline.stop()
    except Exception as e:
        info["stream_test_error"] = str(e)

    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--realsense", action="store_true", help="also query the connected D435i")
    args = ap.parse_args()

    report = {
        "jetson": probe_jetson(),
        "python_env": probe_python_env(),
        "gtsam": probe_gtsam(),
        "pyrealsense2": probe_pyrealsense2(),
    }
    if args.realsense:
        report["realsense_device"] = probe_realsense_device()

    print(json.dumps(report, indent=2))

    # human-readable summary + hard warnings
    print("\n--- summary ---")
    if report["jetson"]["is_jetson"]:
        j = report["jetson"]
        print(f"Jetson: {j.get('model')}  RAM={j.get('ram_gb', '?')}GB  "
              f"JetPack={j.get('jetpack_version', '?')}  L4T={j.get('l4t_release_raw', '?')}  "
              f"CUDA={j.get('cuda_version', 'NOT FOUND')}")
        print(f"  OpenCV CUDA: {'yes' if j.get('opencv_has_cuda') else 'NO -- CPU-only OpenCV build'}")
        if j.get("nvpmodel_query"):
            print(f"  nvpmodel: {j['nvpmodel_query'].splitlines()[0] if j['nvpmodel_query'] else '?'}")
        print(f"  jetson_clocks: {'not queried/available' if not j.get('jetson_clocks_show') else j['jetson_clocks_show'].splitlines()[0]}")
        adv = j.get("power_advice")
        if adv:
            print(f"  POWER: {'OK' if adv['is_max_mode'] else 'WARNING'} -- {adv['note']}")
        if j.get("opencv_has_cuda") is False and j.get("cuda_version"):
            print(f"  (CUDA {j['cuda_version']} toolkit is installed; only OpenCV lacks CUDA -- a rebuild unlocks GPU matching)")
    else:
        print(f"Jetson: not detected ({report['jetson'].get('reason', 'n/a')})")
    print(f"OpenCV: {report['python_env']['opencv']}  "
          f"(USAC_MAGSAC {'available' if report['python_env']['opencv_usac_magsac'] else 'MISSING -> falls back to SOLVEPNP_ITERATIVE'})")
    if report["gtsam"]["available"]:
        print(f"GTSAM: available, functional check: {report['gtsam']['functional_check']}")
    else:
        print(f"GTSAM: NOT available ({report['gtsam']['reason']}) -> pipeline will use the native backend")
    if args.realsense:
        dev = report.get("realsense_device", {})
        if "error" in dev:
            print(f"RealSense: ERROR - {dev['error']}")
        else:
            print(f"RealSense: {dev.get('name')} SN={dev.get('serial_number')} "
                  f"FW={dev.get('firmware_version')} USB={dev.get('usb_type')}")
            if not dev.get("intrinsics_match_expected", True):
                print(f"  WARNING: measured intrinsics differ from the expected rig values: "
                      f"{dev.get('intrinsics_mismatches')}")
            if "accel_rates_hz_available" in dev:
                print(f"  IMU rates available: accel={dev['accel_rates_hz_available']}Hz "
                      f"gyro={dev['gyro_rates_hz_available']}Hz -> part guess: {dev['imu_part_guess']}")
            print(f"  IMU callback-mode supported: {dev.get('imu_callback_mode_supported')} "
                  f"(if False, RealSenseSource will auto-fall-back to 'synced' mode -- see run_slam.py --imu-mode)")
            if "color_format_used" in dev:
                print(f"  Color format: {dev['color_format_used']}")


if __name__ == "__main__":
    main()
