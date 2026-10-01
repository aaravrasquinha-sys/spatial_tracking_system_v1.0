"""
Intel RealSense D435i capture.

Design notes:
  - Depth is aligned to the color stream in hardware/software via
    rs.align, so Frame.rgb and Frame.depth always share one intrinsics
    matrix and one pixel grid. This removes an entire class of bug.
  - Intrinsics are read FROM THE DEVICE, not hardcoded, but the expected
    values for this rig (see run_slam.py --realsense) are:
      fx=606.75 fy=606.57 cx=320.19 cy=237.06 640x480
      baseline=0.0499 depth_scale=0.0010000000474974513
    env_probe.py should be run first to confirm these match; a mismatch
    is a hard warning, not a silent override.
  - Accel and gyro rates are QUERIED from the device's stream profiles
    (see _pick_motion_fps), not hardcoded. An earlier version hardcoded
    250Hz/200Hz, which are correct for the D435i's BMI055 IMU but not
    necessarily for a BMI085 unit (per your brief's part number) or a
    later hardware revision -- see the Phase 1 plan's finding F7 and
    tools/env_probe.py, which reports the queried rates so a wrong
    assumption here would show up before it costs you a recording.
  - Recording uses librealsense's own recorder (config.enable_record_to_file),
    NOT a Python-side writer. This is the F7 fix: a Python writer that
    only sees frames after our (currently ~30-300ms/frame) pipeline has
    consumed them silently subsamples and desyncs IMU from video. The
    native recorder runs in the C++ SDK layer and captures every stream
    at its full hardware rate independent of how fast this process
    drains frames.
  - All timestamps use frame.get_timestamp() (device/hardware clock,
    milliseconds) converted to seconds, per the core.types "one
    monotonic float64 timeline" rule. Host time is not used for fusion.
"""
from __future__ import annotations
from typing import Iterator, Optional
import threading
import numpy as np

from pyslam.core.types import Frame, Intrinsics
from pyslam.core.log import get_logger

log = get_logger("sensors.realsense")

# D435i nominal IMU rates by part, for the env_probe sanity check only
# (never used to choose the actual stream request -- see _pick_motion_fps).
_KNOWN_IMU_PARTS = {
    "BMI055": {"accel_hz": (63, 250), "gyro_hz": (200, 400)},
    "BMI085": {"accel_hz": (100, 200), "gyro_hz": (100, 200)},
}


def infer_imu_part(accel_rates_hz, gyro_rates_hz) -> list[str]:
    """Pure function (no device needed) so this logic is unit-testable
    without hardware -- see tests/gates/test_g0.py::test_imu_part_inference."""
    accel_rates, gyro_rates = set(accel_rates_hz), set(gyro_rates_hz)
    if not accel_rates or not gyro_rates:
        return []
    matches = []
    for part, spec in _KNOWN_IMU_PARTS.items():
        a_lo, a_hi = spec["accel_hz"]
        g_lo, g_hi = spec["gyro_hz"]
        if all(a_lo - 1 <= r <= a_hi + 1 for r in accel_rates) and \
           all(g_lo - 1 <= r <= g_hi + 1 for r in gyro_rates):
            matches.append(part)
    return matches


def _pick_motion_fps(dev, rs, stream_type) -> Optional[int]:
    """Query the device's own advertised profiles for a motion stream and
    return the highest fps it actually offers, rather than assuming a
    fixed rate. Returns None if the stream isn't available at all."""
    best = None
    for sensor in dev.query_sensors():
        for p in sensor.get_stream_profiles():
            if p.stream_type() == stream_type:
                fps = p.fps()
                if best is None or fps > best:
                    best = fps
    return best


class RealSenseSource:
    """Live capture from a connected D435i, or (playback=True) deterministic
    replay of a native .bag recorded by this same class."""

    def __init__(self, width: int = 640, height: int = 480, fps: int = 30,
                 enable_imu: bool = True, serial: Optional[str] = None,
                 record_path: Optional[str] = None, playback_path: Optional[str] = None,
                 imu_capture_mode: str = "callback"):
        """
        imu_capture_mode (WP-J3, Orin port): "callback" (default) or
        "synced".
          - "synced" is the original Phase-0 behaviour: motion samples are
            only drained from whatever frameset wait_for_frames() happens
            to return alongside a color+depth pair. Between two
            wait_for_frames() calls (which on a slower frontend, e.g. a
            different platform, can take noticeably longer), any IMU
            samples that arrived and were NOT bundled into that frameset
            are dropped by the SDK before we ever see them -- the
            recorder's own C++-layer capture (record_path) is unaffected,
            but LIVE fusion quality (gravity alignment, the P5.2 tilt
            prior) degrades silently on a slower frontend.
          - "callback" opens the motion sensor directly and registers a
            librealsense-SDK-managed callback thread that appends every
            accel/gyro sample into the same self._accel_buf/_gyro_buf
            lists this class already used, under a lock. This is NOT this
            project's own concurrency (the "no threads before P6" rule in
            the architecture doc is about pyslam's OWN pipeline threading
            model -- tracking/mapping/optimiser -- not the vendor SDK's
            I/O thread, which exists either way); __iter__ itself stays
            single-threaded and synchronous. Falls back to "synced" with
            a logged warning if opening the sensor callback fails (older
            firmware / some RSUSB-backend configurations).
        """
        try:
            import pyrealsense2 as rs
        except ImportError as e:
            raise RuntimeError(
                "pyrealsense2 is not importable. Install with `pip install pyrealsense2` "
                "or run env_probe.py to see what's missing."
            ) from e
        self._rs = rs
        self.width, self.height, self.fps = width, height, fps
        self.enable_imu = enable_imu
        self._playback = playback_path is not None
        self._imu_buf_lock = threading.Lock()
        self._imu_callback_active = False
        self._motion_sensor = None  # set only if imu_capture_mode="callback" succeeds
        self._accel_stream_profile_for_extrinsics = None  # WP-LIVE 3.5: set by
            # _start_imu_callback() when callback mode resolves its own
            # accel stream profile; used by the extrinsic query below in
            # preference to querying the (accel-less, in callback mode)
            # main pipeline profile.

        self.pipeline = rs.pipeline()
        cfg = rs.config()
        # WP-J3: request rgb8 directly instead of bgr8 -- the original
        # code requested bgr8 then did `bgr[:, :, ::-1].copy()` every
        # frame to flip channel order. That's a full-frame reversed-stride
        # copy on every frame for no reason: the D400 colour sensor
        # exposes rgb8 as a supported UYVY-derived output format directly,
        # so requesting it removes the flip+copy entirely. Falls back to
        # bgr8 (+ the original flip) if the connected unit/firmware
        # doesn't support rgb8 for this stream -- caught at pipeline
        # start, not assumed.
        self._color_format = rs.format.rgb8
        if playback_path is not None:
            if record_path is not None:
                raise ValueError("record_path and playback_path are mutually exclusive")
            cfg.enable_device_from_file(playback_path, repeat_playback=False)
        else:
            if serial:
                cfg.enable_device(serial)
            cfg.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
            cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
            if enable_imu and imu_capture_mode != "callback":
                # "callback" mode opens+starts the motion sensor itself,
                # separately, below -- do NOT also request it through the
                # main pipeline config, or the SDK will try to deliver
                # motion frames through both paths.
                accel_hz = _pick_motion_fps(rs.context().query_devices()[0], rs, rs.stream.accel) or 200
                gyro_hz = _pick_motion_fps(rs.context().query_devices()[0], rs, rs.stream.gyro) or 200
                cfg.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, accel_hz)
                cfg.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, gyro_hz)
                log.info(f"IMU streams requested at device-reported rates: accel={accel_hz}Hz gyro={gyro_hz}Hz")
            if record_path is not None:
                cfg.enable_record_to_file(record_path)

        try:
            profile = self.pipeline.start(cfg)
        except RuntimeError as e:
            if playback_path is None and self._color_format == rs.format.rgb8:
                log.warning(f"rgb8 color stream rejected by device/firmware ({e}); "
                            f"falling back to bgr8 (+ per-frame channel flip).")
                cfg.disable_stream(rs.stream.color)
                cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
                self._color_format = rs.format.bgr8
                profile = self.pipeline.start(cfg)
            else:
                raise

        if enable_imu and imu_capture_mode not in ("callback", "synced"):
            raise ValueError(f"imu_capture_mode must be 'callback' or 'synced', got {imu_capture_mode!r}")

        if enable_imu and not self._playback and imu_capture_mode == "callback":
            try:
                self._start_imu_callback(rs, profile.get_device())
            except Exception as e:
                log.warning(f"IMU callback capture failed to start ({e}); falling back to "
                            f"'synced' mode (motion samples drained from wait_for_frames() "
                            f"framesets only -- see imu_capture_mode's docstring for what "
                            f"this costs on a slower frontend). Restarting the pipeline with "
                            f"motion streams enabled on the main config.")
                try:
                    self.pipeline.stop()
                except Exception:
                    pass
                accel_hz = _pick_motion_fps(rs.context().query_devices()[0], rs, rs.stream.accel) or 200
                gyro_hz = _pick_motion_fps(rs.context().query_devices()[0], rs, rs.stream.gyro) or 200
                cfg.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, accel_hz)
                cfg.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, gyro_hz)
                profile = self.pipeline.start(cfg)
                self._imu_callback_active = False

        if self._playback:
            playback = profile.get_device().as_playback()
            playback.set_real_time(False)  # replay as fast as we can consume, drop nothing

        self.align = rs.align(rs.stream.color)

        # WP-T3: the real accel-to-color rotation, queried from the
        # device's own extrinsics rather than assumed. pipeline.py's
        # R_body_cam parameter previously defaulted to identity
        # everywhere (flagged in WP-P5.2 as "KNOWN WRONG for this
        # project's own D435i rig" -- found via a synthetic fixture, but
        # never actually replaced with a real hardware value). This is
        # that real value: v_body_physical = R_body_cam @ v_cam, where
        # "cam" is the colour-optical frame every pose in this codebase
        # is expressed in, and "body" is whatever physical frame the
        # IMU's own accel/gyro axes are reported in.
        self.R_body_cam: np.ndarray = np.eye(3)
        if enable_imu:
            try:
                # WP-LIVE 3.5 fix: prefer the accel stream profile
                # _start_imu_callback() already resolved (callback mode --
                # the default, and the only path where the main pipeline
                # profile has NO accel stream to query); fall back to the
                # main pipeline profile's accel stream (synced mode, or
                # callback mode's own fallback-to-synced path, both of
                # which DO put accel on the main profile).
                if self._accel_stream_profile_for_extrinsics is not None:
                    accel_sp = self._accel_stream_profile_for_extrinsics
                else:
                    accel_sp = profile.get_stream(rs.stream.accel).as_motion_stream_profile()
                color_sp = profile.get_stream(rs.stream.color).as_video_stream_profile()
                extr = accel_sp.get_extrinsics_to(color_sp)
                # rs2_extrinsics.rotation is a COLUMN-MAJOR 3x3 (per the
                # librealsense API docs), satisfying p_color = R @ p_accel
                # + t for a point p. We want the inverse rotation (accel
                # <- color, i.e. R_body_cam): for an orthogonal matrix
                # that's just the transpose.
                R_color_accel = np.array(extr.rotation, dtype=np.float64).reshape(3, 3, order="F")
                self.R_body_cam = R_color_accel.T
            except Exception as e:
                log.warning(f"Could not read IMU<->colour extrinsics from the device "
                            f"({e}); falling back to IDENTITY for R_body_cam. Gravity "
                            f"alignment (WP-T3) and the P5.2 tilt prior will be wrong on "
                            f"this rig until this is fixed -- see gravity_frame.py and "
                            f"pipeline.py's own docstrings on why identity is a known-bad "
                            f"default, not a safe one.")

        color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        rs_intr = color_stream.get_intrinsics()
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()

        # baseline: the D435i depth sensor exposes the true IR-pair stereo
        # baseline (in mm) directly via rs.option.stereo_baseline. An
        # earlier version derived "baseline" from the depth<->color
        # extrinsic translation, which is the ~15mm IR-to-RGB module
        # offset, not the ~50mm stereo baseline the noise model in
        # tests/synth/world.py (sigma_z ~= z^2*sigma_d/(f*b)) assumes --
        # using it would understate depth noise by roughly 3x. Fall back
        # to the nominal 0.0499m only if the option truly isn't exposed
        # (e.g. replaying an older recording), and say so loudly.
        baseline = None
        try:
            if depth_sensor.supports(rs.option.stereo_baseline):
                baseline = depth_sensor.get_option(rs.option.stereo_baseline) / 1000.0  # mm -> m
        except Exception:
            baseline = None
        if baseline is None or baseline < 1e-4:
            baseline = 0.0499
            log.warning("Could not read rs.option.stereo_baseline from the device; "
                        "falling back to the nominal D435i baseline (0.0499m). "
                        "Depth noise modelling downstream may be off if your unit differs.")

        self.intr = Intrinsics(
            fx=rs_intr.fx, fy=rs_intr.fy, cx=rs_intr.ppx, cy=rs_intr.ppy,
            width=rs_intr.width, height=rs_intr.height,
            depth_scale=self.depth_scale, baseline=baseline,
        )
        log.info(f"RealSense started ({'playback' if self._playback else 'live'}): {self.intr}")

        self._imu_buffer: list[tuple[float, np.ndarray]] = []  # (t, [gx,gy,gz] or accel)
        self._accel_buf: list[tuple[float, np.ndarray]] = []
        self._gyro_buf: list[tuple[float, np.ndarray]] = []
        self._prev_t: Optional[float] = None
        self._frame_id = 0

    def intrinsics(self) -> Intrinsics:
        return self.intr

    def imu_to_color_rotation(self) -> np.ndarray:
        """WP-T3: R_body_cam, queried from the device (or identity with a
        logged warning if that query failed/IMU disabled) -- see the
        constructor's own comment. Pass this straight into
        Pipeline(..., R_body_cam=...)."""
        return self.R_body_cam.copy()

    def _start_imu_callback(self, rs, device) -> None:
        """WP-J3: open the motion sensor directly and register an
        SDK-managed callback thread, independent of wait_for_frames()'s
        cadence. Raises on any failure -- the caller (constructor) is
        responsible for the 'synced' fallback."""
        motion_sensor = None
        for sensor in device.query_sensors():
            stream_types = {p.stream_type() for p in sensor.get_stream_profiles()}
            if rs.stream.accel in stream_types and rs.stream.gyro in stream_types:
                motion_sensor = sensor
                break
        if motion_sensor is None:
            raise RuntimeError("no sensor on this device exposes both accel and gyro streams")

        accel_hz = _pick_motion_fps(device, rs, rs.stream.accel) or 200
        gyro_hz = _pick_motion_fps(device, rs, rs.stream.gyro) or 200
        accel_profile = next((p for p in motion_sensor.get_stream_profiles()
                               if p.stream_type() == rs.stream.accel and p.fps() == accel_hz), None)
        gyro_profile = next((p for p in motion_sensor.get_stream_profiles()
                              if p.stream_type() == rs.stream.gyro and p.fps() == gyro_hz), None)
        if accel_profile is None or gyro_profile is None:
            raise RuntimeError(f"could not resolve accel/gyro stream profiles at "
                                f"accel={accel_hz}Hz gyro={gyro_hz}Hz")

        def _callback(frame) -> None:
            # Runs on librealsense's own SDK thread -- keep this minimal
            # (append under lock, nothing else) so it never becomes a
            # source of dropped samples itself.
            if not frame.is_motion_frame():
                return
            mf = frame.as_motion_frame()
            t = mf.get_timestamp() / 1000.0
            data = mf.get_motion_data()
            v = np.array([data.x, data.y, data.z], dtype=np.float64)
            stream_type = mf.get_profile().stream_type()
            with self._imu_buf_lock:
                if stream_type == rs.stream.accel:
                    self._accel_buf.append((t, v))
                elif stream_type == rs.stream.gyro:
                    self._gyro_buf.append((t, v))

        motion_sensor.open([accel_profile, gyro_profile])
        motion_sensor.start(_callback)
        self._motion_sensor = motion_sensor
        self._imu_callback_active = True
        # WP-LIVE 3.5 fix: stash the RESOLVED accel stream profile so the
        # extrinsic query below can use it directly. Previously the
        # extrinsic query always called profile.get_stream(rs.stream.accel)
        # on the MAIN PIPELINE profile -- which has no accel stream at all
        # in callback mode (accel/gyro are opened on motion_sensor
        # directly, deliberately NOT requested on the main pipeline
        # config; see this constructor's own comment above). That query
        # threw every time under the DEFAULT imu_capture_mode="callback",
        # silently falling back to IDENTITY for R_body_cam on exactly the
        # capture path this project actually recommends (README_ORIN.md
        # Part 6 defaults to callback mode) -- gravity alignment (WP-T3)
        # and the P5.2 tilt prior were therefore wrong on real hardware
        # runs using the default settings, the "known-bad, not a safe
        # default" identity fallback silently doing its job with nobody
        # aware it had engaged. See tests/gates/test_g_live.py's
        # extrinsic-resolution regression check.
        self._accel_stream_profile_for_extrinsics = accel_profile
        log.info(f"IMU callback capture active (accel={accel_hz}Hz gyro={gyro_hz}Hz, "
                 f"independent of frame-capture cadence).")

    def _drain_motion(self, frames) -> None:
        """'synced' mode only: pull motion frames bundled into a
        wait_for_frames() frameset. In 'callback' mode this is never
        called -- samples already arrive via _start_imu_callback's own
        callback thread."""
        rs = self._rs
        for f in frames:
            if f.is_motion_frame():
                mf = f.as_motion_frame()
                t = mf.get_timestamp() / 1000.0
                data = mf.get_motion_data()
                v = np.array([data.x, data.y, data.z], dtype=np.float64)
                if mf.get_profile().stream_type() == rs.stream.accel:
                    self._accel_buf.append((t, v))
                elif mf.get_profile().stream_type() == rs.stream.gyro:
                    self._gyro_buf.append((t, v))

    def _pair_imu_since(self, t_prev: float, t_cur: float) -> np.ndarray:
        """Build the (N,7) [t,gx,gy,gz,ax,ay,az] array for gyro samples in
        (t_prev, t_cur], with accel linearly interpolated onto gyro times.
        WP-J3: buffers are shared with the IMU callback thread in
        'callback' mode, so every read/prune here is lock-protected."""
        with self._imu_buf_lock:
            gyro = [(t, v) for (t, v) in self._gyro_buf if t_prev < t <= t_cur]
            if not gyro or len(self._accel_buf) < 2:
                self._gyro_buf = [(t, v) for (t, v) in self._gyro_buf if t > t_cur]
                return np.zeros((0, 7), dtype=np.float64)
            accel_t = np.array([t for t, _ in self._accel_buf])
            accel_v = np.array([v for _, v in self._accel_buf])
            rows = []
            for t, w in gyro:
                a = np.array([np.interp(t, accel_t, accel_v[:, k]) for k in range(3)])
                rows.append([t, *w, *a])
            # keep buffers bounded
            self._gyro_buf = [(t, v) for (t, v) in self._gyro_buf if t > t_cur]
            self._accel_buf = [(t, v) for (t, v) in self._accel_buf if t > t_prev - 0.05]
            return np.array(rows, dtype=np.float64)

    def __iter__(self) -> Iterator[Frame]:
        rs = self._rs
        while True:
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=5000)
            except RuntimeError:
                # playback exhausted (or a live device genuinely stalled
                # for 5s, which is itself worth surfacing rather than
                # hanging forever) -- either way, end the stream cleanly
                # so run_slam.py's try/finally still writes its outputs.
                if self._playback:
                    log.info("Playback reached end of file.")
                return
            if self.enable_imu and not self._imu_callback_active:
                self._drain_motion(frames)  # 'synced' mode only -- in 'callback'
                    # mode, samples already arrive via the SDK callback thread
                    # (see _start_imu_callback); draining here too would be
                    # redundant at best and racy against the lock at worst.
            aligned = self.align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                continue
            t = color_frame.get_timestamp() / 1000.0

            raw = np.asanyarray(color_frame.get_data())
            if self._color_format == rs.format.rgb8:
                rgb = raw.copy()  # already RGB order -- no flip needed (WP-J3)
            else:
                rgb = raw[:, :, ::-1].copy()  # bgr8 fallback: flip to RGB
            depth = np.asanyarray(depth_frame.get_data()).copy()  # uint16, raw units

            imu = None
            if self.enable_imu and self._prev_t is not None:
                imu = self._pair_imu_since(self._prev_t, t)
            self._prev_t = t

            yield Frame(t=t, rgb=rgb, depth=depth, intr=self.intr, imu=imu,
                        frame_id=self._frame_id)
            self._frame_id += 1

    def close(self) -> None:
        if self._motion_sensor is not None:
            # WP-J3: stop+close the directly-opened motion sensor BEFORE
            # stopping the main pipeline -- the callback thread holds
            # self._imu_buf_lock briefly on every sample, and tearing down
            # the main pipeline first (which can itself briefly block on
            # SDK-internal state) has no defined ordering guarantee
            # against a still-running motion callback otherwise.
            try:
                self._motion_sensor.stop()
            except Exception:
                pass
            try:
                self._motion_sensor.close()
            except Exception:
                pass
            self._motion_sensor = None
            self._imu_callback_active = False
        try:
            self.pipeline.stop()
        except Exception:
            pass
