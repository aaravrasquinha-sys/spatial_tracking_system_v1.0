"""Re-run the same accuracy simulation against the PATCHED code."""
import numpy as np
from poi_perception.contracts import Intrinsics
from poi_localization.config import M4Config
from poi_localization.frames.types import FloorFrameTransform
from poi_localization.measurement.types import Measurement
from poi_localization import geometry
from poi_localization.tracking.track_manager import TrackManager, DetectionMeasurements
from poi_localization.measurement.raycast_measurement import compute_raycast_measurement
from poi_localization.measurement.depth_measurement import compute_depth_measurement

intr = Intrinsics(fx=606.75, fy=606.57, cx=320.19, cy=237.06, width=640, height=480, depth_scale=0.001, baseline=0.05)
H, PITCH = 2.3, 25.0
T_true = FloorFrameTransform.from_height_and_yaw(H, PITCH, 0.0)

def project(p_floor, T):
    R,t = T.R, T.t
    p_cam = R.T @ (np.asarray(p_floor,float) - t)
    if p_cam[2] <= 0.05: return None, None
    u = intr.fx*p_cam[0]/p_cam[2] + intr.cx
    v = intr.fy*p_cam[1]/p_cam[2] + intr.cy
    return (u,v), p_cam

def run_walk(extrinsic_pitch_err_deg=0.0, extrinsic_h_err_m=0.0, sigma_rot_deg=0.0, sigma_trans_m=0.0, gait=True, seed=7):
    rng = np.random.default_rng(seed)
    cfg = M4Config.default()
    T_est = FloorFrameTransform.from_height_and_yaw(
        H+extrinsic_h_err_m, PITCH+extrinsic_pitch_err_deg, 0.0,
        sigma_rot_rad=np.radians(sigma_rot_deg), sigma_trans_m=sigma_trans_m,
    )
    tm = TrackManager(cfg)
    dt = 1/30.
    errs, nees, srcs = [], [], []
    for k in range(300):
        t = k*dt
        x = 1.0 + 1.3*t
        y = 0.6
        if x > 5.5: break
        gp = np.array([x,y,0.0])
        torso = np.array([x, y, 1.25])
        phase = np.sin(2*np.pi*0.9*t)
        both_ankles = rng.random() > 0.25
        if gait and not both_ankles:
            ank = gp + np.array([0.35*phase, 0.0, 0.0])
        else:
            ank = gp.copy()
        px_t, pcam_t = project(torso, T_true)
        if px_t is None: continue
        z_true = pcam_t[2]
        z_front = z_true - 0.12
        z_meas = z_front + rng.normal(0, 0.015*z_true + 0.005) + 0.008*z_true
        # shoulders (facing camera -> use facing thickness, matches sim's 0.12 offset)
        sh_l_px, _ = project(torso + np.array([0,-0.2,0]), T_true)
        sh_r_px, _ = project(torso + np.array([0, 0.2,0]), T_true)
        u_noisy, v_noisy = px_t[0]+rng.normal(0,1.5), px_t[1]+rng.normal(0,1.5)
        p_cam_meas = geometry.deproject_pixel(u_noisy, v_noisy, z_meas, intr)
        p_fl = T_est.apply_point(p_cam_meas)
        cam_xy = T_est.camera_origin_floor()[:2]
        vdir = p_fl[:2]-cam_xy; vdir/=np.linalg.norm(vdir)
        thickness = cfg.depth_measurement.body_thickness_m  # facing-on scenario
        p_fl[:2] += vdir*thickness
        f_depth = cfg.depth_measurement.depth_stereo_fx_px
        sz_prec = z_meas**2*cfg.depth_measurement.sigma_disparity_px/(f_depth*intr.baseline)
        sz_bias = z_meas*cfg.depth_measurement.depth_bias_frac
        sz = float(np.hypot(sz_prec, sz_bias))
        ray_cam = p_cam_meas/np.linalg.norm(p_cam_meas)
        hf = np.linalg.norm(T_est.apply_direction(ray_cam)[:2])
        cov = geometry.radial_tangential_to_xy_cov(sz*hf, z_meas*cfg.depth_measurement.lateral_px_noise/intr.fx, vdir)
        range_m = float(np.linalg.norm(p_fl[:2]-cam_xy))
        extra = geometry.extrinsic_position_variance(range_m, T_est.sigma_rot_rad, T_est.sigma_trans_m)
        cov = geometry.add_isotropic_variance(cov, extra)
        dm = Measurement((float(p_fl[0]),float(p_fl[1])), cov, "depth") if z_meas<=cfg.depth_measurement.max_range_m else None
        px_a,_ = project(ank, T_true)
        rm = None
        if px_a is not None:
            noisy = (px_a[0]+rng.normal(0,2.0), px_a[1]+rng.normal(0,2.0))
            rm = compute_raycast_measurement(noisy, "ankles" if both_ankles else "ankle_single",
                                             intr, T_est, cfg.raycast_measurement)
        if dm is None and rm is None: continue
        det = DetectionMeasurements(1, 0.9, (0,0,10,10), dm, rm, 1.7)
        outs,_ = tm.update(t, dt, [det])
        for o in outs:
            if o.state!="confirmed": continue
            e = np.array(o.position_xy)-gp[:2]
            errs.append(np.linalg.norm(e)); srcs.append(o.src)
            nees.append(geometry.mahalanobis_sq(e, geometry.cov_upper_to_matrix(o.cov_xy)))
    return np.array(errs), np.array(nees), srcs

from collections import Counter
print(f"{'scenario':45} {'median cm':>10} {'p90 cm':>8} {'max cm':>8} {'NEES med':>9}  src mix")
for label,kw in [
    ("perfect extrinsics, no gait", dict(gait=False)),
    ("perfect extrinsics, gait", dict(gait=True)),
    ("1.0deg pitch err, NO sigma reported (old-style)", dict(extrinsic_pitch_err_deg=1.0, sigma_rot_deg=0.0)),
    ("1.0deg pitch err, sigma_rot=1.0deg reported (Phase A honest)", dict(extrinsic_pitch_err_deg=1.0, sigma_rot_deg=1.0, sigma_trans_m=0.02)),
    ("2cm h err+1deg pitch, sigma reported", dict(extrinsic_pitch_err_deg=1.0, extrinsic_h_err_m=0.02, sigma_rot_deg=1.0, sigma_trans_m=0.02)),
]:
    e,n,s = run_walk(**kw)
    print(f"{label:45} {np.median(e)*100:10.1f} {np.percentile(e,90)*100:8.1f} {e.max()*100:8.1f} {np.median(n):9.2f}  {dict(Counter(s))}")
