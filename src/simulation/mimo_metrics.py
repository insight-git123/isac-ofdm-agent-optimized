"""MIMO 感知误差指标：角度/距离/速度 RMSE。

用于定量对比 MIMO 与 SISO 的感知精度。
"""
import numpy as np

from src.simulation.radar_processing_mimo import MIMORadarSimulator


def estimate_peak_3d(cube, range_axis, velocity_axis, angle_axis,
                     target, r_win=5, v_win=3, a_win=3):
    """在 3D cube 里找目标附近的峰值。

    Args:
        cube: (n_range, n_doppler, n_angle)
        target: {"range", "velocity", "az_deg"}

    Returns:
        (peak_range, peak_velocity, peak_angle) 或 None
    """
    r_idx = np.argmin(np.abs(range_axis - target["range"]))
    v_idx = np.argmin(np.abs(velocity_axis - target["velocity"]))
    a_idx = np.argmin(np.abs(angle_axis - target["az_deg"]))

    r_lo = max(0, r_idx - r_win)
    r_hi = min(cube.shape[0], r_idx + r_win + 1)
    v_lo = max(0, v_idx - v_win)
    v_hi = min(cube.shape[1], v_idx + v_win + 1)
    a_lo = max(0, a_idx - a_win)
    a_hi = min(cube.shape[2], a_idx + a_win + 1)

    sub = cube[r_lo:r_hi, v_lo:v_hi, a_lo:a_hi]
    if sub.size == 0:
        return None

    peak = np.unravel_index(np.argmax(sub), sub.shape)
    return (range_axis[r_lo:r_hi][peak[0]],
            velocity_axis[v_lo:v_hi][peak[1]],
            angle_axis[a_lo:a_hi][peak[2]])


def run_mimo_mc(scs_khz: float, targets: list, n_antennas: int = 8,
                snr_db: float = 20, n_trials: int = 20,
                seed: int = 42, rms_delay_ns: float = 5.0):
    """运行 MIMO 蒙特卡洛，返回 RMSE 指标。"""
    num_sym = 512
    r_errors, v_errors, a_errors = [], [], []
    n_detected = 0

    for trial in range(n_trials):
        np.random.seed(seed + trial)

        sim = MIMORadarSimulator(
            scs_khz=scs_khz, fft_size=1024, num_symbols=num_sym,
            fc_ghz=3.5, n_antennas=n_antennas, angle_fft_size=64
        )
        X, Y_mimo = sim.generate_echo_mimo(
            targets=targets, snr_db=snr_db,
            use_swerling=True, use_multipath=True,
            tdl_model="TDL-A", rms_delay_ns=rms_delay_ns,
        )
        cube, range_axis, velocity_axis, angle_axis = sim.compute_3d_cube(X, Y_mimo)

        for tgt in targets:
            peak = estimate_peak_3d(cube, range_axis, velocity_axis,
                                    angle_axis, tgt)
            if peak is None:
                continue
            pr, pv, pa = peak
            r_errors.append(pr - tgt["range"])
            v_errors.append(pv - tgt["velocity"])
            a_errors.append(pa - tgt["az_deg"])
            n_detected += 1

    def rmse(errors):
        if not errors:
            return 0.0
        return float(np.sqrt(np.mean(np.array(errors) ** 2)))

    return {
        "n_antennas": n_antennas,
        "n_trials": n_trials,
        "n_detections": n_detected,
        "range_rmse_m": round(rmse(r_errors), 3),
        "velocity_rmse_ms": round(rmse(v_errors), 3),
        "angle_rmse_deg": round(rmse(a_errors), 3),
    }