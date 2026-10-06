"""基准场景定义：AWGN vs 单径 vs TDL-A 多径。

用于回答: 不同信道条件对 ISAC 检测性能的影响有多大?
"""
import numpy as np

from src.simulation.radar_processing import RadarSimulator
from src.simulation.cfar import ca_cfar_2d


def _cluster_detections(rdm_mag, detected, range_axis, velocity_axis,
                        r_gate=3, v_gate=3):
    """NMS 聚类 (与 task4_run 保持一致)。"""
    rows, cols = np.where(detected)
    if len(rows) == 0:
        return [], [], []
    points = sorted(zip(rows, cols), key=lambda p: -rdm_mag[p[0], p[1]])
    clustered = []
    for (r, c) in points:
        if not any(abs(r - cr) <= r_gate and abs(c - cc) <= v_gate
                   for cr, cc in clustered):
            clustered.append((r, c))
    det_ranges = [range_axis[r] for r, _ in clustered]
    det_velocities = [velocity_axis[c] for _, c in clustered]
    det_mags = [rdm_mag[r, c] for r, c in clustered]
    return det_ranges, det_velocities, det_mags


# 场景配置表
SCENARIO_CONFIG = {
    "AWGN":        {"use_swerling": False, "use_multipath": False},
    "SinglePath":  {"use_swerling": True,  "use_multipath": False},
    "TDL-A":       {"use_swerling": True,  "use_multipath": True},
}


def run_scenario(scenario_name: str, scs_khz: float, targets: list,
                 snr_db: float, n_trials: int = 20, seed: int = 42,
                 rms_delay_ns: float = 5.0, num_sym: int = 512) -> dict:
    """在指定场景下运行 N 次 MC，返回统计指标。

    Args:
        scenario_name: "AWGN" / "SinglePath" / "TDL-A"
        scs_khz: 子载波间隔
        targets: 目标真值列表
        snr_db: 信噪比
        n_trials: MC 次数
        seed: 随机种子
        rms_delay_ns: TDL-A 的 RMS 时延扩展
        num_sym: OFDM 符号数

    Returns:
        dict: {scenario, snr_db, detection_rate, avg_hits,
               range_rmse_m, velocity_rmse_ms, n_trials}
    """
    if scenario_name not in SCENARIO_CONFIG:
        raise ValueError(f"未知场景: {scenario_name}. "
                         f"可选: {list(SCENARIO_CONFIG.keys())}")

    cfg = SCENARIO_CONFIG[scenario_name]

    hits_list = []
    range_errors = []
    velocity_errors = []

    for trial in range(n_trials):
        np.random.seed(seed + trial)

        sim = RadarSimulator(scs_khz=scs_khz, fft_size=1024,
                             num_symbols=num_sym, fc_ghz=3.5)

        X, Y = sim.generate_echo(
            targets=targets, snr_db=snr_db,
            use_swerling=cfg["use_swerling"],
            use_multipath=cfg["use_multipath"],
            tdl_model="TDL-A",
            rms_delay_ns=rms_delay_ns,
        )
        rdm_mag, range_axis, velocity_axis = sim.compute_rdm(X, Y)

        # CFAR + 峰值过滤
        rdm_power = np.abs(rdm_mag) ** 2
        detected_cfar = ca_cfar_2d(rdm_power, guard_cells=2,
                                    train_cells=4, pfa=1e-3)
        max_power = np.max(rdm_power)
        detected = detected_cfar & (rdm_power > max_power * 0.1)

        # NMS
        range_res = sim.c / (2 * sim.bandwidth)
        r_gate = max(3, int(round(20.0 / range_res)))
        det_ranges, det_velocities, _ = _cluster_detections(
            rdm_mag, detected, range_axis, velocity_axis,
            r_gate=r_gate, v_gate=3
        )

        # 匹配真值
        R_TOL = max(5.0, 2 * range_res)
        V_TOL = 20.0
        hits = 0
        for tgt in targets:
            best_r_err = None
            best_v_err = None
            for (r, v) in zip(det_ranges, det_velocities):
                if abs(r - tgt["range"]) <= R_TOL and \
                   abs(v - tgt["velocity"]) <= V_TOL:
                    hits += 1
                    best_r_err = abs(r - tgt["range"])
                    best_v_err = abs(v - tgt["velocity"])
                    break
            if best_r_err is not None:
                range_errors.append(best_r_err)
                velocity_errors.append(best_v_err)

        hits_list.append(hits)

    avg_hits = float(np.mean(hits_list))
    detection_rate = avg_hits / len(targets)

    if range_errors:
        range_rmse = float(np.sqrt(np.mean(np.array(range_errors) ** 2)))
        velocity_rmse = float(np.sqrt(np.mean(np.array(velocity_errors) ** 2)))
    else:
        range_rmse = 0.0
        velocity_rmse = 0.0

    return {
        "scenario": scenario_name,
        "snr_db": snr_db,
        "detection_rate": round(detection_rate, 3),
        "avg_hits": round(avg_hits, 2),
        "range_rmse_m": round(range_rmse, 3),
        "velocity_rmse_ms": round(velocity_rmse, 3),
        "n_trials": n_trials,
    }