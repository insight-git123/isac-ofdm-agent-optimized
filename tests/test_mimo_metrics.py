"""MIMO 误差指标单元测试。"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.simulation.mimo_metrics import estimate_peak_3d, run_mimo_mc


def test_estimate_peak_3d_finds_synthetic_target():
    """人造 cube 中的峰值应被正确找到。

    注意: peak 位置必须与 target 描述一致。
    - range=200m → range_axis[20] (因为 range_axis = arange(64)*10)
    - velocity=0 → velocity_axis[15] (linspace(-60,60,32) 最接近 0 的点)
    - az_deg=0 → angle_axis[8] (linspace(-90,90,16) 最接近 0 的点)
    """
    cube = np.zeros((64, 32, 16))
    cube[20, 15, 8] = 100.0    # ★ 峰值位置与 target 对齐

    range_axis = np.arange(64) * 10.0        # 0~630m
    velocity_axis = np.linspace(-60, 60, 32)
    angle_axis = np.linspace(-90, 90, 16)

    target = {"range": 200.0, "velocity": 0.0, "az_deg": 0.0}
    peak = estimate_peak_3d(cube, range_axis, velocity_axis, angle_axis, target)

    assert peak is not None
    # 峰值位置应在 target 附近
    assert abs(peak[0] - 200.0) < 30.0
    assert abs(peak[1] - 0.0) < 10.0
    assert abs(peak[2] - 0.0) < 15.0


def test_mimo_mc_returns_metrics():
    """MC 应返回所有必需字段。"""
    targets = [{"range": 150.0, "velocity": 30.0, "rcs": 1.0, "az_deg": 0.0}]
    res = run_mimo_mc(120, targets, n_antennas=4, n_trials=2, seed=42)
    for key in ["n_antennas", "range_rmse_m", "velocity_rmse_ms",
                "angle_rmse_deg", "n_detections"]:
        assert key in res


def test_mimo_mc_reproducible():
    """相同种子应返回相同结果。"""
    targets = [{"range": 150.0, "velocity": 30.0, "rcs": 1.0, "az_deg": 0.0}]
    r1 = run_mimo_mc(120, targets, n_antennas=4, n_trials=2, seed=42)
    r2 = run_mimo_mc(120, targets, n_antennas=4, n_trials=2, seed=42)
    assert r1["angle_rmse_deg"] == r2["angle_rmse_deg"]


def test_mimo_angle_error_reasonable():
    """角度误差应在合理范围 (8 元 ULA 理论角分辨率 ≈ 14.5°)。"""
    targets = [{"range": 200.0, "velocity": 0.0, "rcs": 1.0, "az_deg": 0.0}]
    res = run_mimo_mc(120, targets, n_antennas=8, n_trials=3, seed=42)
    # 角度 RMSE 应小于 2 倍 Rayleigh 分辨率
    assert res["angle_rmse_deg"] < 30.0