"""基准场景单元测试。"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.simulation.scenarios import run_scenario, SCENARIO_CONFIG


TARGETS = [
    {"range": 150.0, "velocity": 30.0, "rcs": 1.0},
    {"range": 300.0, "velocity": -20.0, "rcs": 0.5},
    {"range": 450.0, "velocity": 0.0, "rcs": 0.8},
]


def test_scenario_config_complete():
    """场景配置应包含 3 个标准场景。"""
    assert "AWGN" in SCENARIO_CONFIG
    assert "SinglePath" in SCENARIO_CONFIG
    assert "TDL-A" in SCENARIO_CONFIG


def test_awgn_high_snr_detects_all():
    """高 SNR 下 AWGN 场景应检测到全部目标。"""
    res = run_scenario("AWGN", 120, TARGETS, snr_db=30,
                       n_trials=3, seed=42)
    assert res["detection_rate"] >= 0.9


def test_low_snr_worse_than_high_snr():
    """低 SNR 检测率应 <= 高 SNR 检测率。"""
    res_low = run_scenario("AWGN", 120, TARGETS, snr_db=0,
                           n_trials=5, seed=42)
    res_high = run_scenario("AWGN", 120, TARGETS, snr_db=25,
                            n_trials=5, seed=42)
    assert res_low["detection_rate"] <= res_high["detection_rate"]


def test_scenario_returns_required_fields():
    """返回结果应包含所有必需字段。"""
    res = run_scenario("AWGN", 120, TARGETS, snr_db=20, n_trials=2, seed=42)
    required = ["scenario", "snr_db", "detection_rate", "avg_hits",
                "range_rmse_m", "velocity_rmse_ms", "n_trials"]
    for key in required:
        assert key in res, f"Missing key: {key}"


def test_scenario_reproducible():
    """相同种子应返回相同结果。"""
    r1 = run_scenario("AWGN", 120, TARGETS, snr_db=20, n_trials=2, seed=42)
    r2 = run_scenario("AWGN", 120, TARGETS, snr_db=20, n_trials=2, seed=42)
    assert r1["detection_rate"] == r2["detection_rate"]
    assert r1["range_rmse_m"] == r2["range_rmse_m"]


def test_tdl_a_scenario_runs():
    """TDL-A 场景应能正常运行。"""
    res = run_scenario("TDL-A", 120, TARGETS, snr_db=20,
                       n_trials=2, seed=42)
    assert res["scenario"] == "TDL-A"
    assert 0 <= res["detection_rate"] <= 1


def test_invalid_scenario_raises():
    """未知场景应报错。"""
    try:
        run_scenario("InvalidScenario", 120, TARGETS, snr_db=20,
                     n_trials=1, seed=42)
        assert False, "应抛出 ValueError"
    except ValueError:
        pass