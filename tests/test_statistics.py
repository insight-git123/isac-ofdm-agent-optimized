"""统计工具单元测试。"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.simulation.statistics import (
    bootstrap_ci, wilson_ci_no_scipy
)


def test_bootstrap_ci_mean():
    """Bootstrap 均值应接近样本均值。"""
    values = [2, 3, 2, 3, 3, 2, 2, 3, 3, 2]
    result = bootstrap_ci(values, n_bootstrap=1000)
    assert abs(result["mean"] - np.mean(values)) < 1e-6


def test_bootstrap_ci_contains_mean():
    """置信区间应包含样本均值。"""
    values = list(np.random.default_rng(42).integers(0, 4, size=50))
    result = bootstrap_ci(values, n_bootstrap=1000)
    assert result["ci_lo"] <= result["mean"] <= result["ci_hi"]


def test_bootstrap_ci_reproducible():
    """相同种子 → 相同结果。"""
    values = list(np.random.default_rng(42).integers(0, 4, size=50))
    r1 = bootstrap_ci(values, n_bootstrap=500, seed=42)
    r2 = bootstrap_ci(values, n_bootstrap=500, seed=42)
    assert r1["ci_lo"] == r2["ci_lo"]
    assert r1["ci_hi"] == r2["ci_hi"]


def test_bootstrap_ci_smaller_with_more_samples():
    """样本数越多, CI 越窄。"""
    rng = np.random.default_rng(42)
    small = list(rng.integers(0, 4, size=10))
    large = list(rng.integers(0, 4, size=1000))
    ci_small = bootstrap_ci(small, n_bootstrap=500)["ci_half_width"]
    ci_large = bootstrap_ci(large, n_bootstrap=500)["ci_half_width"]
    assert ci_large < ci_small


def test_bootstrap_ci_empty():
    """空输入应返回 0。"""
    result = bootstrap_ci([], n_bootstrap=100)
    assert result["mean"] == 0.0
    assert result["n_samples"] == 0


def test_wilson_ci_known_case():
    """已知案例: 50/100 的 95% Wilson 区间 ≈ [0.404, 0.596]。"""
    result = wilson_ci_no_scipy(50, 100, confidence=0.95)
    assert abs(result["p_hat"] - 0.5) < 1e-6
    assert 0.39 < result["ci_lo"] < 0.42
    assert 0.58 < result["ci_hi"] < 0.61


def test_wilson_ci_zero_successes():
    """0 成功时, 区间下界 = 0。"""
    result = wilson_ci_no_scipy(0, 100, confidence=0.95)
    assert result["p_hat"] == 0.0
    assert result["ci_lo"] == 0.0
    assert result["ci_hi"] > 0.0


def test_wilson_ci_all_successes():
    """全部成功时, 区间上界 = 1 (允许浮点精度)。"""
    result = wilson_ci_no_scipy(100, 100, confidence=0.95)
    assert np.isclose(result["p_hat"], 1.0)
    assert np.isclose(result["ci_hi"], 1.0, atol=1e-6)
    assert result["ci_lo"] < 1.0


def test_wilson_ci_narrower_with_more_samples():
    """样本数越多, Wilson CI 越窄。"""
    ci_10 = wilson_ci_no_scipy(5, 10)["ci_half_width"]
    ci_100 = wilson_ci_no_scipy(50, 100)["ci_half_width"]
    ci_1000 = wilson_ci_no_scipy(500, 1000)["ci_half_width"]
    assert ci_10 > ci_100 > ci_1000


def test_wilson_ci_zero_n():
    """n=0 应返回全零。"""
    result = wilson_ci_no_scipy(0, 0)
    assert result["p_hat"] == 0.0
    assert result["ci_half_width"] == 0.0