"""调制方案单元测试。"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.simulation.modulation import (
    qpsk_modulate, qam16_modulate, qam64_modulate, qam256_modulate, modulate
)


def test_qpsk_average_power():
    """QPSK 归一化平均功率 = 1。"""
    rng = np.random.default_rng(42)
    sym = qpsk_modulate(10000, rng)
    assert abs(np.mean(np.abs(sym) ** 2) - 1.0) < 0.01


def test_qam16_average_power():
    rng = np.random.default_rng(42)
    sym = qam16_modulate(10000, rng)
    assert abs(np.mean(np.abs(sym) ** 2) - 1.0) < 0.01


def test_qam64_average_power():
    rng = np.random.default_rng(42)
    sym = qam64_modulate(10000, rng)
    assert abs(np.mean(np.abs(sym) ** 2) - 1.0) < 0.01


def test_qam256_average_power():
    rng = np.random.default_rng(42)
    sym = qam256_modulate(10000, rng)
    assert abs(np.mean(np.abs(sym) ** 2) - 1.0) < 0.01


def test_qpsk_constellation():
    """QPSK 只有 4 个星座点。"""
    rng = np.random.default_rng(42)
    sym = qpsk_modulate(10000, rng)
    unique_pts = np.unique(np.round(sym, 4))
    assert len(unique_pts) == 4


def test_qam16_constellation():
    """16QAM 有 16 个星座点。"""
    rng = np.random.default_rng(42)
    sym = qam16_modulate(10000, rng)
    unique_pts = np.unique(np.round(sym, 4))
    assert len(unique_pts) == 16


def test_modulate_dispatcher():
    """统一入口应能正确分派。

    注意: 256QAM 星座点功率范围大 (0.01~2.65), 1000 样本的均值
    统计波动可能达 ±0.05。用 10000 样本 (波动 ÷ √10) + 容差 0.05。
    """
    for scheme in ["qpsk", "16qam", "64qam", "256qam"]:
        sym = modulate(10000, scheme)   # ★ 从 1000 改到 10000
        assert len(sym) == 10000
        assert abs(np.mean(np.abs(sym) ** 2) - 1.0) < 0.05