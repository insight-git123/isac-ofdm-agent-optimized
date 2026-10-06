"""TDL 信道模型单元测试。"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.simulation.tdl_model import (
    get_tdl_taps, apply_tdl_channel, tdl_frequency_response,
    TDL_CONFIGS, TDL_A_TAPS
)


def test_tdl_a_tap_count_matches_standard():
    """TDL-A 应包含 23 个抽头（TR 38.901 Table 7.7.2-1）。"""
    taps, cfg = get_tdl_taps("TDL-A")
    assert len(taps) == 23
    assert cfg["rms_delay_ns"] == 30
    assert cfg["los"] is False


def test_tdl_b_tap_count():
    """TDL-B 应包含 23 个抽头。"""
    taps, _ = get_tdl_taps("TDL-B")
    assert len(taps) == 23


def test_tdl_c_tap_count():
    """TDL-C 应包含 24 个抽头。"""
    taps, _ = get_tdl_taps("TDL-C")
    assert len(taps) == 24


def test_tdl_d_is_los():
    """TDL-D 应为 LOS 场景，Rician K=13.3 dB。"""
    _, cfg = get_tdl_taps("TDL-D")
    assert cfg["los"] is True
    assert abs(cfg["k_factor_db"] - 13.3) < 0.1


def test_tdl_e_is_los():
    """TDL-E 应为 LOS 场景，Rician K=22.0 dB。"""
    _, cfg = get_tdl_taps("TDL-E")
    assert cfg["los"] is True
    assert abs(cfg["k_factor_db"] - 22.0) < 0.1


def test_tdl_a_first_tap_delay_is_zero():
    """TDL-A 第一个抽头时延应为 0。"""
    taps, _ = get_tdl_taps("TDL-A")
    assert taps[0][0] == 0.0


def test_tdl_a_max_delay():
    """TDL-A 最大时延应为 9.6586 × 30 ns ≈ 289.76 ns。"""
    taps, _ = get_tdl_taps("TDL-A")
    max_delay = max(d for d, _ in taps)
    assert abs(max_delay - 9.6586 * 30) < 0.1


def test_rms_delay_override():
    """RMS 时延扩展可覆盖。"""
    taps_default, _ = get_tdl_taps("TDL-A")
    taps_override, _ = get_tdl_taps("TDL-A", rms_delay_ns=100)
    # 100/30 ≈ 3.33 倍缩放
    ratio = taps_override[-1][0] / taps_default[-1][0]
    assert abs(ratio - 100/30) < 0.01


def test_apply_tdl_channel_preserves_length():
    fs = 100e6
    rx = np.random.randn(1000) + 1j * np.random.randn(1000)
    out = apply_tdl_channel(rx, fs=fs, model="TDL-A")
    assert len(out) == len(rx)


def test_apply_tdl_channel_energy_conservation():
    """归一化后总能量不应爆炸。"""
    fs = 100e6
    rx = np.random.randn(5000) + 1j * np.random.randn(5000)
    out = apply_tdl_channel(rx, fs=fs, model="TDL-A", normalize_power=True)
    assert np.sum(np.abs(out) ** 2) < 10 * np.sum(np.abs(rx) ** 2)


def test_frequency_response_shape():
    """频域响应形状应为 (fft_size, 1)。"""
    H = tdl_frequency_response(fft_size=1024, scs_hz=120e3, model="TDL-A")
    assert H.shape == (1024, 1)


def test_frequency_response_reasonable_power():
    """TDL-A 频响平均功率在合理范围。

    物理: 归一化后 Σ amp_p² = 1，但 TDL-A 的密集抽头
    (时延差 << 1/BW) 会导致相干叠加，频响功率 > 1。
    理论上界是 (Σ amp_p)²，实际取决于抽头间距。
    """
    H = tdl_frequency_response(fft_size=1024, scs_hz=120e3,
                               model="TDL-A", normalize_power=True)
    avg_power = np.mean(np.abs(H) ** 2)
    # 下限：独立叠加；上限：完全相干
    assert 0.5 < avg_power < 5.0, f"avg_power={avg_power:.4f}"

    # 形状和类型检查
    assert H.shape == (1024, 1)
    assert H.dtype == complex
    # 不应全零
    assert np.mean(np.abs(H) ** 2) > 0