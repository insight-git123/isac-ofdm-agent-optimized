"""CP-OFDM 全链路单元测试。"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.simulation.cp_ofdm_link import CPOFDMLink, CPOFDMLinkOversampled
from src.simulation.time_domain_channel import TimeDomainTDLChannel


def test_link_no_channel_roundtrip():
    """无信道时, 收发应完全一致。"""
    link = CPOFDMLink(n_fft=64, cp_len=8)
    X = (np.random.randn(64, 4) + 1j * np.random.randn(64, 4)) / np.sqrt(2)
    tx = link.transmit(X)
    rx = link.receive(tx, n_symbols=4)
    assert np.allclose(X, rx, atol=1e-10)


def test_link_transmit_length():
    link = CPOFDMLink(n_fft=64, cp_len=8)
    X = np.ones((64, 5), dtype=complex)
    tx = link.transmit(X)
    assert len(tx) == 5 * (64 + 8)


def test_link_cp_content():
    """CP 应为最后 cp_len 个采样的副本。"""
    link = CPOFDMLink(n_fft=64, cp_len=8)
    X = np.random.randn(64, 1) + 1j * np.random.randn(64, 1)
    tx = link.transmit(X)
    time_data = np.fft.ifft(X[:, 0]) * np.sqrt(64)
    assert np.allclose(tx[:8], time_data[-8:])


def test_channel_frequency_response_fft():
    """FIR 信道的频域响应应等于 FFT(fir)。"""
    ch = TimeDomainTDLChannel(model="TDL-A", rms_delay_ns=30,
                               fs_hz=122.88e6)
    H = ch.frequency_response(n_fft=1024)
    assert len(H) == 1024
    H_direct = np.fft.fft(ch.fir, 1024)
    assert np.allclose(H, H_direct)


def test_channel_matches_tdl_frequency_response():
    """时域 FIR 的频响应与 tdl_model 的频响在**全频段**一致。

    P4.1 修复: 使用相同的 FFT 频率轴 (fftfreq 顺序)。
    预期相关系数 > 0.95 (剩余差异来自时延量化到 8.14 ns 采样间隔)。
    """
    from src.simulation.tdl_model import tdl_frequency_response
    fs = 122.88e6
    n_fft = 1024
    scs_hz = 120e3

    ch = TimeDomainTDLChannel(model="TDL-A", rms_delay_ns=30,
                               fs_hz=fs, normalize_power=True)
    H_time = ch.frequency_response(n_fft)

    H_freq = tdl_frequency_response(n_fft, scs_hz, model="TDL-A",
                                     rms_delay_ns=30,
                                     normalize_power=True).flatten()

    # 全频段对比
    corr = np.corrcoef(np.abs(H_time), np.abs(H_freq))[0, 1]
    # 0.95 阈值: 量化误差上限 (离散 FIR vs 连续时延)
    # 修复频率轴前: 0.42; 修复后: > 0.95
    assert corr > 0.95, f"全频段相关系数 {corr} < 0.95"

    # 相位也应匹配 (低频段)
    phase_diff = np.angle(H_time[:100] * np.conj(H_freq[:100]))
    phase_diff_centered = phase_diff - np.median(phase_diff)
        # 相位误差来自时延量化 (采样间隔 8.14 ns → 最大半采样误差 4 ns)
    # 低频段 100 子载波 (12 MHz) 的相位误差上限 ≈ 2π·12e6·4e-9 ≈ 0.3 rad
    max_phase_err = np.max(np.abs(phase_diff_centered))
    assert max_phase_err < 0.3, \
        f"相位误差过大: max={max_phase_err:.4f} (预期 < 0.3 rad)"


def test_channel_apply_preserves_length():
    ch = TimeDomainTDLChannel(model="TDL-A", rms_delay_ns=30,
                               fs_hz=122.88e6)
    signal = np.random.randn(1000) + 1j * np.random.randn(1000)
    out = ch.apply(signal)
    assert len(out) == len(signal)


def test_oversampled_link_roundtrip():
    """过采样链路回环测试。"""
    link = CPOFDMLinkOversampled(n_fft=128, n_active_sc=64, cp_len=16)
    X = (np.random.randn(64, 3) + 1j * np.random.randn(64, 3)) / np.sqrt(2)
    tx = link.transmit(X)
    rx = link.receive(tx, n_symbols=3)
    assert rx.shape == (64, 3)
    assert np.allclose(X, rx, atol=1e-10)


def test_oversampled_transmit_length():
    link = CPOFDMLinkOversampled(n_fft=128, n_active_sc=64, cp_len=16)
    X = np.ones((64, 4), dtype=complex)
    tx = link.transmit(X)
    assert len(tx) == 4 * (128 + 16)


def test_cp_longer_than_channel_no_isi():
    """CP 长度 >= 信道时延 → 时域信道等效为频域乘法。"""
    fs = 122.88e6
    scs_hz = 120e3
    n_fft = 1024
    cp_samples = 72

    link = CPOFDMLink(n_fft=n_fft, cp_len=cp_samples)
    ch = TimeDomainTDLChannel(model="TDL-A", rms_delay_ns=30, fs_hz=fs)

    X = (np.random.randn(n_fft, 4) + 1j * np.random.randn(n_fft, 4)) / np.sqrt(2)

    tx = link.transmit(X)
    rx = ch.apply(tx)
    X_rx = link.receive(rx, n_symbols=4)

    H_theory = ch.frequency_response(n_fft)
    X_expected = X * H_theory.reshape(-1, 1)

    err = np.mean(np.abs(X_rx[:, 1:-1] - X_expected[:, 1:-1]) ** 2) / \
          np.mean(np.abs(X_expected) ** 2)
    assert err < 0.01, f"CP 长度不足导致误差 {err}"


def test_cp_shorter_than_channel_increases_isi():
    """CP 长度不足时，误差应显著大于 CP 充足的场景。"""
    from src.simulation.modulation import modulate
    fs = 122.88e6
    n_fft = 1024

    ch = TimeDomainTDLChannel(model="TDL-A", rms_delay_ns=30, fs_hz=fs)
    H_theory = ch.frequency_response(n_fft)

    rng = np.random.default_rng(42)
    qpsk_flat = modulate(n_fft * 4, "qpsk", rng=rng)
    X = qpsk_flat.reshape(n_fft, 4)
    X_expected = X * H_theory.reshape(-1, 1)

    def measure_error(cp_samples):
        link = CPOFDMLink(n_fft=n_fft, cp_len=cp_samples)
        tx = link.transmit(X)
        rx = ch.apply(tx)
        X_rx = link.receive(rx, n_symbols=4)
        return float(np.mean(np.abs(X_rx[:, 1:3] - X_expected[:, 1:3]) ** 2) /
                     np.mean(np.abs(X_expected) ** 2))

    err_long = measure_error(72)
    err_short = measure_error(4)

    print(f"\n  CP=72 (充足): err = {err_long:.4e}")
    print(f"  CP=4 (过短): err = {err_short:.4e}")
    print(f"  误差比 = {err_short / err_long:.1f}×")

    assert err_short > 10 * err_long, \
        f"CP 缩短后误差未显著增大: long={err_long:.4e}, short={err_short:.4e}"


def test_fir_maximum_delay():
    """FIR 滤波器最大时延应与 TDL-A 标准一致。"""
    ch = TimeDomainTDLChannel(model="TDL-A", rms_delay_ns=30, fs_hz=122.88e6)
    expected_samples = int(round(289.76e-9 * 122.88e6))
    assert ch.maximum_delay_samples() == expected_samples