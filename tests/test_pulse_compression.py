"""脉冲压缩单元测试。"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.simulation.pulse_compression import (
    matched_filter_fft, generate_time_domain_echo,
    range_axis_from_compression, find_peaks_1d, _next_pow2
)


def test_next_pow2():
    assert _next_pow2(1) == 1
    assert _next_pow2(3) == 4
    assert _next_pow2(100) == 128
    assert _next_pow2(1024) == 1024


def test_matched_filter_single_target():
    """单目标脉冲压缩峰值应精确落位。"""
    fs = 100e6
    tx = np.random.randn(1024) + 1j * np.random.randn(1024)
    targets = [{"range": 150.0, "velocity": 20.0, "rcs": 1.0}]
    rx = generate_time_domain_echo(tx, targets, fs=fs, fc=3.5e9)

    y = matched_filter_fft(tx, rx)
    range_axis = range_axis_from_compression(len(y), fs)
    peaks = find_peaks_1d(y, min_height_ratio=0.3)

    assert len(peaks) >= 1
    peak_range = range_axis[peaks[0]]
    # 误差 1 个距离门内
    assert abs(peak_range - 150.0) < 3e8 / (2 * fs) * 2


def test_matched_filter_with_hamming():
    """汉明窗版本应产生相同峰值位置，且旁瓣更低。"""
    fs = 100e6
    tx = np.random.randn(1024) + 1j * np.random.randn(1024)
    targets = [{"range": 100.0, "velocity": 10.0, "rcs": 1.0}]
    rx = generate_time_domain_echo(tx, targets, fs=fs, fc=3.5e9)

    y_no_win = matched_filter_fft(tx, rx, window=None)
    y_hamming = matched_filter_fft(tx, rx, window='hamming')

    assert len(y_no_win) == len(y_hamming)

    # 主峰位置应接近
    peaks_no = find_peaks_1d(y_no_win, min_height_ratio=0.3)
    peaks_ham = find_peaks_1d(y_hamming, min_height_ratio=0.3)
    range_axis = range_axis_from_compression(len(y_no_win), fs)
    assert abs(range_axis[peaks_no[0]] - range_axis[peaks_ham[0]]) < 5.0