"""严格时域脉冲压缩：基于 FFT 的匹配滤波实现。

实现要点:
1. 补零至 N_fft >= Nx + Ny - 1，避免循环卷积混叠
2. 匹配滤波器频响 H = conj(FFT(tx))
3. 输出 y_compressed = IFFT(FFT(rx) * H)
4. 可选汉明窗加权，旁瓣抑制 ~-42.7 dB
5. 输出为时域压缩后的一维距离像
"""
import numpy as np


def _next_pow2(n: int) -> int:
    """返回 >= n 的最小 2 的幂次。"""
    p = 1
    while p < n:
        p *= 2
    return p


def matched_filter_fft(tx: np.ndarray, rx: np.ndarray,
                       window: str = None) -> np.ndarray:
    """频域匹配滤波（脉冲压缩）。

    参数:
        tx: 发射参考信号 (1D complex array)
        rx: 接收信号 (1D complex array)
        window: None | 'hamming' | 'hann' | 'blackman'

    返回:
        compressed: 时域压缩结果，长度 = Nx + Ny - 1
    """
    Nx, Ny = len(tx), len(rx)
    N_fft = _next_pow2(Nx + Ny - 1)

    # 匹配滤波器设计（可选加窗）
    if window is not None:
        if window == 'hamming':
            w = np.hamming(Nx)
        elif window == 'hann':
            w = np.hanning(Nx)
        elif window == 'blackman':
            w = np.blackman(Nx)
        else:
            raise ValueError(f"未知窗类型: {window}")
        tx_ref = tx * w
    else:
        tx_ref = tx

    TX = np.fft.fft(tx_ref, N_fft)
    RX = np.fft.fft(rx, N_fft)
    H = np.conj(TX)

    y = np.fft.ifft(RX * H)
    return y[:Nx + Ny - 1]


def generate_time_domain_echo(tx: np.ndarray, targets: list,
                              fs: float, fc: float) -> np.ndarray:
    """从时域信号生成多目标回波（时延 + 多普勒 + 幅度衰减）。

    参数:
        tx: 发射时域信号
        targets: [{"range": m, "velocity": m/s, "rcs": float}, ...]
        fs: 采样率 (Hz)
        fc: 载波频率 (Hz)

    返回:
        rx: 接收时域信号，与 tx 等长
    """
    c = 3e8
    N = len(tx)
    t = np.arange(N) / fs
    rx = np.zeros(N, dtype=complex)

    for tgt in targets:
        tau = 2 * tgt["range"] / c
        fd = 2 * tgt["velocity"] * fc / c
        amp = np.sqrt(tgt.get("rcs", 1.0))

        delay_samples = int(round(tau * fs))

        # 正向时延 (不用 np.roll, 避免循环混叠)
        if delay_samples > 0:
            if delay_samples >= N:
                continue
            shifted = np.concatenate([
                np.zeros(delay_samples, dtype=complex),
                tx[:-delay_samples]
            ])
        else:
            shifted = tx

        doppler_phase = np.exp(1j * 2 * np.pi * fd * t)
        rx += amp * shifted * doppler_phase

    return rx


def add_awgn(signal: np.ndarray, snr_db: float) -> np.ndarray:
    """按指定 SNR 加高斯白噪声。"""
    sig_pow = np.mean(np.abs(signal) ** 2)
    noise_pow = sig_pow / (10 ** (snr_db / 10))
    noise = (np.random.randn(*signal.shape) +
             1j * np.random.randn(*signal.shape)) * np.sqrt(noise_pow / 2)
    return signal + noise


def range_axis_from_compression(compressed_len: int, fs: float) -> np.ndarray:
    """计算一维距离像的距离轴。

    距离分辨率 dr = c / (2 * fs)，因为匹配滤波把时域采样点映射到距离门。
    """
    c = 3e8
    dr = c / (2 * fs)
    return np.arange(compressed_len) * dr


def find_peaks_1d(profile: np.ndarray, min_height_ratio: float = 0.3) -> list:
    """一维距离像的简单峰值检测。

    返回峰值索引列表。
    """
    mag = np.abs(profile)
    threshold = mag.max() * min_height_ratio
    peaks = []
    for i in range(1, len(mag) - 1):
        if mag[i] > threshold and mag[i] > mag[i-1] and mag[i] > mag[i+1]:
            peaks.append(i)
    return peaks