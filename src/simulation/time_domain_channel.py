"""时域 TDL 信道：真实 FIR 卷积 + 频域等效验证。

与 src.simulation.tdl_model.tdl_frequency_response 的关系:
- 本文件: 时域 FIR 卷积 (物理真实)
- tdl_frequency_response: 频域直接计算
- 二者应等价 (CP 长度 >= 最大时延扩展时)
"""
import numpy as np

from src.simulation.tdl_model import get_tdl_taps


class TimeDomainTDLChannel:
    """时域 FIR TDL 信道。"""
    def __init__(self, model: str = "TDL-A", rms_delay_ns: float = 30.0,
                 fs_hz: float = 122.88e6, normalize_power: bool = True):
        self.model = model
        self.rms_delay_ns = rms_delay_ns
        self.fs = fs_hz
        self.normalize_power = normalize_power

        # 获取抽头
        taps, cfg = get_tdl_taps(model, rms_delay_ns)
        powers = 10 ** (np.array([p for _, p in taps]) / 10)
        if normalize_power:
            powers = powers / np.sum(powers)
        amps = np.sqrt(powers)

        # 构造 FIR 滤波器系数
        delays_samples = [int(round(d * 1e-9 * fs_hz)) for d, _ in taps]
        max_delay = max(delays_samples)
        self.fir = np.zeros(max_delay + 1, dtype=complex)
        for amp, delay in zip(amps, delays_samples):
            self.fir[delay] += amp

        self.tap_delays_ns = [d for d, _ in taps]
        self.tap_amps = amps

    def apply(self, signal: np.ndarray) -> np.ndarray:
        """时域卷积 (FIR 滤波)。

        使用 'full' 卷积, 截断到输入长度 (因果 FIR)。
        """
        return np.convolve(signal, self.fir, mode='full')[:len(signal)]

    def frequency_response(self, n_fft: int) -> np.ndarray:
        """频域响应 H[k]，k=0..n_fft-1。

        数学: H[k] = FFT(fir)[k]
        与 tdl_model.tdl_frequency_response 应一致 (相对误差 < 1e-6)。
        """
        return np.fft.fft(self.fir, n_fft)

    def maximum_delay_samples(self) -> int:
        return len(self.fir) - 1

    def maximum_delay_ns(self) -> float:
        return self.maximum_delay_samples() / self.fs * 1e9