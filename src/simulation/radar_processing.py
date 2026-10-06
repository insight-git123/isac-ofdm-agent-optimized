"""Task3: 频域多目标 ISAC 雷达模型 + RDM + Swerling-I + 标准 TDL 多径。

P1 更新: 多径模型统一调用 tdl_model.tdl_frequency_response()，
        不再使用自定义三径。
"""
import numpy as np
from src.simulation.tdl_model import tdl_frequency_response


class RadarSimulator:
    def __init__(self, scs_khz: float, fft_size: int = 1024,
                 num_symbols: int = 64, fc_ghz: float = 3.5,
                  modulation: str = "qpsk"):
        self.fft_size = fft_size
        self.num_symbols = num_symbols
        self.fc_ghz = fc_ghz
        self.c = 3e8
        self.scs = scs_khz * 1e3           # Hz
        self.bandwidth = fft_size * self.scs
        self.symbol_duration = 1 / self.scs
        self.modulation = modulation

    def generate_echo(self, targets: list, snr_db: float = 20,
                      use_swerling: bool = False,
                      use_multipath: bool = False,
                      tdl_model: str = "TDL-A",
                      rms_delay_ns: float = None):
        """频域多目标回波，含时延/多普勒/RCS/Swerling/TDL 多径。

        Args:
            use_multipath: 为 True 时，用标准 TDL 模型给每个目标加多径
            tdl_model: TDL-A / TDL-B / TDL-C / TDL-D / TDL-E
        """
              # 使用指定调制方案生成频域数据
        from src.simulation.modulation import modulate
        X = np.zeros((self.fft_size, self.num_symbols), dtype=complex)
        for l in range(self.num_symbols):
            X[:, l] = modulate(self.fft_size, getattr(self, 'modulation', 'qpsk'))

        k = np.arange(self.fft_size).reshape(-1, 1)     # 子载波索引
        l = np.arange(self.num_symbols).reshape(1, -1)  # 符号索引

        # 每个目标的 TDL 频域响应（对每个目标独立）
        if use_multipath:
            H_tdl = tdl_frequency_response(
                self.fft_size, self.scs, model=tdl_model,
                rms_delay_ns=rms_delay_ns, normalize_power=True
            )   # (fft_size, 1)
        else:
            H_tdl = np.ones((self.fft_size, 1), dtype=complex)

        Y = np.zeros_like(X)
        for tgt in targets:
            tau = 2 * tgt["range"] / self.c
            fd = 2 * tgt["velocity"] * self.fc_ghz * 1e9 / self.c
            mean_rcs = tgt.get("rcs", 1.0)

            if use_swerling:
                rcs_scalar = np.random.exponential(scale=mean_rcs)
                amp = np.sqrt(rcs_scalar)
            else:
                amp = np.sqrt(mean_rcs)

            phase_range = -2 * np.pi * k * self.scs * tau
            phase_doppler = 2 * np.pi * l * self.symbol_duration * fd

            # 对每个目标施加 TDL 频域响应
            Y += amp * X * np.exp(1j * (phase_range + phase_doppler)) * H_tdl

        sig_pow = np.mean(np.abs(Y) ** 2)
        noise_pow = sig_pow / (10 ** (snr_db / 10))
        Y += (np.random.randn(*Y.shape) + 1j * np.random.randn(*Y.shape)) * np.sqrt(noise_pow / 2)

        return X, Y

    def compute_rdm(self, X, Y):
        """信道估计 + 2D-FFT 得到距离-多普勒图。"""
        H = Y * np.conj(X) / (np.abs(X) ** 2 + 1e-12)
        range_profile = np.fft.ifft(H, axis=0)
        rdm = np.fft.fftshift(np.fft.fft(range_profile, axis=1), axes=1)
        rdm_mag = np.abs(rdm)

        range_axis = np.arange(self.fft_size) * self.c / (2 * self.bandwidth)
        doppler_freq = np.fft.fftshift(np.fft.fftfreq(
            self.num_symbols, d=self.symbol_duration))
        velocity_axis = doppler_freq * self.c / (2 * self.fc_ghz * 1e9)

        return rdm_mag, range_axis, velocity_axis