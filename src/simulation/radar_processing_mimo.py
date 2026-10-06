"""MIMO 雷达处理：3D-FFT 距离-多普勒-角度。

在 Task3 频域模型基础上，加 MIMO 角度维。
"""
import numpy as np

from src.simulation.mimo_array import UniformLinearArray
from src.simulation.radar_processing import RadarSimulator


class MIMORadarSimulator(RadarSimulator):
    """带 MIMO 阵列的雷达仿真器。

    在 RadarSimulator 基础上增加:
    - N 元 ULA 天线阵列
    - 角度域 FFT
    - 3D-FFT 输出 (Range × Doppler × Angle)
    """
    def __init__(self, scs_khz: float, fft_size: int = 1024,
                 num_symbols: int = 64, fc_ghz: float = 3.5,
                 n_antennas: int = 8, angle_fft_size: int = 64):
        super().__init__(scs_khz, fft_size, num_symbols, fc_ghz)
        self.array = UniformLinearArray(n_elements=n_antennas, fc_ghz=fc_ghz)
        self.n_antennas = n_antennas
        self.angle_fft_size = angle_fft_size

    def generate_echo_mimo(self, targets: list, snr_db: float = 20,
                           use_swerling: bool = False,
                           use_multipath: bool = False,
                           tdl_model: str = "TDL-A",
                           rms_delay_ns: float = 5.0):
        """生成 MIMO 接收信号。

        Returns:
            X: (fft_size, num_symbols) 发射频域导频
            Y_mimo: (n_antennas, fft_size, num_symbols) 接收信号
        """
        # 1. 生成基础 SISO 频域回波 (复用父类)
        X, Y_siso = super().generate_echo(
            targets=targets, snr_db=snr_db,
            use_swerling=use_swerling,
            use_multipath=use_multipath,
            tdl_model=tdl_model,
            rms_delay_ns=rms_delay_ns,
        )

        # 2. 对每个天线通道施加阵列响应
        Y_mimo = np.zeros((self.n_antennas, self.fft_size, self.num_symbols),
                          dtype=complex)

        for tgt in targets:
            az_deg = tgt.get("az_deg", 0.0)
            az_rad = np.deg2rad(az_deg)
            a = self.array.steering_vector(az_rad).flatten()  # (N,)
            # 单目标信号复制到每个天线 (乘以转向矢量元素)
            for n in range(self.n_antennas):
                Y_mimo[n] += a[n] * Y_siso / np.sqrt(len(targets))

        # 加噪 (每通道独立)
        sig_pow = np.mean(np.abs(Y_mimo) ** 2)
        noise_pow = sig_pow / (10 ** (snr_db / 10))
        Y_mimo += (np.random.randn(*Y_mimo.shape) +
                   1j * np.random.randn(*Y_mimo.shape)) * np.sqrt(noise_pow / 2)

        return X, Y_mimo

    def compute_3d_cube(self, X, Y_mimo):
        """3D-FFT: 距离 × 多普勒 × 角度。

        Returns:
            cube: (fft_size, num_symbols, angle_fft_size)
            range_axis, velocity_axis, angle_axis
        """
        # 1. 逐天线做距离-多普勒处理
        rdm_per_ant = []
        for n in range(self.n_antennas):
            H = Y_mimo[n] * np.conj(X) / (np.abs(X) ** 2 + 1e-12)
            range_profile = np.fft.ifft(H, axis=0)
            rdm = np.fft.fftshift(np.fft.fft(range_profile, axis=1), axes=1)
            rdm_per_ant.append(rdm)

        # rdm_per_ant: list of (fft_size, num_symbols)
        # 2. 沿天线维做角度 FFT
        antenna_stack = np.stack(rdm_per_ant, axis=-1)  # (fft_size, num_symbols, N)
        # Padding 到 angle_fft_size
        pad_size = self.angle_fft_size - self.n_antennas
        if pad_size > 0:
            antenna_stack = np.pad(antenna_stack,
                                    ((0, 0), (0, 0), (0, pad_size)),
                                    mode='constant')
        # 角度 FFT (沿天线维)
        cube = np.fft.fftshift(np.fft.fft(antenna_stack, axis=-1), axes=-1)
        cube = np.abs(cube)

        # 3. 坐标轴
        range_axis = np.arange(self.fft_size) * self.c / (2 * self.bandwidth)
        doppler_freq = np.fft.fftshift(np.fft.fftfreq(
            self.num_symbols, d=self.symbol_duration))
        velocity_axis = doppler_freq * self.c / (2 * self.fc_ghz * 1e9)
        angle_axis = self.array.angle_axis(self.angle_fft_size)

        return cube, range_axis, velocity_axis, angle_axis