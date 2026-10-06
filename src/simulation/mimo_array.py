"""MIMO 天线阵列：ULA 均匀线阵 + 角度响应。"""
import numpy as np


class UniformLinearArray:
    def __init__(self, n_elements: int = 8, fc_ghz: float = 3.5,
                 spacing_ratio: float = 0.5):
        self.n_elements = n_elements
        self.fc_ghz = fc_ghz
        self.c = 3e8
        self.wavelength = self.c / (fc_ghz * 1e9)
        self.spacing = spacing_ratio * self.wavelength

    def steering_vector(self, theta_rad: float) -> np.ndarray:
        n = np.arange(self.n_elements).reshape(-1, 1)
        return np.exp(1j * 2 * np.pi * self.spacing / self.wavelength
                      * np.sin(theta_rad) * n)

    def angle_axis(self, n_fft: int = 64) -> np.ndarray:
        """角度 FFT 对应的角度轴 (度)。

        ★ 正确公式: sin(θ) = k · λ / d
          其中 k = fftshift(fftfreq(n_fft)) ∈ [-0.5, 0.5]。
          d = λ/2 时覆盖 ±90°。
        """
        k = np.fft.fftshift(np.fft.fftfreq(n_fft, d=1.0))
        sin_theta = k * self.wavelength / self.spacing
        sin_theta = np.clip(sin_theta, -1.0, 1.0)
        return np.degrees(np.arcsin(sin_theta))


class UniformPlanarArray:
    def __init__(self, n_h: int = 4, n_v: int = 4, fc_ghz: float = 3.5,
                 spacing_ratio: float = 0.5):
        self.n_h = n_h
        self.n_v = n_v
        self.n_elements = n_h * n_v
        self.fc_ghz = fc_ghz
        self.c = 3e8
        self.wavelength = self.c / (fc_ghz * 1e9)
        self.spacing = spacing_ratio * self.wavelength

    def steering_vector(self, az_rad: float, el_rad: float) -> np.ndarray:
        m = np.arange(self.n_h).reshape(-1, 1)
        n = np.arange(self.n_v).reshape(1, -1)
        phase_h = 2 * np.pi * self.spacing / self.wavelength * np.sin(az_rad) * np.cos(el_rad)
        phase_v = 2 * np.pi * self.spacing / self.wavelength * np.sin(el_rad)
        a_h = np.exp(1j * phase_h * m)
        a_v = np.exp(1j * phase_v * n)
        a_2d = a_h @ a_v
        return a_2d.reshape(-1, 1)


def generate_multipath_angles(n_targets: int, targets: list,
                               n_paths: int = 3,
                               angle_spread_deg: float = 10.0,
                               rng: np.random.Generator = None) -> list:
    if rng is None:
        rng = np.random.default_rng(42)
    paths = []
    for i, tgt in enumerate(targets):
        az_los = tgt.get("az_deg", 0.0)
        for j in range(n_paths):
            if j == 0:
                paths.append({
                    "target_idx": i, "path_idx": j,
                    "atten_db": 0.0, "delay_ns": 0.0, "az_deg": az_los,
                })
            else:
                az_scat = az_los + rng.normal(0, angle_spread_deg)
                atten = rng.uniform(-10, -3)
                delay = rng.uniform(20, 150)
                paths.append({
                    "target_idx": i, "path_idx": j,
                    "atten_db": atten, "delay_ns": delay, "az_deg": az_scat,
                })
    return paths