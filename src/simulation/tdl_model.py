"""3GPP TR 38.901 标准 TDL 信道模型。

参数来源: 3GPP TR 38.901 V17.0.0 (2022-03) Table 7.7.2-1 至 7.7.2-5。

支持:
    TDL-A: 23 抽头, NLOS, RMS 时延扩展 30ns (标准值)
    TDL-B: 23 抽头, NLOS, RMS 时延扩展 100ns
    TDL-C: 24 抽头, NLOS, RMS 时延扩展 300ns
    TDL-D: 13 抽头, LOS (Rician K=13.3dB)
    TDL-E: 14 抽头, LOS (Rician K=22.0dB)
"""
import numpy as np


# ============================================================
# TDL-A: Table 7.7.2-1, 23 taps, NLOS
# ============================================================
TDL_A_TAPS = [
    # (normalized_delay, power_dB)
    (0.0000, -13.4), (0.3819,   0.0), (0.4025,  -2.2), (0.5868,  -4.0),
    (0.4610,  -6.0), (0.5375,  -8.2), (0.6708,  -9.9), (0.5750, -10.5),
    (0.7618,  -7.5), (1.5375, -15.9), (1.8978,  -6.6), (2.2242, -16.7),
    (2.1718, -12.4), (2.4942, -15.2), (2.5119, -10.8), (3.0582, -11.3),
    (4.0810, -12.7), (4.4579, -16.2), (4.5695, -18.3), (4.7966, -18.9),
    (5.0066, -16.6), (5.3043, -19.9), (9.6586, -29.7),
]

# ============================================================
# TDL-B: Table 7.7.2-2, 23 taps, NLOS
# ============================================================
TDL_B_TAPS = [
    (0.0000,   0.0), (0.1072,  -2.2), (0.2155,  -4.0), (0.2095,  -3.2),
    (0.2870,  -9.8), (0.2986,  -1.2), (0.3752,  -3.4), (0.5055,  -5.2),
    (0.3681,  -7.6), (0.3697,  -3.0), (0.5700,  -8.9), (0.5283,  -9.0),
    (1.1021,  -10.4), (1.2756, -11.2), (1.5474, -15.6), (1.7842, -14.3),
    (2.0169, -18.3), (2.8294, -20.6), (3.0219, -18.2), (3.6187, -21.2),
    (4.1067, -20.1), (4.2790, -21.9), (4.7834, -22.8),
]

# ============================================================
# TDL-C: Table 7.7.2-3, 24 taps, NLOS
# ============================================================
TDL_C_TAPS = [
    (0.0000,  -4.4), (0.2099,  -1.2), (0.2219,  -3.5), (0.2329,  -5.2),
    (0.2176,  -2.5), (0.6366,   0.0), (0.6448,  -2.2), (0.6560,  -3.9),
    (0.6584,  -7.4), (0.7935,  -7.1), (0.8213, -10.7), (0.9336, -11.1),
    (1.2285,  -7.1), (1.3083, -10.8), (2.1704,  -8.5), (2.7105, -13.1),
    (4.2589,  -14.9), (4.6003,  -4.8), (5.4902,  -6.1), (5.6077,  -9.9),
    (6.3065,  -9.3), (6.6374, -12.0), (7.0427, -14.2), (8.6523, -17.1),
]

# ============================================================
# TDL-D: Table 7.7.2-4, 13 taps, LOS (Rician K=13.3 dB)
# 抽头 1 是 LOS 主径
# ============================================================
TDL_D_TAPS = [
    (0.0000,  -0.2), (0.0000, -13.5), (0.0359, -18.8), (0.0364, -21.0),
    (0.0505, -22.8), (0.0673, -17.9), (0.0872, -16.1), (0.1194, -20.8),
    (0.1737, -18.3), (0.2065, -20.3), (0.2511, -22.8), (0.3187, -25.0),
    (0.3512, -22.3),
]

# ============================================================
# TDL-E: Table 7.7.2-5, 14 taps, LOS (Rician K=22.0 dB)
# ============================================================
TDL_E_TAPS = [
    (0.0000,  -0.03), (0.0000, -22.03), (0.0359, -15.8), (0.0364, -18.0),
    (0.0505, -19.8), (0.0673, -14.9), (0.0872, -13.1), (0.1194, -17.8),
    (0.1737, -15.3), (0.2065, -17.3), (0.2511, -19.8), (0.3187, -22.0),
    (0.3512, -19.3), (0.4024, -24.0),
]

# ============================================================
# 模型配置
# ============================================================
TDL_CONFIGS = {
    "TDL-A": {"taps": TDL_A_TAPS, "rms_delay_ns": 30,  "los": False, "k_factor_db": None},
    "TDL-B": {"taps": TDL_B_TAPS, "rms_delay_ns": 100, "los": False, "k_factor_db": None},
    "TDL-C": {"taps": TDL_C_TAPS, "rms_delay_ns": 300, "los": False, "k_factor_db": None},
    "TDL-D": {"taps": TDL_D_TAPS, "rms_delay_ns": 30,  "los": True,  "k_factor_db": 13.3},
    "TDL-E": {"taps": TDL_E_TAPS, "rms_delay_ns": 30,  "los": True,  "k_factor_db": 22.0},
}


def get_tdl_taps(model: str = "TDL-A", rms_delay_ns: float = None):
    """返回 (时延_ns, 功率_dB) 列表。

    Args:
        model: TDL-A/B/C/D/E
        rms_delay_ns: 覆盖默认的 RMS 时延扩展 (ns)；None 表示使用标准值
    """
    if model not in TDL_CONFIGS:
        raise ValueError(f"未知 TDL 模型: {model}. 可选: {list(TDL_CONFIGS.keys())}")

    cfg = TDL_CONFIGS[model]
    rms = rms_delay_ns if rms_delay_ns is not None else cfg["rms_delay_ns"]

    taps = []
    for norm_delay, power_db in cfg["taps"]:
        actual_delay_ns = norm_delay * rms
        taps.append((actual_delay_ns, power_db))
    return taps, cfg


def apply_tdl_channel(rx: np.ndarray, fs: float, model: str = "TDL-A",
                      rms_delay_ns: float = None,
                      normalize_power: bool = True) -> np.ndarray:
    """时域 TDL 信道 (抽头时延 + 功率加权)。"""
    taps, _ = get_tdl_taps(model, rms_delay_ns)

    powers_linear = 10 ** (np.array([p for _, p in taps]) / 10)
    if normalize_power:
        powers_linear = powers_linear / np.sum(powers_linear)
    amps = np.sqrt(powers_linear)
    delays_samples = [int(round(d * 1e-9 * fs)) for d, _ in taps]

    N = len(rx)
    out = np.zeros(N, dtype=complex)
    for amp, ds in zip(amps, delays_samples):
        if ds >= N:
            continue
        if ds > 0:
            out[ds:] += amp * rx[:-ds]
        else:
            out += amp * rx
    return out


def tdl_frequency_response(fft_size: int, scs_hz: float, model: str = "TDL-A",
                           rms_delay_ns: float = None,
                           normalize_power: bool = True) -> np.ndarray:
    """TDL 信道的频域响应 H[k]，维度 = (fft_size, 1)。

    ★ P4.1 修复: FFT 频率轴必须与 OFDM 子载波索引对齐。

    数学推导:
      OFDM 的 IFFT: x[n] = (1/sqrt(N)) * sum_k X[k] * exp(j*2*pi*k*n/N)
      所以 X[k] 对应的基带频率是 fftfreq(N) * fs
      其中 fs = N * scs_hz
      即 f_k = [0, SCS, ..., (N/2-1)*SCS, -N/2*SCS, ..., -SCS]

    物理: H(f) = sum_p amp_p * exp(-j*2*pi*f*tau_p)
    """
    taps, _ = get_tdl_taps(model, rms_delay_ns)

    powers_linear = 10 ** (np.array([p for _, p in taps]) / 10)
    if normalize_power:
        powers_linear = powers_linear / np.sum(powers_linear)
    amps = np.sqrt(powers_linear)

    # ★ 用 fftfreq 生成正确排序的子载波频率轴 (不加 ifftshift)
    freq_axis = np.fft.fftfreq(
        fft_size, d=1.0 / (fft_size * scs_hz)
    ).reshape(-1, 1)

    H = np.zeros((fft_size, 1), dtype=complex)
    for amp, delay_ns in zip(amps, [d for d, _ in taps]):
        tau = delay_ns * 1e-9
        H += amp * np.exp(-1j * 2 * np.pi * freq_axis * tau)
    return H


def print_tdl_summary(model: str = "TDL-A"):
    """打印 TDL 模型的抽头摘要。"""
    taps, cfg = get_tdl_taps(model)
    print(f"  TDL 模型: {model}")
    print(f"    RMS 时延扩展: {cfg['rms_delay_ns']} ns")
    print(f"    LOS: {cfg['los']}"
          + (f", Rician K = {cfg['k_factor_db']} dB" if cfg['los'] else ""))
    print(f"    抽头数: {len(taps)}")
    max_delay = max(d for d, _ in taps)
    print(f"    最大时延: {max_delay:.1f} ns (距离扩展 {max_delay*1e-9*3e8/2:.1f} m)")