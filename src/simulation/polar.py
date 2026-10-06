"""Polar 码：编码 + SC 译码。

用于 PBCH/PDCCH 等控制信道 (3GPP NR 采用 Polar)。
本实现为教学/研究简化版，参数与 NR 一致但非完全标准。

Reference: Arikan, E. (2009). "Channel polarization..."
"""
import numpy as np


def bhattacharyya_params(n: int, design_snr_db: float = 0.0) -> np.ndarray:
    """计算 N=2^n 个子信道的 Bhattacharyya 参数 (BEC 近似)。

    递归: z_{i+1}^{2i} = 2*z_i - z_i^2, z_{i+1}^{2i+1} = z_i^2
    """
    # 用设计 SNR 映射到初始 BEC 参数
    sigma = 10 ** (-design_snr_db / 20)
    z_0 = float(np.exp(-1 / (2 * sigma**2)))
    z_0 = min(0.999, max(0.001, z_0))

    z = np.array([z_0])
    for _ in range(n):
        z_new = np.zeros(2 * len(z))
        z_new[0::2] = 2 * z - z**2      # minus 分支
        z_new[1::2] = z**2               # plus 分支
        z = z_new
    return z


def polar_transform(u: np.ndarray) -> np.ndarray:
    """Polar 变换: x = u * F^⊗n, F = [[1,0],[1,1]]。

    复杂度 O(N log N)。
    """
    u = np.array(u, dtype=int)
    N = len(u)
    n = int(np.log2(N))

    x = u.copy()
    for stage in range(n):
        step = 2 ** (stage + 1)
        half = step // 2
        for i in range(0, N, step):
            left = x[i:i + half].copy()
            right = x[i + half:i + step].copy()
            x[i:i + half] = (left + right) % 2
    return x


def polar_encode(info_bits: np.ndarray, N: int,
                 design_snr_db: float = 0.0,
                 crc_type: str = None) -> tuple:
    """Polar 编码。

    Args:
        crc_type: None = 无 CRC (SC 用); "CRC11" 等 = 加 CRC (SCL 用)
    """
    from src.simulation.polar_sequence import get_frozen_set

    n = int(np.log2(N))
    assert 2**n == N
    K = len(info_bits)

    if crc_type is None:
        bits_payload = info_bits
    else:
        from src.simulation.crc import crc_encode
        bits_payload = crc_encode(info_bits, crc_type)

    K_payload = len(bits_payload)
    frozen_mask = get_frozen_set(N, K_payload)

    u = np.zeros(N, dtype=int)
    info_positions = np.where(~frozen_mask)[0]
    u[info_positions[:K_payload]] = bits_payload

    x = polar_transform(u)
    return x, frozen_mask, crc_type


# ============================================================
# SC 译码 (递归版)
# ============================================================
def _f_llr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Min-sum 近似: sign(a)*sign(b)*min(|a|,|b|)。"""
    sign = np.sign(a) * np.sign(b)
    sign = np.where(sign == 0, 1.0, sign)
    return sign * np.minimum(np.abs(a), np.abs(b))


def _g_llr(a: np.ndarray, b: np.ndarray, u: np.ndarray) -> np.ndarray:
    """g 函数: b + (1-2u)*a。"""
    return b + (1 - 2 * u.astype(float)) * a


def _sc_recursive(llr: np.ndarray, frozen: np.ndarray) -> np.ndarray:
    """递归 SC 译码。

    关键: g 函数需要左子块的 partial sum (x_left = F·u_left),
    而不是 u 域判决本身。
    """
    n = len(llr)

    if n == 1:
        if frozen[0]:
            return np.array([0], dtype=np.int8)
        return np.array([0 if llr[0] > 0 else 1], dtype=np.int8)

    half = n // 2

    # 左半: f 函数
    llr_left = _f_llr(llr[:half], llr[half:])
    u_left = _sc_recursive(llr_left, frozen[:half])

    # ★ partial sum: u_left → x_left
    x_left = polar_transform(u_left)

    # 右半: g 函数 (用 x_left)
    llr_right = _g_llr(llr[:half], llr[half:], x_left)
    u_right = _sc_recursive(llr_right, frozen[half:])

    return np.concatenate([u_left, u_right])


def polar_decode(llr: np.ndarray, frozen_mask: np.ndarray,
                 info_bit_count: int,
                 crc_type: str = None,
                 use_sc: bool = True) -> np.ndarray:
    """Polar SC 译码。"""
    from src.simulation.crc import remove_crc, CRC_POLYNOMIALS

    llr = np.asarray(llr, dtype=float)
    frozen_mask = np.asarray(frozen_mask, dtype=bool)

    u_hat = _sc_recursive(llr, frozen_mask)
    info_positions = np.where(~frozen_mask)[0]

    if crc_type is None:
        return u_hat[info_positions[:info_bit_count]]

    crc_len, _ = CRC_POLYNOMIALS[crc_type]
    K_with_crc = info_bit_count + crc_len
    bits_with_crc = u_hat[info_positions[:K_with_crc]]
    return remove_crc(bits_with_crc, crc_type)[:info_bit_count]