"""标准 SCL 译码器（递归实现，正确性优先）。

Reference: Tal & Vardy (2015). "List decoding of polar codes."
"""
import numpy as np

from src.simulation.polar import _f_llr, _g_llr, polar_transform


def _scl_recursive(llr: np.ndarray, frozen: np.ndarray,
                   L: int) -> list:
    """递归 SCL 译码。

    返回 [(u_hat, path_metric), ...]，按 metric 升序，最多 L 条。

    复杂度: O((L+1)^log N)。N=64, L=8 约 50 万次操作。
    """
    n = len(llr)

    if n == 1:
        if frozen[0]:
            return [(np.array([0], dtype=np.int8), 0.0)]
        llr_val = float(llr[0])
        pm0 = max(0.0, -llr_val)
        pm1 = max(0.0, llr_val)
        return [
            (np.array([0], dtype=np.int8), pm0),
            (np.array([1], dtype=np.int8), pm1),
        ]

    half = n // 2

    # 左子树 (递归一次)
    llr_left = _f_llr(llr[:half], llr[half:])
    left_paths = _scl_recursive(llr_left, frozen[:half], L)

    # 对每条左路径展开右子树
    all_paths = []
    for u_left, m_left in left_paths:
        x_left = polar_transform(u_left)
        llr_right = _g_llr(llr[:half], llr[half:], x_left)
        right_paths = _scl_recursive(llr_right, frozen[half:], L)
        for u_right, m_right in right_paths:
            all_paths.append(
                (np.concatenate([u_left, u_right]), m_left + m_right)
            )

    all_paths.sort(key=lambda p: p[1])
    return all_paths[:L]


class SCLDecoder:
    """标准 SCL 译码器 (递归实现)。"""

    def __init__(self, N: int, L: int = 8):
        self.N = N
        self.n = int(np.log2(N))
        self.L = L

    def decode(self, llr: np.ndarray,
               frozen_mask: np.ndarray) -> np.ndarray:
        """返回最佳路径的 u_hat。"""
        llr = np.asarray(llr, dtype=float)
        paths = _scl_recursive(llr, frozen_mask, self.L)
        return paths[0][0]

    def decode_with_list(self, llr: np.ndarray,
                         frozen_mask: np.ndarray) -> list:
        """返回 top-L 路径的 u_hat 列表 (按 PM 升序)。"""
        llr = np.asarray(llr, dtype=float)
        paths = _scl_recursive(llr, frozen_mask, self.L)
        return [u for u, _ in paths]


def scl_decode(llr: np.ndarray, frozen_mask: np.ndarray,
               info_bit_count: int, L: int = 8,
               crc_type: str = None) -> np.ndarray:
    """SCL 译码 (无 CRC 时返回最佳路径)。"""
    from src.simulation.crc import remove_crc, CRC_POLYNOMIALS

    N = len(llr)
    decoder = SCLDecoder(N, L)

    if crc_type is None:
        u_hat = decoder.decode(llr, frozen_mask)
        info_positions = np.where(~frozen_mask)[0]
        return u_hat[info_positions[:info_bit_count]]

    # 有 CRC → 用 top-L 路径检查
    top_paths = decoder.decode_with_list(llr, frozen_mask)
    crc_len, _ = CRC_POLYNOMIALS[crc_type]
    K_with_crc = info_bit_count + crc_len
    info_positions = np.where(~frozen_mask)[0]

    from src.simulation.crc import crc_check
    for u_hat in top_paths:
        bits_with_crc = u_hat[info_positions[:K_with_crc]]
        if crc_check(bits_with_crc, crc_type):
            return remove_crc(bits_with_crc, crc_type)[:info_bit_count]

    # 无路径通过 CRC → 返回最佳路径
    best_u = top_paths[0]
    bits_with_crc = best_u[info_positions[:K_with_crc]]
    return remove_crc(bits_with_crc, crc_type)[:info_bit_count]


def scl_decode_with_crc_selection(llr: np.ndarray,
                                  frozen_mask: np.ndarray,
                                  info_bit_count: int, L: int = 8,
                                  crc_type: str = "CRC11") -> np.ndarray:
    """SCL + CRC 辅助路径选择（强制要求 CRC）。"""
    if crc_type is None:
        raise ValueError("scl_decode_with_crc_selection 需要有效的 crc_type")
    return scl_decode(llr, frozen_mask, info_bit_count,
                      L=L, crc_type=crc_type)