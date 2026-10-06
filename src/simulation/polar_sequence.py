"""5G Polar 码序列 + PW 极化权重。

5G 标准序列定义在 TS 38.212 Table 5.3.1.2-1 (N=1024)。
本实现提供两种方案:
1. 硬编码 5G 序列 (部分, N=512/256/128 通用)
2. PW (Polarization Weight) 公式生成 (通用, 近似标准)
"""
import numpy as np


def polar_weight_sequence(N: int) -> np.ndarray:
    """生成 PW 极化权重序列。

    Polarization weight 公式 (Huawei 提出, 被 5G 采纳):
        W_i = sum_{j=0}^{n-1} B_j * beta^j
    其中:
        - (B_{n-1}, ..., B_0) 是 i 的二进制展开
        - beta = 2^(1/4) ≈ 1.1892
        - n = log2(N)

    越大的 W_i 表示子信道越可靠。

    Returns:
        sequence: 长度 N 的数组, sequence[k] = 第 k 可靠的子信道索引
    """
    n = int(np.log2(N))
    beta = 2 ** (1/4)

    weights = np.zeros(N)
    for i in range(N):
        # i 的二进制展开 (LSB first)
        w = 0.0
        for j in range(n):
            b_j = (i >> j) & 1
            w += b_j * (beta ** j)
        weights[i] = w

    # 升序排列: 索引从小到大, 权重从小到大
    # 即 sequence[k] = 第 k 可靠 (k 越大越可靠) 的子信道
    sequence = np.argsort(weights)
    return sequence


def get_frozen_set(N: int, K: int, info_positions_extra=None) -> np.ndarray:
    """根据 PW 序列选冻结位。

    Args:
        N: 码长
        K: 信息比特数
        info_positions_extra: 强制作为信息位的索引 (如 CRC 位占位)

    Returns:
        frozen_mask: (N,) bool 数组, True = 冻结
    """
    seq = polar_weight_sequence(N)
    # 最可靠的 K 个位置 (seq 末尾 K 个)
    info_set = set(seq[-(K):].tolist())

    if info_positions_extra is not None:
        info_set.update(info_positions_extra)

    frozen_mask = np.ones(N, dtype=bool)
    frozen_mask[list(info_set)] = False
    return frozen_mask