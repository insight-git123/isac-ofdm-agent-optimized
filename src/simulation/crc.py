"""CRC 编解码 (3GPP TS 38.212 §5.1)。

支持 NR 使用的 CRC-6/11/24A/24B。
"""
import numpy as np


# CRC 多项式 (位表示, 省略最高位)
CRC_POLYNOMIALS = {
    "CRC6":   (6,  [1, 0, 0, 0, 0, 1, 1]),                     # x^6 + x^5 + 1
    "CRC11":  (11, [1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1]),      # 完整多项式
    "CRC24A": (24, [1,1,0,0,0,0,1,1,0,0,1,0,0,1,1,0,0,1,1,1,1,1,0,1,1]),
    "CRC24B": (24, [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,1,1]),
    "CRC24C": (24, [1,1,0,1,1,0,0,1,0,1,0,1,1,0,0,0,1,0,0,0,1,0,1,1,1]),
}


def _polynomial_to_int(poly_bits: list) -> int:
    """多项式位列表 → 整数。"""
    val = 0
    for b in poly_bits:
        val = (val << 1) | int(b)
    return val


def crc_encode(bits: np.ndarray, crc_type: str = "CRC11") -> np.ndarray:
    """计算 CRC，返回附加 CRC 后的比特序列。

    原理: 用多项式除法计算余数，附加到数据末尾。

    Args:
        bits: 输入比特 (0/1)
        crc_type: "CRC6" / "CRC11" / "CRC24A" / "CRC24B" / "CRC24C"

    Returns:
        bits_with_crc: 长度 = len(bits) + crc_length
    """
    assert crc_type in CRC_POLYNOMIALS, f"未知 CRC: {crc_type}"
    crc_len, poly_bits = CRC_POLYNOMIALS[crc_type]
    poly_int = _polynomial_to_int(poly_bits)

    bits = np.asarray(bits, dtype=np.int8)
    K = len(bits)

    # 数据左移 crc_len 位
    msg = np.concatenate([bits, np.zeros(crc_len, dtype=np.int8)])

    # 用 GF(2) 除法求余数
    # 转成整数操作太慢, 用位数组实现
    msg_list = list(msg)
    for i in range(K):
        if msg_list[i] == 1:
            for j in range(crc_len + 1):
                msg_list[i + j] ^= poly_bits[j]

    crc_bits = np.array(msg_list[K:], dtype=np.int8)
    return np.concatenate([bits, crc_bits])


def crc_check(bits_with_crc: np.ndarray, crc_type: str = "CRC11") -> bool:
    """校验带 CRC 的比特序列。

    Args:
        bits_with_crc: 长度 = K + crc_len
        crc_type: CRC 类型

    Returns:
        True 表示 CRC 通过
    """
    assert crc_type in CRC_POLYNOMIALS
    crc_len, poly_bits = CRC_POLYNOMIALS[crc_type]
    K = len(bits_with_crc) - crc_len

    msg_list = list(np.asarray(bits_with_crc, dtype=np.int8))
    for i in range(K):
        if msg_list[i] == 1:
            for j in range(crc_len + 1):
                msg_list[i + j] ^= poly_bits[j]

    remainder = msg_list[K:]
    return all(b == 0 for b in remainder)


def remove_crc(bits_with_crc: np.ndarray, crc_type: str = "CRC11") -> np.ndarray:
    """去掉末尾的 CRC 位。"""
    crc_len, _ = CRC_POLYNOMIALS[crc_type]
    return bits_with_crc[:-crc_len]