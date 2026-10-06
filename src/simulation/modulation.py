"""NR 支持的调制方案：QPSK / 16QAM / 64QAM / 256QAM。

参考 3GPP TS 38.211 Section 5.1。
"""
import numpy as np


def qpsk_modulate(num_symbols: int, rng: np.random.Generator = None) -> np.ndarray:
    """QPSK 调制，归一化平均功率 = 1。

    星座: {±1±1j} / sqrt(2)
    """
    if rng is None:
        rng = np.random.default_rng()
    bits = rng.integers(0, 2, size=(num_symbols, 2))
    symbols = (1 - 2 * bits[:, 0]) + 1j * (1 - 2 * bits[:, 1])
    return symbols / np.sqrt(2)


def qam16_modulate(num_symbols: int, rng: np.random.Generator = None) -> np.ndarray:
    """16QAM 调制，归一化平均功率 = 1。

    星座: {±1, ±3} × {±1, ±3} / sqrt(10)
    """
    if rng is None:
        rng = np.random.default_rng()
    levels = np.array([-3, -1, 1, 3])
    idx_i = rng.integers(0, 4, size=num_symbols)
    idx_q = rng.integers(0, 4, size=num_symbols)
    symbols = levels[idx_i] + 1j * levels[idx_q]
    return symbols / np.sqrt(10)


def qam64_modulate(num_symbols: int, rng: np.random.Generator = None) -> np.ndarray:
    """64QAM 调制，归一化平均功率 = 1。

    星座: {±1, ±3, ±5, ±7} × {±1, ±3, ±5, ±7} / sqrt(42)
    """
    if rng is None:
        rng = np.random.default_rng()
    levels = np.array([-7, -5, -3, -1, 1, 3, 5, 7])
    idx_i = rng.integers(0, 8, size=num_symbols)
    idx_q = rng.integers(0, 8, size=num_symbols)
    symbols = levels[idx_i] + 1j * levels[idx_q]
    return symbols / np.sqrt(42)


def qam256_modulate(num_symbols: int, rng: np.random.Generator = None) -> np.ndarray:
    """256QAM 调制，归一化平均功率 = 1。"""
    if rng is None:
        rng = np.random.default_rng()
    levels = np.arange(-15, 16, 2)   # -15, -13, ..., 13, 15
    idx_i = rng.integers(0, 16, size=num_symbols)
    idx_q = rng.integers(0, 16, size=num_symbols)
    symbols = levels[idx_i] + 1j * levels[idx_q]
    return symbols / np.sqrt(170)


MODULATION_MAP = {
    "qpsk":   qpsk_modulate,
    "16qam":  qam16_modulate,
    "64qam":  qam64_modulate,
    "256qam": qam256_modulate,
}


def modulate(num_symbols: int, scheme: str = "qpsk",
             rng: np.random.Generator = None) -> np.ndarray:
    """统一入口。

    Args:
        num_symbols: 符号数
        scheme: 'qpsk' / '16qam' / '64qam' / '256qam'
        rng: 随机数生成器（可选，用于复现）

    Returns:
        归一化复数符号数组，平均功率 = 1
    """
    scheme = scheme.lower()
    if scheme not in MODULATION_MAP:
        raise ValueError(f"未知调制方案: {scheme}. 可选: {list(MODULATION_MAP.keys())}")
    return MODULATION_MAP[scheme](num_symbols, rng)