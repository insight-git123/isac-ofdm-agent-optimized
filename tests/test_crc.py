"""CRC 单元测试。"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.simulation.crc import crc_encode, crc_check, remove_crc


def test_crc11_length():
    bits = np.array([1, 0, 1, 1, 0, 1, 0], dtype=np.int8)
    out = crc_encode(bits, "CRC11")
    assert len(out) == len(bits) + 11


def test_crc6_length():
    bits = np.array([1, 0, 1, 1, 0], dtype=np.int8)
    out = crc_encode(bits, "CRC6")
    assert len(out) == len(bits) + 6


def test_crc24a_length():
    bits = np.array([1] * 100, dtype=np.int8)
    out = crc_encode(bits, "CRC24A")
    assert len(out) == 124


def test_crc_encode_decode():
    """编码后, CRC 校验应通过。"""
    rng = np.random.default_rng(42)
    for crc_type in ["CRC6", "CRC11", "CRC24A", "CRC24B"]:
        bits = rng.integers(0, 2, size=50).astype(np.int8)
        encoded = crc_encode(bits, crc_type)
        assert crc_check(encoded, crc_type), f"{crc_type} 校验失败"


def test_crc_detects_error():
    """翻转一位应被检测出来。"""
    rng = np.random.default_rng(42)
    bits = rng.integers(0, 2, size=50).astype(np.int8)
    encoded = crc_encode(bits, "CRC11")
    encoded[20] ^= 1   # 翻转一位
    assert not crc_check(encoded, "CRC11")


def test_crc_remove():
    bits = np.array([1, 0, 1, 1, 0], dtype=np.int8)
    encoded = crc_encode(bits, "CRC11")
    recovered = remove_crc(encoded, "CRC11")
    assert np.array_equal(recovered, bits)


def test_crc_reproducible():
    bits = np.array([1, 0, 1, 1, 0], dtype=np.int8)
    c1 = crc_encode(bits, "CRC11")
    c2 = crc_encode(bits, "CRC11")
    assert np.array_equal(c1, c2)