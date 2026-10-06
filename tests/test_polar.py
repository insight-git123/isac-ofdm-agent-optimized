"""Polar 码单元测试 (Phase 1: PW 序列 + CRC)。"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.simulation.polar import (
    polar_encode, polar_decode, polar_transform, bhattacharyya_params
)


def test_polar_transform_shape():
    u = np.array([1, 0, 1, 0, 1, 1, 0, 0])
    x = polar_transform(u)
    assert len(x) == 8


def test_polar_transform_binary():
    u = np.random.default_rng(42).integers(0, 2, size=16)
    x = polar_transform(u)
    assert np.all(np.isin(x, [0, 1]))


def test_bhattacharyya_shape():
    z = bhattacharyya_params(4)
    assert len(z) == 16
    assert np.all(z >= 0) and np.all(z <= 1)


def test_polar_encode_shape():
    info = np.array([1, 0, 1, 1])
    codeword, frozen, crc_type = polar_encode(info, N=32, crc_type="CRC6")
    assert len(codeword) == 32
    assert crc_type == "CRC6"


def test_polar_encode_with_crc():
    """N=64, K=11, CRC-6 → 17 位信息 + CRC。"""
    info = np.array([1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1])
    codeword, frozen, crc_type = polar_encode(info, N=64, crc_type="CRC6")
    assert len(codeword) == 64
    assert (~frozen).sum() == 11 + 6   # K + CRC


def test_polar_noiseless_decoding():
    """无噪声: 应 100% 恢复。"""
    rng = np.random.default_rng(42)
    info = rng.integers(0, 2, size=20)
    codeword, frozen, crc_type = polar_encode(info, N=64, crc_type="CRC11")
    llr = np.where(codeword == 0, 10.0, -10.0)
    decoded = polar_decode(llr, frozen, len(info), crc_type=crc_type)
    assert len(decoded) == len(info)
    assert np.array_equal(info, decoded)