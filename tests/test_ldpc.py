"""LDPC 码单元测试。"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.simulation.ldpc import RegularLDPC


def test_ldpc_init():
    code = RegularLDPC(n_var=96, dv=3, dc=6)
    assert code.N == 96
    assert code.M == 48
    assert code.K == 48


def test_ldpc_encode_length():
    code = RegularLDPC(n_var=96, dv=3, dc=6)
    info = np.zeros(code.K, dtype=int)
    codeword = code.encode(info)
    assert len(codeword) == code.N


def test_ldpc_encode_binary():
    code = RegularLDPC(n_var=96, dv=3, dc=6)
    rng = np.random.default_rng(42)
    info = rng.integers(0, 2, size=code.K)
    codeword = code.encode(info)
    assert np.all(np.isin(codeword, [0, 1]))


def test_ldpc_decoder_noiseless():
    """无噪声: 应 100% 恢复。"""
    code = RegularLDPC(n_var=96, dv=3, dc=6)
    rng = np.random.default_rng(42)
    info = rng.integers(0, 2, size=code.K)
    codeword = code.encode(info)
    # 无噪声 LLR
    llr = np.where(codeword == 0, 10.0, -10.0)
    decoded = code.decode(llr, n_iter=30)
    assert len(decoded) == code.K
    assert np.array_equal(info, decoded)