"""SCL 译码单元测试 (Phase 2, 标准实现)。"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.simulation.polar import polar_encode
from src.simulation.scl_decoder import (
    SCLDecoder, scl_decode, scl_decode_with_crc_selection
)


def test_scl_noiseless():
    """无噪声: SCL 应 100% 恢复。"""
    rng = np.random.default_rng(42)
    info = rng.integers(0, 2, size=20)
    codeword, frozen, ct = polar_encode(info, N=64, crc_type="CRC11")
    llr = np.where(codeword == 0, 10.0, -10.0)
    decoded = scl_decode(llr, frozen, 20, L=8, crc_type=ct)
    assert np.array_equal(info, decoded)


def test_scl_reproducible():
    """相同输入 → 相同输出。"""
    rng = np.random.default_rng(42)
    info = rng.integers(0, 2, size=20)
    codeword, frozen, ct = polar_encode(info, N=64, crc_type="CRC11")
    llr = np.where(codeword == 0, 5.0, -5.0)
    d1 = scl_decode(llr, frozen, 20, L=8, crc_type=ct)
    d2 = scl_decode(llr, frozen, 20, L=8, crc_type=ct)
    assert np.array_equal(d1, d2)


def test_scl_list_returns_l_paths():
    """decode_with_list 应返回 L 条路径。"""
    rng = np.random.default_rng(42)
    info = rng.integers(0, 2, size=20)
    codeword, frozen, ct = polar_encode(info, N=64, crc_type="CRC11")
    llr = np.where(codeword == 0, 5.0, -5.0)
    decoder = SCLDecoder(N=64, L=8)
    paths = decoder.decode_with_list(llr, frozen)
    assert len(paths) == 8
    assert all(len(p) == 64 for p in paths)


def test_scl_handles_noise():
    """有噪声: SCL 应返回正确长度和值域。"""
    rng = np.random.default_rng(42)
    info = rng.integers(0, 2, size=20)
    codeword, frozen, ct = polar_encode(info, N=64, crc_type="CRC11")
    sigma = 0.8
    rx = (1 - 2 * codeword.astype(float)) + sigma * rng.normal(size=64)
    llr = 2 * rx / sigma**2
    decoded = scl_decode(llr, frozen, 20, L=8, crc_type=ct)
    assert len(decoded) == 20
    assert np.all(np.isin(decoded, [0, 1]))


def test_scl_crc_selection():
    """CRC 辅助路径选择应正常工作。"""
    rng = np.random.default_rng(42)
    info = rng.integers(0, 2, size=20)
    codeword, frozen, ct = polar_encode(info, N=64, crc_type="CRC11")
    llr = np.where(codeword == 0, 5.0, -5.0)
    decoded = scl_decode_with_crc_selection(llr, frozen, 20, L=8, crc_type=ct)
    assert len(decoded) == 20
    # 无噪声下应正确
    assert np.array_equal(info, decoded)


def test_scl_l1_equals_sc():
    """L=1 的 SCL 应等价于 SC 译码。"""
    from src.simulation.polar import polar_decode
    rng = np.random.default_rng(42)
    info = rng.integers(0, 2, size=20)
    codeword, frozen, _ = polar_encode(info, N=64, crc_type=None)
    llr = np.where(codeword == 0, 5.0, -5.0)
    # SC
    sc_result = polar_decode(llr, frozen, 20, crc_type=None)
    # SCL L=1
    decoder = SCLDecoder(N=64, L=1)
    u_hat = decoder.decode(llr, frozen)
    info_positions = np.where(~frozen)[0]
    scl_l1 = u_hat[info_positions[:20]]
    assert np.array_equal(sc_result, scl_l1)


def test_scl_performance():
    """性能: N=64 应在 5 秒内完成 (递归版较慢)。"""
    import time
    rng = np.random.default_rng(42)
    info = rng.integers(0, 2, size=20)
    codeword, frozen, ct = polar_encode(info, N=64, crc_type="CRC11")
    llr = np.where(codeword == 0, 5.0, -5.0)
    t0 = time.time()
    decoded = scl_decode(llr, frozen, 20, L=8, crc_type=ct)
    elapsed = time.time() - t0
    assert elapsed < 5.0, f"SCL 耗时 {elapsed:.3f}s, 期望 < 5s"