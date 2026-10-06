"""信道编码单元测试。"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.simulation.channel_coding import (
    ConvolutionalEncoder, ViterbiDecoder, awgn_channel, measure_ber
)


def test_encoder_output_length():
    """(2,1,3) 编码器: 输入 n 比特 → 输出 2*(n+2) 比特。"""
    enc = ConvolutionalEncoder()
    bits = np.array([1, 0, 1, 1, 0])
    encoded = enc.encode(bits)
    assert len(encoded) == 2 * (len(bits) + 2)


def test_encoder_deterministic():
    """相同输入 → 相同输出。"""
    enc = ConvolutionalEncoder()
    bits = np.array([1, 0, 1, 1, 0])
    assert np.array_equal(enc.encode(bits), enc.encode(bits))


def test_encoder_single_bit():
    """单比特 1: 输出应可预测。"""
    enc = ConvolutionalEncoder()
    out = enc.encode(np.array([1]))
    # 输入 [1,0,0] → 寄存器 [1,0,0] → G1:1, G2:1
    # 输入 [0,0,0] → 寄存器 [0,1,0] → G1:1, G2:0
    # 输入 [0,0,0] → 寄存器 [0,0,1] → G1:1, G2:1
    assert list(out) == [1, 1, 1, 0, 1, 1]


def test_decoder_noiseless():
    """无噪声时应 100% 恢复。"""
    enc = ConvolutionalEncoder()
    dec = ViterbiDecoder()
    bits = np.random.default_rng(42).integers(0, 2, size=100)
    encoded = enc.encode(bits)
    decoded = dec.decode(encoded)
    assert np.array_equal(bits, decoded[:len(bits)])


def test_decoder_corrects_single_error():
    """单比特翻转应被 Viterbi 纠正。"""
    enc = ConvolutionalEncoder()
    dec = ViterbiDecoder()
    bits = np.zeros(100, dtype=int)
    encoded = enc.encode(bits)
    # 翻转一个比特
    encoded[10] ^= 1
    decoded = dec.decode(encoded)
    # 大部分应被纠正
    errors = np.sum(bits != decoded[:len(bits)])
    assert errors < 10, f"太多错误: {errors}"


def test_awgn_channel_high_snr():
    """高 SNR 下误码率应很低。"""
    rng = np.random.default_rng(42)
    bits = rng.integers(0, 2, size=10000)
    received = awgn_channel(bits, snr_db=10, rng=rng)
    ber = np.mean(bits != received)
    assert ber < 0.01


def test_awgn_channel_low_snr():
    """低 SNR 下误码率应接近 0.5。"""
    rng = np.random.default_rng(42)
    bits = rng.integers(0, 2, size=10000)
    received = awgn_channel(bits, snr_db=-5, rng=rng)
    ber = np.mean(bits != received)
    assert ber > 0.2


def test_measure_ber_coding_gain():
    """高 SNR 下编码应带来 BER 改善。"""
    res_uncoded = measure_ber(n_bits=5000, snr_db_range=[5], use_coding=False)
    res_coded = measure_ber(n_bits=5000, snr_db_range=[5], use_coding=True)
    # 编码后 BER 应更低
    assert res_coded["ber"][0] <= res_uncoded["ber"][0]