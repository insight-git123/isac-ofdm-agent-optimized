"""PW 序列单元测试。"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.simulation.polar_sequence import (
    polar_weight_sequence, get_frozen_set
)


def test_pw_sequence_length():
    for N in [16, 32, 128, 256]:
        seq = polar_weight_sequence(N)
        assert len(seq) == N


def test_pw_sequence_permutation():
    """序列应是 0..N-1 的排列。"""
    N = 128
    seq = polar_weight_sequence(N)
    assert sorted(seq) == list(range(N))


def test_pw_sequence_monotonic_weight():
    """PW 序列末尾应是权重最大的索引。

    物理: PW 公式 W_i = Σ B_j · β^j (β=2^(1/4))。
    i=0 权重最小 (0), i=N-1 权重最大 (Σ β^j)。
    """
    for N in [16, 32, 64, 128]:
        seq = polar_weight_sequence(N)
        # seq[-1] 应是 i = N-1 (二进制全 1, 权重最大)
        assert seq[-1] == N - 1, \
            f"N={N}: seq[-1]={seq[-1]}, 期望 {N-1}"
        # seq[0] 应是 i = 0 (权重最小)
        assert seq[0] == 0, f"N={N}: seq[0]={seq[0]}, 期望 0"


def test_pw_sequence_power_of_two_positions():
    """PW 序列的可扩展性: 高权重索引应在序列末尾。

    物理: i=N-1 (全 1 二进制) 权重最大 → 最可靠。
          i=0 (全 0) 权重最小 → 最不可靠。
    """
    N = 64
    seq = polar_weight_sequence(N)

    # 最可靠: seq[-1] = N-1
    assert seq[-1] == N - 1, f"seq[-1]={seq[-1]}, 期望 {N-1}"

    # 最不可靠: seq[0] = 0
    assert seq[0] == 0, f"seq[0]={seq[0]}, 期望 0"

    # 单调性: 越靠后权重越大
    beta = 2 ** (1/4)
    def weight(i):
        return sum(((i >> j) & 1) * (beta ** j) for j in range(6))

    for k in range(1, N):
        assert weight(seq[k-1]) <= weight(seq[k]), \
            f"位置 {k-1} 权重大于 {k}"


def test_frozen_set_size():
    """冻结位数量应正确。"""
    N, K = 128, 64
    frozen = get_frozen_set(N, K)
    assert frozen.sum() == N - K
    assert (~frozen).sum() == K


def test_frozen_set_includes_crc_positions():
    """强制添加的位置应作为信息位。"""
    N, K = 128, 64
    extra = [10, 20, 30]
    frozen = get_frozen_set(N, K + len(extra), info_positions_extra=extra)
    for pos in extra:
        assert not frozen[pos]