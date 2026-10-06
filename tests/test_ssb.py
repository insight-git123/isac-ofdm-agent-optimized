"""SS/PBCH block 单元测试。"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.simulation.ssb import (
    generate_pss_sequence, generate_sss_sequence, generate_pbch_dmrs,
    SSBGrid, occupancy_stats, _generate_gold_sequence
)


def test_pss_length():
    """PSS 应为 127 长。"""
    for n_id_2 in [0, 1, 2]:
        pss = generate_pss_sequence(n_id_2)
        assert len(pss) == 127


def test_pss_bpsk_values():
    """PSS 取值应为 ±1 (BPSK)。"""
    pss = generate_pss_sequence(1)
    assert np.all(np.isin(pss.real, [-1, 1]))
    assert np.all(pss.imag == 0)


def test_pss_different_nid():
    """不同 n_id_2 应产生不同 PSS。"""
    pss0 = generate_pss_sequence(0)
    pss1 = generate_pss_sequence(1)
    assert not np.array_equal(pss0, pss1)


def test_sss_length():
    """SSS 应为 127 长。"""
    sss = generate_sss_sequence(n_id_1=10, n_id_2=1)
    assert len(sss) == 127


def test_sss_bpsk_values():
    """SSS 取值应为 ±1。"""
    sss = generate_sss_sequence(n_id_1=10, n_id_2=1)
    assert np.all(np.isin(sss.real, [-1, 1]))


def test_gold_sequence_binary():
    """Gold 序列应为 0/1。"""
    c = _generate_gold_sequence(c_init=42, length=100)
    assert len(c) == 100
    assert np.all(np.isin(c, [0, 1]))


def test_pbch_dmrs_length():
    """PBCH DMRS 应为 144 长。"""
    r = generate_pbch_dmrs(n_id=42, n_hf=0, i_ssb=0, n_symbol=1)
    assert len(r) == 144


def test_pbch_dmrs_qpsk_constellation():
    """PBCH DMRS 应为 QPSK 星座点。"""
    r = generate_pbch_dmrs(n_id=42, n_hf=0, i_ssb=0, n_symbol=1)
    # QPSK: 幅度 = 1
    assert np.allclose(np.abs(r), 1.0)


def test_ssb_grid_shape():
    """SSB 网格应为 240 × 4。"""
    ssb = SSBGrid(n_id=42).build()
    assert ssb.grid.shape == (240, 4)


def test_ssb_pss_position():
    """PSS 应在 symbol 0 的子载波 56~182。"""
    ssb = SSBGrid(n_id=42).build()
    pss_sc = np.where(ssb.mask[:, 0] == "pss")[0]
    assert len(pss_sc) == 127
    assert pss_sc.min() == 56
    assert pss_sc.max() == 182


def test_ssb_sss_position():
    """SSS 应在 symbol 2 的子载波 56~182。"""
    ssb = SSBGrid(n_id=42).build()
    sss_sc = np.where(ssb.mask[:, 2] == "sss")[0]
    assert len(sss_sc) == 127
    assert sss_sc.min() == 56
    assert sss_sc.max() == 182


def test_ssb_symbol0_only_pss():
    """Symbol 0 只有 PSS 有值，其余为零。"""
    ssb = SSBGrid(n_id=42).build()
    # Symbol 0 除 56~182 外全零
    non_pss = np.ones(240, dtype=bool)
    non_pss[56:183] = False
    assert np.all(ssb.grid[non_pss, 0] == 0)


def test_ssb_occupancy_stats():
    """占用统计应合理。"""
    ssb = SSBGrid(n_id=42).build()
    stats = occupancy_stats(ssb)
    assert stats["pss"] == 127
    assert stats["sss"] == 127
    assert stats["pbch"] > 0
    assert stats["dmrs"] > 0
    assert stats["total"] == 240 * 4


def test_ssb_reproducible():
    """相同输入 → 相同网格。"""
    ssb1 = SSBGrid(n_id=42, i_ssb=0).build()
    ssb2 = SSBGrid(n_id=42, i_ssb=0).build()
    assert np.array_equal(ssb1.mask, ssb2.mask)