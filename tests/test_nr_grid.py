"""NR 资源网格单元测试。"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.simulation.nr_grid import NRResourceGrid


def test_grid_shape():
    g = NRResourceGrid(n_rb=52, n_symbols=14)
    assert g.grid.shape == (52 * 12, 14)
    assert g.mask.shape == (52 * 12, 14)


def test_dc_null():
    g = NRResourceGrid(n_rb=52, n_symbols=14, dc_null=True)
    dc_idx = g.n_sc // 2
    assert np.all(g.grid[dc_idx, :] == 0)
    assert np.all(g.mask[dc_idx, :] == "dc")


def test_guard_band():
    g = NRResourceGrid(n_rb=52, n_symbols=14, guard_rb=2)
    guard_sc = 2 * 12
    assert np.all(g.grid[:guard_sc, :] == 0)
    assert np.all(g.grid[-guard_sc:, :] == 0)
    assert np.all(g.mask[:guard_sc, :] == "guard")


def test_dmrs_mapping():
    g = NRResourceGrid(n_rb=52, n_symbols=14, dc_null=True, guard_rb=2)
    g.map_dmrs(symbol_positions=[2], dmrs_type=1)
    dmrs_count = int((g.mask[:, 2] == "dmrs").sum())
    assert dmrs_count > 0


def test_dmrs_skips_guard_and_dc():
    g = NRResourceGrid(n_rb=52, n_symbols=14, dc_null=True, guard_rb=2)
    g.map_dmrs(symbol_positions=[2], dmrs_type=1)
    # 保护带和 DC 上不应有 DMRS
    guard_sc = 2 * 12
    assert not np.any(g.mask[:guard_sc, 2] == "dmrs")
    assert not np.any(g.mask[-guard_sc:, 2] == "dmrs")
    assert g.mask[g.n_sc // 2, 2] != "dmrs"


def test_prs_mapping():
    g = NRResourceGrid(n_rb=52, n_symbols=14, dc_null=True, guard_rb=2)
    g.map_prs(symbol_positions=[5, 6, 7, 8], comb_size=4)
    for sym in [5, 6, 7, 8]:
        assert (g.mask[:, sym] == "prs").sum() > 0


def test_prs_comb_size():
    g = NRResourceGrid(n_rb=52, n_symbols=14, dc_null=True, guard_rb=2)
    g.map_prs(symbol_positions=[5], comb_size=2)
    prs_count = int((g.mask[:, 5] == "prs").sum())
    # comb=2 时，大约一半的可用子载波是 PRS
    assert prs_count > 50


def test_data_fill():
    g = NRResourceGrid(n_rb=52, n_symbols=14)
    g.fill_data()
    data_mask = g.mask == "data"
    assert np.all(np.abs(g.grid[data_mask]) > 0)


def test_data_does_not_overwrite_dmrs():
    g = NRResourceGrid(n_rb=52, n_symbols=14, dc_null=True, guard_rb=2)
    g.map_dmrs(symbol_positions=[2], dmrs_type=1)
    dmrs_vals_before = g.grid[g.mask == "dmrs"].copy()
    g.fill_data()
    dmrs_vals_after = g.grid[g.mask == "dmrs"]
    assert np.allclose(dmrs_vals_before, dmrs_vals_after)


def test_occupancy_stats():
    g = NRResourceGrid(n_rb=52, n_symbols=14, dc_null=True, guard_rb=2)
    g.map_dmrs(symbol_positions=[2], dmrs_type=1)
    g.fill_data()
    s = g.occupancy_stats()
    assert s["data"] > 0
    assert s["dmrs"] > 0
    assert s["guard"] > 0
    assert s["dc"] > 0
    assert 0 < s["data_ratio"] < 1.0


def test_to_ofdm_symbols_shape():
    g = NRResourceGrid(n_rb=52, n_symbols=14)
    g.fill_data()
    time_signal = g.to_ofdm_symbols()
    assert time_signal.shape == (52 * 12, 14)