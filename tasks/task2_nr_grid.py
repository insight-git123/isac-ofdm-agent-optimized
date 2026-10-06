"""Task2 补充: NR 资源网格可视化 (DC + guard + DMRS + PRS)。"""
import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from src.simulation.nr_grid import NRResourceGrid, visualize_grid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mu", type=int, default=3)
    parser.add_argument("--n-rb", type=int, default=52)
    parser.add_argument("--guard-rb", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    print(f"[Task2-NRGrid] NR 资源网格生成 (n_rb={args.n_rb}, guard={args.guard_rb} RB)")

    out_dir = ROOT / "outputs" / "task2"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 场景 1: 纯数据网格
    grid1 = NRResourceGrid(n_rb=args.n_rb, n_symbols=14,
                           dc_null=True, guard_rb=args.guard_rb)
    grid1.fill_data()
    path1 = out_dir / f"nr_grid_mu{args.mu}_data.png"
    visualize_grid(grid1, save_path=path1,
                   title=f"NR Grid (Data only, {args.n_rb} RB)")
    print(f"  [1/3] 纯数据网格 → {path1.name}")

    # 场景 2: 数据 + DMRS
    grid2 = NRResourceGrid(n_rb=args.n_rb, n_symbols=14,
                           dc_null=True, guard_rb=args.guard_rb)
    grid2.map_dmrs(symbol_positions=[2, 11], dmrs_type=1)
    grid2.fill_data()
    path2 = out_dir / f"nr_grid_mu{args.mu}_dmrs.png"
    visualize_grid(grid2, save_path=path2,
                   title=f"NR Grid (Data + DMRS, {args.n_rb} RB)")
    print(f"  [2/3] 数据 + DMRS → {path2.name}")

    # 场景 3: 数据 + DMRS + PRS
    grid3 = NRResourceGrid(n_rb=args.n_rb, n_symbols=14,
                           dc_null=True, guard_rb=args.guard_rb)
    grid3.map_dmrs(symbol_positions=[2], dmrs_type=1)
    grid3.map_prs(symbol_positions=[5, 6, 7, 8], comb_size=4, re_offset=0)
    grid3.fill_data()
    path3 = out_dir / f"nr_grid_mu{args.mu}_dmrs_prs.png"
    visualize_grid(grid3, save_path=path3,
                   title=f"NR Grid (Data + DMRS + PRS, {args.n_rb} RB)")
    print(f"  [3/3] 数据 + DMRS + PRS → {path3.name}")

    print("\n  资源占用统计:")
    for name, g in [("Data only", grid1),
                    ("Data+DMRS", grid2),
                    ("Data+DMRS+PRS", grid3)]:
        s = g.occupancy_stats()
        print(f"    {name:16s}: data={s['data']:5d}, dmrs={s['dmrs']:4d}, "
              f"prs={s['prs']:4d}, guard={s['guard']:4d}, dc={s['dc']:2d}, "
              f"data_ratio={s['data_ratio']*100:.1f}%")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())