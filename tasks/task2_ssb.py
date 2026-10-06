"""Task2 补充: SS/PBCH block 生成与可视化。"""
import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from src.simulation.ssb import SSBGrid, visualize_ssb, occupancy_stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-id", type=int, default=42,
                        help="物理层小区 ID (0-1007)")
    parser.add_argument("--i-ssb", type=int, default=0,
                        help="SSB 索引 (0-7)")
    args = parser.parse_args()

    print(f"[Task2-SSB] SS/PBCH Block (N_ID={args.n_id}, i_SSB={args.i_ssb})")

    # 构建 SSB
    ssb = SSBGrid(n_id=args.n_id, n_hf=0, i_ssb=args.i_ssb).build()

    # 输出统计
    stats = occupancy_stats(ssb)
    print(f"\n  SSB 尺寸: {SSBGrid.N_SC} 子载波 × {SSBGrid.N_SYMBOLS} 符号")
    print(f"  资源占用:")
    print(f"    PSS:  {stats['pss']:4d} RE ({stats['pss']/stats['total']*100:.1f}%)")
    print(f"    SSS:  {stats['sss']:4d} RE ({stats['sss']/stats['total']*100:.1f}%)")
    print(f"    PBCH: {stats['pbch']:4d} RE ({stats['pbch']/stats['total']*100:.1f}%)")
    print(f"    DMRS: {stats['dmrs']:4d} RE ({stats['dmrs']/stats['total']*100:.1f}%)")
    print(f"    Zero: {stats['zero']:4d} RE ({stats['zero']/stats['total']*100:.1f}%)")

    # 验证 PSS/SSS 位置
    pss_sc = np.where(ssb.mask[:, 0] == "pss")[0]
    sss_sc = np.where(ssb.mask[:, 2] == "sss")[0]
    print(f"\n  验证:")
    print(f"    PSS 子载波范围: {pss_sc.min()} ~ {pss_sc.max()} (期望 56~182)")
    print(f"    SSS 子载波范围: {sss_sc.min()} ~ {sss_sc.max()} (期望 56~182)")
    print(f"    PSS/SSS 符号数: {len(pss_sc)}/{len(sss_sc)} (期望 127)")

    # 绘图
    out_dir = ROOT / "outputs" / "task2"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_path = out_dir / f"ssb_grid_nid{args.n_id}.png"
    visualize_ssb(ssb, save_path=plot_path)
    print(f"\n[output] {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())