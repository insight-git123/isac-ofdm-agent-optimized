"""Task3-MIMO-Metrics: 定量评估 MIMO 角度/距离/速度 RMSE。"""
import sys
import json
import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.simulation.mimo_metrics import run_mimo_mc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mu", type=int, default=3)
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(ROOT / "outputs" / "task1" / "params.yaml", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    num = next((n for n in params["numerology"] if n["mu"] == args.mu), None)
    scs_khz = num["scs_khz"]

    targets = [
        {"range": 150.0, "velocity": 30.0, "rcs": 1.0, "az_deg": -30.0},
        {"range": 300.0, "velocity": -20.0, "rcs": 0.5, "az_deg": 0.0},
        {"range": 450.0, "velocity": 0.0, "rcs": 0.8, "az_deg": 40.0},
    ]

    antenna_configs = [4, 8, 16]
    results = []

    print(f"[Task3-MIMO-Metrics] MIMO 误差指标 (mu={args.mu}, SCS={scs_khz}kHz)")
    print(f"  目标: {[(t['range'], t['velocity'], t['az_deg']) for t in targets]}")
    print(f"  MC: {args.n_trials} 次/配置\n")

    for n_ant in antenna_configs:
        print(f"  --- ULA N={n_ant} ---")
        res = run_mimo_mc(scs_khz, targets, n_antennas=n_ant,
                          n_trials=args.n_trials, seed=args.seed)
        results.append(res)
        print(f"    距离 RMSE = {res['range_rmse_m']:.2f} m")
        print(f"    速度 RMSE = {res['velocity_rmse_ms']:.2f} m/s")
        print(f"    角度 RMSE = {res['angle_rmse_deg']:.2f} deg")

    out_dir = ROOT / "outputs" / "task3"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "mimo_metrics.json"
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False),
                         encoding="utf-8")

    # 绘图
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    n_ants = [r["n_antennas"] for r in results]

    axes[0].bar(n_ants, [r["range_rmse_m"] for r in results], color='steelblue')
    axes[0].set_xlabel("ULA elements (N)")
    axes[0].set_ylabel("Range RMSE (m)")
    axes[0].set_title("Range Estimation Error")
    axes[0].set_xticks(n_ants)

    axes[1].bar(n_ants, [r["velocity_rmse_ms"] for r in results], color='coral')
    axes[1].set_xlabel("ULA elements (N)")
    axes[1].set_ylabel("Velocity RMSE (m/s)")
    axes[1].set_title("Velocity Estimation Error")
    axes[1].set_xticks(n_ants)

    axes[2].bar(n_ants, [r["angle_rmse_deg"] for r in results], color='seagreen')
    axes[2].set_xlabel("ULA elements (N)")
    axes[2].set_ylabel("Angle RMSE (deg)")
    axes[2].set_title("Angle Estimation Error")
    axes[2].set_xticks(n_ants)

    plt.tight_layout()
    plot_path = out_dir / "mimo_metrics.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print(f"\n[output] JSON: {json_path}")
    print(f"[output] 汇总图: {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())