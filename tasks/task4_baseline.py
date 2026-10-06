"""Task4-Baseline: 基准场景对比 (AWGN vs 单径 vs TDL-A)。

输出:
- 检测率 vs SNR 曲线 (3 个场景对比)
- 距离 RMSE vs SNR
- 速度 RMSE vs SNR
"""
import sys
import json
import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.simulation.scenarios import run_scenario


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mu", type=int, default=3)
    parser.add_argument("--n-trials", type=int, default=20,
                        help="每个 (场景, SNR) 的 MC 次数")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # 读 numerology
    with open(ROOT / "outputs" / "task1" / "params.yaml", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    num = next((n for n in params["numerology"] if n["mu"] == args.mu), None)
    scs_khz = num["scs_khz"]

    targets = [
        {"range": 150.0, "velocity": 30.0, "rcs": 1.0},
        {"range": 300.0, "velocity": -20.0, "rcs": 0.5},
        {"range": 450.0, "velocity": 0.0, "rcs": 0.8},
    ]

    scenarios = ["AWGN", "SinglePath", "TDL-A"]
    # 相干增益 = 1024 (距离) × 512 (多普勒) ≈ 57 dB
    # 要看到 S 型检测曲线, 输入 SNR 需在 -70 ~ -40 dB
    snr_range = [-70, -65, -60, -55, -50, -45, -40]

    print(f"[Task4-Baseline] 基准场景对比 (mu={args.mu}, SCS={scs_khz}kHz)")
    print(f"  场景: {scenarios}")
    print(f"  SNR: {snr_range} dB")
    print(f"  MC: {args.n_trials} 次/(场景, SNR)")
    print(f"  种子: {args.seed}\n")

    all_results = {}
    for scenario in scenarios:
        results = []
        print(f"  --- 场景: {scenario} ---")
        for snr in snr_range:
            res = run_scenario(scenario, scs_khz, targets, snr,
                               n_trials=args.n_trials, seed=args.seed)
            results.append(res)
            print(f"    SNR={snr:2d}dB: "
                  f"检测率={res['detection_rate']*100:5.1f}%, "
                  f"距离RMSE={res['range_rmse_m']:5.2f}m, "
                  f"速度RMSE={res['velocity_rmse_ms']:5.2f}m/s")
        all_results[scenario] = results

    # 保存 JSON
    out_dir = ROOT / "outputs" / "task4"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "baseline_comparison.json"
    json_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False),
                          encoding="utf-8")

    # 绘图
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    colors = {"AWGN": "steelblue", "SinglePath": "seagreen", "TDL-A": "coral"}
    markers = {"AWGN": "o", "SinglePath": "s", "TDL-A": "^"}

    # 图 1: 检测率 vs SNR
    ax = axes[0]
    for scenario, results in all_results.items():
        snrs = [r["snr_db"] for r in results]
        dets = [r["detection_rate"] * 100 for r in results]
        ax.plot(snrs, dets, marker=markers[scenario], color=colors[scenario],
                label=scenario, linewidth=2, markersize=8)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Detection Rate (%)")
    ax.set_title("Detection Rate vs SNR")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim(-5, 110)

    # 图 2: 距离 RMSE vs SNR
    ax = axes[1]
    for scenario, results in all_results.items():
        snrs = [r["snr_db"] for r in results]
        rmse = [r["range_rmse_m"] for r in results]
        ax.plot(snrs, rmse, marker=markers[scenario], color=colors[scenario],
                label=scenario, linewidth=2, markersize=8)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Range RMSE (m)")
    ax.set_title("Range Estimation Error")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 图 3: 速度 RMSE vs SNR
    ax = axes[2]
    for scenario, results in all_results.items():
        snrs = [r["snr_db"] for r in results]
        rmse = [r["velocity_rmse_ms"] for r in results]
        ax.plot(snrs, rmse, marker=markers[scenario], color=colors[scenario],
                label=scenario, linewidth=2, markersize=8)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Velocity RMSE (m/s)")
    ax.set_title("Velocity Estimation Error")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plot_path = out_dir / "baseline_comparison.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print(f"\n[output] JSON: {json_path}")
    print(f"[output] 汇总图: {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())