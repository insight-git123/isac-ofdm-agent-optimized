"""Task4 (P4.2 版): ISAC 感知性能扫描 - Bootstrap/Wilson 置信区间。

相比 P2.3:
- 用 Bootstrap 重采样替代正态近似的 95% CI
- 用 Wilson 区间报告检测率（比例型统计量）
- MC 次数可配置 (默认 100, 支持 --mc-trials)
"""
import sys
import json
import yaml
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.simulation.radar_processing import RadarSimulator
from src.simulation.cfar import ca_cfar_2d
from src.simulation.statistics import bootstrap_ci, wilson_ci_no_scipy


def cluster_detections(rdm_mag, detected, range_axis, velocity_axis,
                       r_gate=3, v_gate=3):
    rows, cols = np.where(detected)
    if len(rows) == 0:
        return [], [], []
    points = sorted(zip(rows, cols), key=lambda p: -rdm_mag[p[0], p[1]])
    clustered = []
    for (r, c) in points:
        if not any(abs(r - cr) <= r_gate and abs(c - cc) <= v_gate
                   for cr, cc in clustered):
            clustered.append((r, c))
    det_ranges = [range_axis[r] for r, _ in clustered]
    det_velocities = [velocity_axis[c] for _, c in clustered]
    det_mags = [rdm_mag[r, c] for r, c in clustered]
    return det_ranges, det_velocities, det_mags


def run_single_mu(mu, scs_khz, cp_duration_us, targets,
                  snr_db=20, use_swerling=True, use_multipath=True,
                  tdl_model="TDL-A", rms_delay_ns=5.0,
                  n_trials=100, base_seed=42):
    """对单个 mu 运行 N 次蒙特卡洛，返回统计量 (含 Bootstrap/Wilson CI)。"""
    if scs_khz <= 60:
        num_sym = 64
    elif scs_khz <= 240:
        num_sym = 256
    elif scs_khz <= 480:
        num_sym = 512
    else:
        num_sym = 2048

    hits_list, ghosts_list = [], []
    range_res_final = None
    bandwidth_final = None
    velocity_res_final = None

    for trial in range(n_trials):
        np.random.seed(base_seed + trial)

        sim = RadarSimulator(scs_khz=scs_khz, fft_size=1024,
                             num_symbols=num_sym, fc_ghz=3.5)
        bandwidth_final = sim.bandwidth
        range_res_final = sim.c / (2 * sim.bandwidth)
        cpi = num_sym * sim.symbol_duration
        velocity_res_final = sim.c / (2 * sim.fc_ghz * 1e9 * cpi)

        X, Y = sim.generate_echo(
            targets=targets, snr_db=snr_db,
            use_swerling=use_swerling,
            use_multipath=use_multipath,
            tdl_model=tdl_model,
            rms_delay_ns=rms_delay_ns,
        )
        rdm_mag, range_axis, velocity_axis = sim.compute_rdm(X, Y)

        rdm_power = np.abs(rdm_mag) ** 2
        detected_cfar = ca_cfar_2d(rdm_power, guard_cells=2,
                                   train_cells=4, pfa=1e-3)
        max_power = np.max(rdm_power)
        detected = detected_cfar & (rdm_power > max_power * 0.1)

        r_gate_phys = 20.0
        r_gate = max(3, int(round(r_gate_phys / range_res_final)))
        v_gate = 3
        det_ranges, det_velocities, _ = cluster_detections(
            rdm_mag, detected, range_axis, velocity_axis,
            r_gate=r_gate, v_gate=v_gate
        )

        R_TOL = max(5.0, 2 * range_res_final)
        V_TOL = max(20.0, 2 * velocity_res_final)
        true_hits = 0
        for tgt in targets:
            for (r, v) in zip(det_ranges, det_velocities):
                if abs(r - tgt["range"]) <= R_TOL and \
                   abs(v - tgt["velocity"]) <= V_TOL:
                    true_hits += 1
                    break

        hits_list.append(true_hits)
        ghosts_list.append(len(det_ranges) - true_hits)

    # ★ Bootstrap CI (连续统计量)
    hits_stat = bootstrap_ci(hits_list, n_bootstrap=2000, confidence=0.95)
    ghosts_stat = bootstrap_ci(ghosts_list, n_bootstrap=2000, confidence=0.95)

    # ★ Wilson CI (比例型统计量: 检测率)
    total_targets = n_trials * len(targets)
    total_hits = int(np.sum(hits_list))
    detection_stat = wilson_ci_no_scipy(total_hits, total_targets,
                                         confidence=0.95)

    return {
        "mu": mu,
        "scs_khz": scs_khz,
        "cp_duration_us": cp_duration_us,
        "num_symbols": num_sym,
        "bandwidth_mhz": round(bandwidth_final / 1e6, 2),
        "range_res_m": round(range_res_final, 2),
        "velocity_res_ms": round(velocity_res_final, 2),
        "n_trials": n_trials,
        "avg_hits": round(hits_stat["mean"], 3),
        "ci95_hits_lo": round(hits_stat["ci_lo"], 3),
        "ci95_hits_hi": round(hits_stat["ci_hi"], 3),
        "ci95_hits_half": round(hits_stat["ci_half_width"], 3),
        "avg_ghosts": round(ghosts_stat["mean"], 3),
        "ci95_ghosts_lo": round(ghosts_stat["ci_lo"], 3),
        "ci95_ghosts_hi": round(ghosts_stat["ci_hi"], 3),
        "ci95_ghosts_half": round(ghosts_stat["ci_half_width"], 3),
        "detection_rate": round(detection_stat["p_hat"], 4),
        "detection_ci_lo": round(detection_stat["ci_lo"], 4),
        "detection_ci_hi": round(detection_stat["ci_hi"], 4),
        "hits_list": hits_list,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mc-trials", type=int, default=100,
                        help="蒙特卡洛次数 (默认 100; 评审建议 >= 500 但耗时长)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    params_path = ROOT / "outputs" / "task1" / "params.yaml"
    with open(params_path, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    targets = [
        {"range": 150.0, "velocity": 30.0, "rcs": 1.0},
        {"range": 300.0, "velocity": -20.0, "rcs": 0.5},
        {"range": 450.0, "velocity": 0.0, "rcs": 0.8},
    ]

    N_TRIALS = args.mc_trials
    RMS_DELAY_NS = 5.0

    print(f"[Task4] ISAC 感知性能扫描 (P4.2 Bootstrap/Wilson CI)")
    print(f"  蒙特卡洛: {N_TRIALS} 次/μ")
    print(f"  CFAR: Pfa=1e-3 (功率域 + 峰值过滤)")
    print(f"  统计: Bootstrap (n=2000) + Wilson 区间")
    print(f"  随机种子: {args.seed}")
    print(f"  目标: {[(t['range'], t['velocity']) for t in targets]}\n")

    results = []
    for num in params["numerology"]:
        mu = num["mu"]
        scs_khz = num["scs_khz"]
        cp_dur = num["cp_duration_us"]
        print(f"  [mu={mu}] SCS={scs_khz}kHz, {N_TRIALS} 次 MC...")
        res = run_single_mu(mu, scs_khz, cp_dur, targets,
                            n_trials=N_TRIALS, rms_delay_ns=RMS_DELAY_N_NS
                            if False else RMS_DELAY_NS,
                            base_seed=args.seed)
        results.append(res)
        print(f"    命中: {res['avg_hits']:.2f} [{res['ci95_hits_lo']:.2f}, "
              f"{res['ci95_hits_hi']:.2f}], 检测率: {res['detection_rate']*100:.1f}% "
              f"[{res['detection_ci_lo']*100:.1f}, {res['detection_ci_hi']*100:.1f}]")

    out_dir = ROOT / "outputs" / "task4"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "isac_sweep_report.json"
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False),
                          encoding="utf-8")

    # 可视化
    mus = [r["mu"] for r in results]
    det_pct = [r["detection_rate"] * 100 for r in results]
    det_lo = [r["detection_ci_lo"] * 100 for r in results]
    det_hi = [r["detection_ci_hi"] * 100 for r in results]
    det_err_lo = [d - l for d, l in zip(det_pct, det_lo)]
    det_err_hi = [h - d for d, h in zip(det_pct, det_hi)]

    ghosts = [r["avg_ghosts"] for r in results]
    ghost_lo = [r["ci95_ghosts_lo"] for r in results]
    ghost_hi = [r["ci95_ghosts_hi"] for r in results]
    ghost_err_lo = [g - l for g, l in zip(ghosts, ghost_lo)]
    ghost_err_hi = [h - g for g, h in zip(ghosts, ghost_hi)]

    range_res = [r["range_res_m"] for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 图 1: 检测率 + Wilson CI
    axes[0].bar(mus, det_pct,
                yerr=[det_err_lo, det_err_hi],
                color='steelblue', capsize=5,
                error_kw={'elinewidth': 2, 'ecolor': 'darkred'})
    axes[0].set_xlabel("mu")
    axes[0].set_ylabel("Detection Rate (%)")
    axes[0].set_title(f"Detection Rate ({N_TRIALS} MC, Wilson 95% CI)")
    axes[0].set_xticks(mus)
    axes[0].set_ylim(0, 110)
    for i, v in enumerate(det_pct):
        axes[0].text(mus[i], v + 5, f"{v:.0f}%", ha='center', fontsize=9)

    # 图 2: 鬼影数 + Bootstrap CI
    axes[1].bar(mus, ghosts,
                yerr=[ghost_err_lo, ghost_err_hi],
                color='coral', capsize=5,
                error_kw={'elinewidth': 2, 'ecolor': 'darkred'})
    axes[1].set_xlabel("mu")
    axes[1].set_ylabel("Avg Ghost Count")
    axes[1].set_title(f"Multipath Ghosts ({N_TRIALS} MC, Bootstrap 95% CI)")
    axes[1].set_xticks(mus)
    for i, v in enumerate(ghosts):
        axes[1].text(mus[i], v + 0.3, f"{v:.1f}", ha='center', fontsize=9)

    # 图 3: 距离分辨率
    axes[2].bar(mus, range_res, color='seagreen')
    axes[2].set_xlabel("mu")
    axes[2].set_ylabel("Range Resolution (m)")
    axes[2].set_title("Range Resolution")
    axes[2].set_xticks(mus)
    axes[2].set_yscale('log')

    plt.tight_layout()
    plot_path = out_dir / "isac_sweep_summary.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()

    # 终端表格
    print("\n" + "=" * 100)
    print(f"{'mu':<4}{'SCS':<8}{'BW(MHz)':<10}{'Res(m)':<10}"
          f"{'命中 [95% CI]':<24}{'检测率 [Wilson CI]':<28}")
    print("-" * 100)
    for r in results:
        hit_str = f"{r['avg_hits']:.2f} [{r['ci95_hits_lo']:.2f},{r['ci95_hits_hi']:.2f}]"
        det_str = (f"{r['detection_rate']*100:.1f}% "
                   f"[{r['detection_ci_lo']*100:.1f},{r['detection_ci_hi']*100:.1f}]")
        print(f"{r['mu']:<4}{r['scs_khz']:<8}{r['bandwidth_mhz']:<10}"
              f"{r['range_res_m']:<10}{hit_str:<24}{det_str:<28}")
    print("=" * 100)

    print(f"\n[output] JSON: {json_path}")
    print(f"[output] 汇总图: {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())