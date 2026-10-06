"""Task3: 频域多目标 ISAC 感知 + CFAR + NMS + Swerling-I + TDL 多径 + 鬼影抑制"""
import sys
import json
import yaml
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.simulation.radar_processing import RadarSimulator
from src.simulation.cfar import ca_cfar_2d
from src.simulation.ghost_suppression import suppress_ghosts


def cluster_detections(rdm_mag, detected, range_axis, velocity_axis,
                       r_gate=3, v_gate=3):
    """NMS 聚类，返回 (ranges, velocities, mags)。"""
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mu", type=int, default=0,
                        help="Numerology mu (0-6)")
    parser.add_argument("--swerling", action="store_true",
                        help="启用 Swerling-I RCS 波动")
    parser.add_argument("--multipath", action="store_true",
                        help="启用 TDL 多径信道")
    parser.add_argument("--suppress", action="store_true",
                        help="启用多径鬼影抑制")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子 (默认 42，保证可复现)")
    args = parser.parse_args()

    np.random.seed(args.seed)

    # 1. 读取 Task1 参数
    params_path = ROOT / "outputs" / "task1" / "params.yaml"
    with open(params_path, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    target_num = next((n for n in params["numerology"] if n["mu"] == args.mu), None)
    if not target_num:
        print(f"错误: 未找到 mu={args.mu} 的参数")
        return 1

    scs_khz = target_num["scs_khz"]

    # 2. 初始化雷达仿真器
    num_sym = 64 if scs_khz <= 60 else 512
    sim = RadarSimulator(scs_khz=scs_khz, fft_size=1024,
                         num_symbols=num_sym, fc_ghz=3.5)

    tag_list = []
    if args.swerling:  tag_list.append("Swerling")
    if args.multipath: tag_list.append("Multipath")
    if args.suppress:  tag_list.append("GhostSuppress")
    tag_str = "+".join(tag_list) if tag_list else "Baseline"

    print(f"[Task3] ISAC 感知仿真 (mu={args.mu}, SCS={scs_khz}kHz, {tag_str})")
    print(f"  符号数: {num_sym}, CPI: {num_sym * sim.symbol_duration * 1e6:.1f} us")
    print(f"  带宽: {sim.bandwidth/1e6:.2f} MHz, 距离分辨率: {sim.c/(2*sim.bandwidth):.2f} m")
    print(f"  随机种子: {args.seed}")

    # 3. 多目标真值
    targets = [
        {"range": 150.0, "velocity": 30.0, "rcs": 1.0},
        {"range": 300.0, "velocity": -20.0, "rcs": 0.5},
        {"range": 450.0, "velocity": 0.0, "rcs": 0.8},
    ]
    print(f"  目标: {[(t['range'], t['velocity']) for t in targets]}")

    # 4. 生成回波 + RDM
    X, Y = sim.generate_echo(
        targets=targets, snr_db=20,
        use_swerling=args.swerling,
        use_multipath=args.multipath,
        tdl_model="TDL-A",
        rms_delay_ns=5.0,
    )
    rdm_mag, range_axis, velocity_axis = sim.compute_rdm(X, Y)
    rdm_db = 20 * np.log10(rdm_mag / np.max(rdm_mag) + 1e-12)

    # 5. CFAR 检测 (功率域 + Pfa=1e-3 + 峰值过滤)
    rdm_power = np.abs(rdm_mag) ** 2
    detected_cfar = ca_cfar_2d(rdm_power, guard_cells=2,
                               train_cells=4, pfa=1e-3)
    max_power = np.max(rdm_power)
    detected = detected_cfar & (rdm_power > max_power * 0.1)
    raw_count = int(np.sum(detected))

    # 6. NMS 聚类
    range_res_m = sim.c / (2 * sim.bandwidth)
    r_gate_phys = 20.0
    r_gate = max(3, int(round(r_gate_phys / range_res_m)))
    v_gate = 3
    det_ranges, det_velocities, det_mags = cluster_detections(
        rdm_mag, detected, range_axis, velocity_axis,
        r_gate=r_gate, v_gate=v_gate
    )
    print(f"  [NMS] r_gate={r_gate} 门 (物理 ~{r_gate * range_res_m:.1f}m), v_gate={v_gate}")
    print(f"\n  CFAR+NMS 检测点: {raw_count} → {len(det_ranges)}")
    for i, (r, v) in enumerate(zip(det_ranges, det_velocities)):
        print(f"    [{i+1}] 距离={r:.1f}m, 速度={v:.1f}m/s")

    # 7. 鬼影抑制 (可选)
    true_idx, ghost_idx = [], []
    if args.suppress and len(det_ranges) > 0:
        true_idx, ghost_idx = suppress_ghosts(
            det_ranges, det_velocities, det_mags,
            r_tol=5.0, v_tol=10.0, mag_ratio=0.7, verbose=True
        )
        print(f"\n  === 鬼影抑制结果 ===")
        print(f"  真实目标: {len(true_idx)} 个")
        for i in true_idx:
            print(f"    OK  {det_ranges[i]:.1f}m, {det_velocities[i]:.1f}m/s")
        print(f"  鬼影: {len(ghost_idx)} 个")
        for i in ghost_idx:
            print(f"    XX  {det_ranges[i]:.1f}m, {det_velocities[i]:.1f}m/s")

    # 8. 绘图
    out_dir = ROOT / "outputs" / "task3"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_" + "_".join(t.lower() for t in tag_list) if tag_list else ""
    plot_path = out_dir / f"rdm_mu{args.mu}{suffix}_cfar.png"

    plt.figure(figsize=(10, 6))
    plt.imshow(rdm_db, aspect='auto', cmap='jet',
               extent=[velocity_axis[0], velocity_axis[-1],
                       range_axis[0], range_axis[-1]],
               origin='lower', vmin=-40, vmax=0)
    plt.colorbar(label='Normalized Magnitude (dB)')

    if args.suppress and (true_idx or ghost_idx):
        if true_idx:
            plt.scatter([det_velocities[i] for i in true_idx],
                        [det_ranges[i] for i in true_idx],
                        facecolors='none', edgecolors='lime', s=140,
                        linewidths=2.5, label='True Target')
        if ghost_idx:
            plt.scatter([det_velocities[i] for i in ghost_idx],
                        [det_ranges[i] for i in ghost_idx],
                        marker='x', color='red', s=100, linewidths=2.5,
                        label='Ghost (Rejected)')
    else:
        plt.scatter(det_velocities, det_ranges,
                    facecolors='none', edgecolors='white', s=80,
                    label='CFAR + NMS')

    for tgt in targets:
        plt.axvline(tgt["velocity"], color='cyan', linestyle='--', alpha=0.4)
        plt.axhline(tgt["range"], color='cyan', linestyle='--', alpha=0.4)

    plt.title(f"ISAC RDM (mu={args.mu}, SCS={scs_khz}kHz, {tag_str})")
    plt.xlabel("Velocity (m/s)")
    plt.ylabel("Range (m)")
    plt.xlim(-150, 150)
    plt.ylim(0, 600)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print(f"\n[output] {plot_path}")

    # ================= 8.5 鬼影抑制评估 (P4.4) =================
    R_TOL = max(5.0, 2 * range_res_m)   # ← 提前到 8.5 节之前
    suppression_metrics = None
    if args.suppress and len(det_ranges) > 0:
        # 真值匹配函数 (检测点是否落在真值附近)
        def is_true_target(r, v):
            return any(
                abs(r - t["range"]) <= R_TOL and abs(v - t["velocity"]) <= 15.0
                for t in targets
            )

        # 混淆矩阵
        TP = 0   # 保留的点中, 确实是真实目标
        FP = 0   # 保留的点中, 实际是鬼影 (误保留)
        FN = 0   # 抑制的点中, 实际是真实目标 (误抑制)
        TN = 0   # 抑制的点中, 确实是鬼影

        for i in true_idx:
            if is_true_target(det_ranges[i], det_velocities[i]):
                TP += 1
            else:
                FP += 1

        for i in ghost_idx:
            if is_true_target(det_ranges[i], det_velocities[i]):
                FN += 1   # ❌ 误抑制了真实目标
            else:
                TN += 1   # ✅ 正确抑制了鬼影

        precision = TP / max(TP + FP, 1)   # 保留点中真实目标的纯度
        recall = TP / max(TP + FN, 1)      # 真实目标的保留率
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        ghost_rejection_rate = TN / max(TN + FP, 1)  # 鬼影剔除率

        # 真值总数 (用于计算误抑制率)
        n_true_targets = len(targets)
        false_suppression_rate = FN / max(n_true_targets, 1)

        suppression_metrics = {
            "TP": TP, "FP": FP, "FN": FN, "TN": TN,
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "ghost_rejection_rate": round(ghost_rejection_rate, 3),
            "false_suppression_rate": round(false_suppression_rate, 3),
        }

        print(f"\n  === 鬼影抑制评估 (P4.4) ===")
        print(f"  TP={TP} (保留真实)   FP={FP} (误保留鬼影)")
        print(f"  FN={FN} (误抑制真实)  TN={TN} (正确剔除鬼影)")
        print(f"  Precision = {precision:.3f}  (保留点中真实目标纯度)")
        print(f"  Recall    = {recall:.3f}  (真实目标保留率)")
        print(f"  F1        = {f1:.3f}")
        print(f"  误抑制率  = {false_suppression_rate:.3f}")

    # ================= 9. 保存 result.json (P2.2) =================
   

    def match_type(det_r, det_v):
        for t in targets:
            if abs(det_r - t["range"]) <= R_TOL and abs(det_v - t["velocity"]) <= 15.0:
                return "true", t
        return "ghost", None

    detections_json = []
    max_mag = max(det_mags) if det_mags else 1.0
    for i, (dr, dv, dm) in enumerate(zip(det_ranges, det_velocities, det_mags)):
        dtype, matched = match_type(dr, dv)
        entry = {
            "id": i + 1,
            "det_range": round(float(dr), 2),
            "det_vel": round(float(dv), 2),
            "mag_db": round(float(20 * np.log10(dm / max_mag + 1e-12)), 2),
            "type": dtype,
        }
        if matched:
            entry["true_range"] = matched["range"]
            entry["true_vel"] = matched["velocity"]
        detections_json.append(entry)

    result_json = {
        "meta": {
            "mu": args.mu,
            "scs_khz": scs_khz,
            "bandwidth_mhz": round(sim.bandwidth / 1e6, 2),
            "range_res_m": round(range_res_m, 2),
            "nms_r_gate": r_gate,
            "seed": args.seed,
            "swerling": args.swerling,
            "multipath": args.multipath,
            "suppress": args.suppress,
        },
        "targets": [
            {"range": t["range"], "velocity": t["velocity"], "rcs": t["rcs"]}
            for t in targets
        ],
        "detections": detections_json,
        "summary": {
            "raw_cfar_count": int(raw_count),
            "nms_count": len(det_ranges),
            "true_count": sum(1 for d in detections_json if d["type"] == "true"),
            "ghost_count": sum(1 for d in detections_json if d["type"] == "ghost"),
        },
        "suppression_metrics": suppression_metrics,   # ← 新增这一行
    }
    json_path = out_dir / f"result_mu{args.mu}.json"
    json_path.write_text(json.dumps(result_json, indent=2, ensure_ascii=False),
                          encoding="utf-8")
    print(f"[output] 结果 JSON: {json_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())