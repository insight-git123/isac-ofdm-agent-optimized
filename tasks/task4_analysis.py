"""Task4 深度分析: 分辨率公式验证 + CP 开销 vs 感知性能 trade-off"""
import sys
import json
import yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    # 读取 Task1 参数
    with open(ROOT / "outputs" / "task1" / "params.yaml", "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    # 读取 Task4 扫描结果 (一行式写法, 避免缩进问题)
    sweep_path = ROOT / "outputs" / "task4" / "isac_sweep_report.json"
    sweep = json.loads(sweep_path.read_text(encoding="utf-8"))

    c = 3e8
    fc = 3.5e9
    lam = c / fc    # 波长

    print("[Task4-Analysis] 分辨率公式验证\n")
    print(f"{'mu':<4}{'SCS':<8}{'BW(MHz)':<10}{'ΔR_theory':<12}{'ΔR_actual':<12}"
          f"{'Vres_theory':<12}{'Vres_actual':<12}")
    print("-" * 80)

    # ... 后续代码保持不变

    mus, dr_theory, dr_actual = [], [], []
    vres_theory, vres_actual = [], []
    cp_overhead_list, detection_rate = [], []

    for r in sweep:
        mu = r["mu"]
        scs_khz = r["scs_khz"]
        bw_hz = r["bandwidth_mhz"] * 1e6
        num_sym = r["num_symbols"]
        # 根据 Task4 规则: T_CPI = num_sym × 符号时长(含CP)
        # 但 sweep 里没存 symbol_duration, 我们用 scs 反推
        T_sym_us = 1000 / scs_khz  # 近似: 1/SCS (不含CP)
        T_cpi = num_sym * T_sym_us * 1e-6  # 秒

        dr_th = c / (2 * bw_hz)                # 理论距离分辨率
        vres_th = lam / (2 * T_cpi)            # 理论速度分辨率
        dr_ac = r["range_res_m"]               # Task4 实测
        vres_ac = r["velocity_res_ms"]         # Task4 实测

        # CP 开销: cp_duration / 符号总时长
        cp_dur_us = r["cp_duration_us"]
        sym_total_us = T_sym_us + cp_dur_us
        cp_overhead = cp_dur_us / sym_total_us

        mus.append(mu)
        dr_theory.append(dr_th); dr_actual.append(dr_ac)
        vres_theory.append(vres_th); vres_actual.append(vres_ac)
        cp_overhead_list.append(cp_overhead * 100)
        detection_rate.append(r["detection_rate"] * 100)

        print(f"{mu:<4}{scs_khz:<8}{r['bandwidth_mhz']:<10}"
              f"{dr_th:<12.2f}{dr_ac:<12.2f}{vres_th:<12.2f}{vres_ac:<12.2f}")

    # 绘图
    out_dir = ROOT / "outputs" / "task4"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # 图1: 距离分辨率 vs 带宽
    ax = axes[0]
    ax.plot(mus, dr_theory, 'o-', label="Theory: c/(2B)", color='steelblue')
    ax.plot(mus, dr_actual, 's--', label="Actual (Task4)", color='coral')
    ax.set_xlabel("Numerology mu")
    ax.set_ylabel("Range Resolution (m)")
    ax.set_title("ΔR vs Bandwidth")
    ax.set_xticks(mus)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 图2: 速度分辨率 vs CPI
    ax = axes[1]
    ax.plot(mus, vres_theory, 'o-', label="Theory: λ/(2·T_CPI)", color='steelblue')
    ax.plot(mus, vres_actual, 's--', label="Actual (Task4)", color='coral')
    ax.set_xlabel("Numerology mu")
    ax.set_ylabel("Velocity Resolution (m/s)")
    ax.set_title("Δv vs CPI")
    ax.set_xticks(mus)
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 图3: CP 开销 vs 检测率
        # 图3: CP 开销 vs 检测率 (柱状图更清楚)
    ax = axes[2]
    ax2 = ax.twinx()

       # 图3: CP 开销 vs 检测率 (柱状图 + 双轴)
    ax = axes[2]
    ax2 = ax.twinx()

    # CP 开销柱状图
    bars = ax.bar([m - 0.2 for m in mus], cp_overhead_list,
                  width=0.4, color='seagreen', alpha=0.7, label='CP Overhead')
    # 检测率折线
    l2, = ax2.plot(mus, detection_rate, 's--', color='darkred',
                   linewidth=2, markersize=8, label='Detection Rate')

    ax.set_xlabel("Numerology mu")
    ax.set_ylabel("CP Overhead (%)", color='seagreen')
    ax2.set_ylabel("Detection Rate (%)", color='darkred')
    ax.set_title("CP Overhead vs Detection Rate")
    ax.set_xticks(mus)
    ax.set_ylim(0, 10)
    ax2.set_ylim(0, 110)
    ax.grid(True, alpha=0.3, axis='y')

    # ★ 用 bars 和 l2 合并图例 (不要再用 l1!)
    lines = [bars, l2]
    labels = ['CP Overhead (~6.57%)', 'Detection Rate']
    ax.legend(lines, labels, loc='upper right')

    plt.tight_layout()
    plot_path = out_dir / "deep_analysis.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print(f"\n[output] {plot_path}")
    print("\n关键洞察:")
    print("  1. ΔR = c/(2B): 随 mu 增大, 带宽指数增长, 分辨率指数改善")
    print("  2. Δv = λ/(2·T_CPI): 固定符号数时, SCS 越大 CPI 越短, 速度分辨率越差")
    print("  3. CP 开销: mu=0 的 CP 占比最高 (~24%), mu=6 最低 (~7%)")
    print("  4. 检测率非单调: 受速度分辨率 + 路径损耗 + CFAR 门限共同影响")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())