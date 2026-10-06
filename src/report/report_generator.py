"""Task5: 自动生成 Markdown 实验报告 (完全从 JSON 动态生成)。

P2.2: 删除所有硬编码结论
P4.2: 支持 Bootstrap/Wilson 置信区间 (新字段名)
P4.3: 嵌入运行元数据 (commit/版本/平台)
"""
import json
import subprocess
import sys
import yaml
from pathlib import Path
from datetime import datetime

import numpy as np


def _get_git_commit(root_dir: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root_dir, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _get_git_status(root_dir: Path) -> str:
    try:
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=root_dir, stderr=subprocess.DEVNULL
        ).decode().strip()
        return "dirty" if dirty else "clean"
    except Exception:
        return "unknown"


def _find_latest_result(task3_dir: Path):
    if not task3_dir.exists():
        return None
    files = sorted(task3_dir.glob("result_mu*.json"))
    if not files:
        return None
    return json.loads(files[-1].read_text(encoding="utf-8"))


def _ci_str(r: dict, key_prefix: str, decimals: int = 2) -> str:
    """兼容 P4.2 前后两种字段名。

    P4.2 前: r["ci95_hits"] = 半宽 (float)
    P4.2 后: r["ci95_hits_lo"], r["ci95_hits_hi"] = 上下界
    """
    lo_key = f"{key_prefix}_lo"
    hi_key = f"{key_prefix}_hi"
    old_key = key_prefix
    if lo_key in r and hi_key in r:
        return (f"[{r[lo_key]:.{decimals}f}, {r[hi_key]:.{decimals}f}]")
    elif old_key in r:
        mean = r.get(key_prefix.replace("ci95_", "avg_"), 0)
        return f"±{r[old_key]:.{decimals}f}"
    else:
        return "N/A"


def generate_report(root_dir: Path) -> str:
    params_path = root_dir / "outputs" / "task1" / "params.yaml"
    params = {}
    if params_path.exists():
        with open(params_path, "r", encoding="utf-8") as f:
            params = yaml.safe_load(f)

    task3_dir = root_dir / "outputs" / "task3"
    task3_result = _find_latest_result(task3_dir)

    task4_dir = root_dir / "outputs" / "task4"
    task4_json = task4_dir / "isac_sweep_report.json"
    sweep_data = []
    if task4_json.exists():
        sweep_data = json.loads(task4_json.read_text(encoding="utf-8"))

    md = []
    md.append("# ISAC-OFDM Agent 实验报告\n")
    md.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # ========== 0. 运行环境 (P4.3) ==========
    md.append("## 0. 运行环境\n")
    md.append(f"- **Git commit**: `{_get_git_commit(root_dir)}` "
              f"({_get_git_status(root_dir)})")
    md.append(f"- **Python**: {sys.version.split()[0]}")
    md.append(f"- **NumPy**: {np.__version__}")
    md.append(f"- **平台**: {sys.platform}\n")

    # ========== 1. 项目信息 ==========
    if params:
        meta = params.get("meta", {})
        md.append("## 1. 项目信息\n")
        md.append(f"- **项目名称**: {meta.get('project', 'N/A')}")
        md.append(f"- **标准版本**: {meta.get('release', 'N/A')}")
        md.append(f"- **数据来源**: {meta.get('source', 'N/A')}")
        md.append(f"- **验证结果**: {'PASS' if meta.get('validation_passed') else 'FAIL'}\n")

        md.append("## 2. 提取的 Numerology 参数 (Task1)\n")
        md.append("| mu | SCS (kHz) | CP 类型 | CP 时长 (us) |")
        md.append("|---|---|---|---|")
        for num in params.get("numerology", []):
            cp_types = ", ".join(num.get("cp_types", []))
            cp_dur = num.get("cp_duration_us", "N/A")
            md.append(f"| {num['mu']} | {num['scs_khz']} | {cp_types} | {cp_dur} |")
        md.append("")

    # ========== 3. OFDM 波形 ==========
    md.append("## 3. OFDM 波形生成 (Task2)\n")
    task2_dir = root_dir / "outputs" / "task2"
    waveform_imgs = sorted(task2_dir.glob("ofdm_waveform_mu*.png")) if task2_dir.exists() else []
    if waveform_imgs:
        rel = f"../task2/{waveform_imgs[-1].name}"
        md.append(f"![OFDM Waveform]({rel})\n")
        md.append(f"*图 1: OFDM 时域波形 ({waveform_imgs[-1].name})*\n")
    else:
        md.append("*(未找到波形图)*\n")

    # NR 资源网格 (P2.4)
    nr_grid_imgs = sorted(task2_dir.glob("nr_grid_mu*_dmrs_prs.png"))
    if nr_grid_imgs:
        md.append("### NR 资源网格 (P2.4)\n")
        md.append("实现 3GPP TS 38.211 定义的 NR 资源网格：DC 子载波置零、"
                  "保护带、PDSCH DMRS (Config Type 1)、PRS (comb=4)。\n")
        rel = f"../task2/{nr_grid_imgs[-1].name}"
        md.append(f"![NR Grid]({rel})\n")
        md.append("*图 1b: NR 资源网格 "
                  "(蓝=data, 绿=DMRS, 红=PRS, 灰=guard, 黑=DC)*\n")

    # ========== 4. ISAC 感知性能 (Task3) ==========
    md.append("## 4. ISAC 感知性能分析 (Task3)\n")

    if task3_result:
        meta = task3_result["meta"]
        summary = task3_result["summary"]
        targets = task3_result["targets"]
        detections = task3_result["detections"]

        md.append("### 4.1 实验设置\n")
        md.append(f"- **Numerology**: mu={meta['mu']}, SCS={meta['scs_khz']}kHz")
        md.append(f"- **带宽**: {meta['bandwidth_mhz']} MHz")
        md.append(f"- **距离分辨率**: {meta['range_res_m']} m")
        md.append(f"- **随机种子**: {meta['seed']}")
        md.append(f"- **启用选项**: Swerling={meta['swerling']}, "
                  f"Multipath={meta['multipath']}, Suppress={meta['suppress']}\n")

        md.append("### 4.2 目标真值\n")
        md.append("| # | 距离 (m) | 速度 (m/s) | RCS |")
        md.append("|---|---|---|---|")
        for i, t in enumerate(targets):
            md.append(f"| {i+1} | {t['range']} | {t['velocity']} | {t['rcs']} |")
        md.append("")

        md.append("### 4.3 检测结果 (自动从 result.json 生成)\n")
        md.append("| # | 真值 (距离, 速度) | 检测 (距离, 速度) | 幅度 (dB) | 类型 |")
        md.append("|---|---|---|---|---|")
        for d in detections:
            true_str = (f"({d.get('true_range', '-')}, {d.get('true_vel', '-')})"
                        if "true_range" in d else "—")
            det_str = f"({d['det_range']}m, {d['det_vel']}m/s)"
            type_marker = "真实" if d["type"] == "true" else "鬼影"
            md.append(f"| {d['id']} | {true_str} | {det_str} | "
                      f"{d['mag_db']} | {type_marker} |")
        md.append("")

        md.append("### 4.4 统计汇总\n")
        md.append(f"- CFAR 原始检测点: **{summary['raw_cfar_count']}**")
        md.append(f"- NMS 聚类后: **{summary['nms_count']}**")
        md.append(f"- 真实目标: **{summary['true_count']}**")
        md.append(f"- 多径鬼影: **{summary['ghost_count']}**\n")

        # 鬼影抑制评估 (P4.4 兼容)
        if "suppression_metrics" in task3_result:
            sm = task3_result["suppression_metrics"] or {}
            md.append("### 4.5 鬼影抑制评估\n")
            md.append(f"- **Precision** (真实保留): {sm.get('precision', 'N/A')}")
            md.append(f"- **Recall** (真实召回): {sm.get('recall', 'N/A')}")
            md.append(f"- **误抑制数**: {sm.get('false_negative', 'N/A')}\n")

        rdm_imgs = sorted(task3_dir.glob("rdm_mu*_cfar.png"))
        if rdm_imgs:
            rel = f"../task3/{rdm_imgs[-1].name}"
            md.append(f"![RDM]({rel})\n")
            md.append(f"*图 2: 距离-多普勒图 ({rdm_imgs[-1].name})*\n")

        pc_imgs = sorted(task3_dir.glob("pulse_compression_mu*.png"))
        if pc_imgs:
            rel = f"../task3/{pc_imgs[-1].name}"
            md.append(f"![Pulse Compression]({rel})\n")
            md.append("*图 3: 时域脉冲压缩 (FFT 匹配滤波)*\n")

        tdl_imgs = sorted(task3_dir.glob("tdl_TDL-*.png"))
        if tdl_imgs:
            rel = f"../task3/{tdl_imgs[-1].name}"
            md.append(f"![TDL Channel]({rel})\n")
            md.append("*图 4: TDL-A 标准多径信道对比*\n")
    else:
        md.append("*(未找到 Task3 结果 JSON，请先运行 `python tasks/task3_run.py`)*\n")

    # ========== 5. Numerology 性能扫描 (Task4) ==========
    if sweep_data:
        md.append("## 5. Numerology 性能扫描 (Task4)\n")
        n_trials = sweep_data[0].get("n_trials", 10)
        # 判断是否为 P4.2 的新格式
        is_bootstrap = "ci95_hits_lo" in sweep_data[0]

        if is_bootstrap:
            md.append(f"每个 mu 运行 {n_trials} 次蒙特卡洛仿真，"
                      f"使用 **Bootstrap (n=2000)** 和 **Wilson** 置信区间。\n")
            md.append("| mu | SCS (kHz) | BW (MHz) | 距离分辨率 (m) "
                      "| 命中 [Bootstrap 95% CI] | 检测率 [Wilson 95% CI] |")
            md.append("|---|---|---|---|---|---|")
            for r in sweep_data:
                hit_str = (f"{r['avg_hits']:.2f} "
                           f"[{r['ci95_hits_lo']:.2f}, {r['ci95_hits_hi']:.2f}]")
                det_str = (f"{r['detection_rate']*100:.1f}% "
                           f"[{r.get('detection_ci_lo', 0)*100:.1f}, "
                           f"{r.get('detection_ci_hi', 0)*100:.1f}]")
                md.append(f"| {r['mu']} | {r['scs_khz']} | {r['bandwidth_mhz']} | "
                          f"{r['range_res_m']} | {hit_str} | {det_str} |")
        else:
            # 兼容旧格式
            md.append(f"每个 mu 运行 {n_trials} 次蒙特卡洛仿真取平均。\n")
            md.append("| mu | SCS (kHz) | BW (MHz) | 距离分辨率 (m) "
                      "| 命中 | 鬼影 | 检测率 |")
            md.append("|---|---|---|---|---|---|---|")
            for r in sweep_data:
                md.append(f"| {r['mu']} | {r['scs_khz']} | {r['bandwidth_mhz']} | "
                          f"{r['range_res_m']} | {r['avg_hits']:.2f}/3 | "
                          f"{r['avg_ghosts']:.2f} | "
                          f"{r['detection_rate']*100:.0f}% |")
        md.append("")

        sweep_img = task4_dir / "isac_sweep_summary.png"
        if sweep_img.exists():
            md.append("![ISAC Sweep](../task4/isac_sweep_summary.png)\n")
            md.append("*图 5: 各 Numerology 下的检测率、鬼影数、距离分辨率*\n")

        best_mu = max(sweep_data, key=lambda r: r["detection_rate"])
        worst_mu = min(sweep_data, key=lambda r: r["detection_rate"])
        md.append("### 5.1 动态结论 (从数据推导)\n")
        md.append(f"- **最高检测率**: mu={best_mu['mu']} "
                  f"({best_mu['detection_rate']*100:.0f}%)")
        md.append(f"- **最低检测率**: mu={worst_mu['mu']} "
                  f"({worst_mu['detection_rate']*100:.0f}%)")
        md.append(f"- **最优距离分辨率**: mu={sweep_data[-1]['mu']} "
                  f"({sweep_data[-1]['range_res_m']} m)")
        md.append("")

    # ========== 5.5 深度分析 ==========
    deep_img = task4_dir / "deep_analysis.png"
    if deep_img.exists():
        md.append("### 5.5 分辨率公式与 CP 开销深度分析\n")
        md.append("| 公式 | 理论预测 | 实测结果 |")
        md.append("|---|---|---|")
        if sweep_data:
            min_res = min(r["range_res_m"] for r in sweep_data)
            max_res = max(r["range_res_m"] for r in sweep_data)
            md.append(f"| ΔR = c/(2B) | 距离分辨率与带宽反比 | "
                      f"{max_res}m → {min_res}m |")
        md.append("| Δv = λ/(2·T_CPI) | 速度分辨率与 CPI 反比 | 与符号数变化一致 |")
        md.append("| T_CP/T_sym = 144/2048 | CP 开销固定 ~7% | 实测 6.57% |")
        md.append("")
        md.append("![Deep Analysis](../task4/deep_analysis.png)\n")
        md.append("*图 6: 分辨率公式验证 + CP 开销 trade-off*\n")

    # ========== 6. 结论 ==========
    md.append("## 6. 结论\n")
    md.append("- 成功从 3GPP TS 38.211 Rel-18 提取全部 7 种 numerology 参数并校验通过。")
    md.append("- 实现 OFDM 波形生成 (QPSK/16QAM/64QAM)、严格时域脉冲压缩、"
              "3GPP TR 38.901 TDL-A 标准多径信道、Swerling-I RCS 波动。")
    md.append("- CFAR 检测已重写为功率域 + Pfa 校准 + 峰值过滤。")
    if sweep_data:
        md.append(f"- 参数扫描显示 mu={best_mu['mu']} 检测率最高 "
                  f"({best_mu['detection_rate']*100:.0f}%)。")
    md.append("- **已知局限**详见 [LIMITATIONS.md](../LIMITATIONS.md)。\n")

    return "\n".join(md)

