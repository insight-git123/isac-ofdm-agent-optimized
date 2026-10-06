"""统一 CLI：python -m isac_ofdm <command> [options]。

子命令：
    extract   - 从 3GPP PDF/样例文本提取参数
    simulate  - 波形生成、RDM、MIMO 仿真
    sweep     - 参数扫描 + 基准对比
    report    - 生成 Markdown 报告
    all       - 完整流水线
"""
import argparse
import io
import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run_task(script_name: str, extra_args: list = None) -> int:
    """调用 tasks/ 下的脚本。"""
    script_path = ROOT / "tasks" / script_name
    if not script_path.exists():
        print(f"错误: 脚本不存在 {script_path}")
        return 1
    cmd = [sys.executable, str(script_path)] + (extra_args or [])
    print(f"  → {' '.join([script_name] + (extra_args or []))}")
    return subprocess.call(cmd)


def cmd_extract(args):
    print("[extract] 从 3GPP 文档提取参数")
    extra = []
    if args.raw:
        extra.extend(["--raw", args.raw])
    return _run_task("task1_run.py", extra)


def cmd_simulate(args):
    print(f"[simulate] 波形/感知仿真 (mu={args.mu})")
    rc = 0
    if not args.skip_waveform:
        rc |= _run_task("task2_run.py", ["--mu", str(args.mu)])
        rc |= _run_task("task2_nr_grid.py", ["--mu", str(args.mu)])
    if args.mimo:
        rc |= _run_task("task3_mimo.py", ["--mu", str(args.mu)])
    if args.cp_ofdm:
        rc |= _run_task("task3_cp_ofdm.py", ["--mu", str(args.mu)])
    extra = ["--mu", str(args.mu)]
    if args.swerling: extra.append("--swerling")
    if args.multipath: extra.append("--multipath")
    if args.suppress: extra.append("--suppress")
    rc |= _run_task("task3_run.py", extra)
    return rc


def cmd_sweep(args):
    print(f"[sweep] 参数扫描 + 基准对比 (MC={args.mc_trials})")
    rc = _run_task("task4_run.py", ["--mc-trials", str(args.mc_trials)])
    rc |= _run_task("task4_baseline.py", ["--n-trials", str(max(5, args.mc_trials // 2))])
    rc |= _run_task("task4_analysis.py")
    return rc


def cmd_report(args):
    print("[report] 生成 Markdown 报告")
    return _run_task("task5_run.py")


def cmd_all(args):
    """完整流水线 (extract → simulate → sweep → report)。"""
    print("=" * 60)
    print("  ISAC-OFDM Agent 完整流水线")
    print("=" * 60)
    rc = 0
    rc |= cmd_extract(args)
    if rc != 0:
        print("❌ extract 失败，中止")
        return rc
    rc |= cmd_simulate(args)
    rc |= cmd_sweep(args)
    rc |= cmd_report(args)
    if rc == 0:
        print("\n" + "=" * 60)
        print("  ✅ 全部完成！")
        print(f"  报告: {ROOT}/outputs/task5/experiment_report.md")
        print("=" * 60)
    return rc


def build_parser():
    parser = argparse.ArgumentParser(
        prog="isac_ofdm",
        description="ISAC-OFDM Agent 统一 CLI"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # extract
    p_extract = sub.add_parser("extract", help="3GPP 参数提取")
    p_extract.add_argument("--raw", type=str, default=None,
                            help="PDF/TXT 输入路径")
    p_extract.set_defaults(func=cmd_extract)

    # simulate
    p_sim = sub.add_parser("simulate", help="波形与感知仿真")
    p_sim.add_argument("--mu", type=int, default=3)
    p_sim.add_argument("--swerling", action="store_true")
    p_sim.add_argument("--multipath", action="store_true")
    p_sim.add_argument("--suppress", action="store_true")
    p_sim.add_argument("--mimo", action="store_true", help="额外运行 MIMO 3D")
    p_sim.add_argument("--cp-ofdm", action="store_true", help="额外运行 CP-OFDM 全链路")
    p_sim.add_argument("--skip-waveform", action="store_true")
    p_sim.set_defaults(func=cmd_simulate)

    # sweep
    p_sweep = sub.add_parser("sweep", help="参数扫描与基准对比")
    p_sweep.add_argument("--mc-trials", type=int, default=50)
    p_sweep.set_defaults(func=cmd_sweep)

    # report
    p_rep = sub.add_parser("report", help="生成报告")
    p_rep.set_defaults(func=cmd_report)

    # all
    p_all = sub.add_parser("all", help="完整流水线")
    p_all.add_argument("--mu", type=int, default=3)
    p_all.add_argument("--mc-trials", type=int, default=50)
    p_all.add_argument("--swerling", action="store_true", default=True)
    p_all.add_argument("--multipath", action="store_true", default=True)
    p_all.add_argument("--suppress", action="store_true", default=True)
    p_all.add_argument("--mimo", action="store_true", default=True)
    p_all.add_argument("--cp-ofdm", action="store_true", default=True)
    p_all.add_argument("--skip-waveform", action="store_true")
    p_all.add_argument("--raw", type=str, default=None)
    p_all.set_defaults(func=cmd_all)

    return parser


def main():
    # Configure UTF-8 only for the executable CLI; importing this module must not replace pytest capture streams.
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", write_through=True)
    parser = build_parser()
    args = parser.parse_args()
    rc = args.func(args)
    sys.exit(rc)


if __name__ == "__main__":
    main()



