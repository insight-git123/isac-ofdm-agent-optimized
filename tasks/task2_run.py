"""Task2: 读取 Task1 参数，生成 OFDM 波形并绘图（支持 --mu / --modulation）"""
import sys
import argparse
from pathlib import Path
import yaml
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.simulation.ofdm_generator import OFDMGenerator


def main():
    # 1. 命令行参数
    parser = argparse.ArgumentParser(description="Task2: OFDM waveform generation")
    parser.add_argument("--mu", type=int, default=0, help="Numerology mu (0-6), default=0")
    parser.add_argument("--modulation", type=str, default="qpsk",
                        choices=["qpsk", "16qam", "64qam", "256qam"],
                        help="OFDM 频域调制方案")
    args = parser.parse_args()

    # 2. 读取 Task1 参数
    params_path = ROOT / "outputs" / "task1" / "params.yaml"
    if not params_path.exists():
        print("错误: 找不到 params.yaml，请先运行 Task1")
        return 1

    with open(params_path, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    numerology = params["numerology"]
    print(f"[Task2] 读取到 {len(numerology)} 组 numerology 参数")

    # 3. 动态选择 mu
    target = next((n for n in numerology if n["mu"] == args.mu), None)
    if not target:
        print(f"错误: 未找到 mu={args.mu} 的参数，可用: {[n['mu'] for n in numerology]}")
        return 1

    print(f"  使用参数: mu={args.mu}, SCS={target['scs_khz']}kHz, "
          f"CP={target['cp_duration_us']}us, 调制={args.modulation.upper()}")

    # 4. 生成波形
    generator = OFDMGenerator(
        scs_khz=target["scs_khz"],
        cp_duration_us=target["cp_duration_us"],
        modulation=args.modulation,
    )
    waveform = generator.generate(num_symbols=2)

    # 5. 绘图（文件名带 mu + modulation）
    out_dir = ROOT / "outputs" / "task2"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_path = out_dir / f"ofdm_waveform_mu{args.mu}_{args.modulation}.png"

    plt.figure(figsize=(12, 4))
    plt.plot(np.real(waveform), label="Real Part", alpha=0.8)
    plt.plot(np.imag(waveform), label="Imag Part", alpha=0.6)
    plt.title(f"OFDM Waveform (mu={args.mu}, SCS={target['scs_khz']}kHz, "
              f"CP={target['cp_duration_us']}us, {args.modulation.upper()})")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print(f"[output] 波形图已保存至: {plot_path}")
    print(f"[Task2] 成功！总采样点数: {len(waveform)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())