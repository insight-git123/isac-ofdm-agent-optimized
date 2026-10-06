"""Task2 补充: 卷积编码 BER 曲线 + 编码增益验证。"""
import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt

from src.simulation.channel_coding import measure_ber


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-bits", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    print(f"[Task2-Coding] (2,1,3) 卷积码 BER 曲线")
    print(f"  每次测试 {args.n_bits} 比特")

    snr_range = [0, 1, 2, 3, 4, 5, 6, 7, 8]

    print("\n  [1/2] 未编码 (BPSK only)...")
    res_uncoded = measure_ber(n_bits=args.n_bits, snr_db_range=snr_range,
                              use_coding=False, seed=args.seed)

    print("  [2/2] 卷积编码 + Viterbi 译码...")
    res_coded = measure_ber(n_bits=args.n_bits, snr_db_range=snr_range,
                            use_coding=True, seed=args.seed)

    # 输出表格
    print("\n" + "=" * 60)
    print(f"{'SNR (dB)':<10}{'BER (uncoded)':<18}{'BER (coded)':<18}{'增益'}")
    print("-" * 60)
    for snr, ber_u, ber_c in zip(res_uncoded["snr_db"],
                                  res_uncoded["ber"],
                                  res_coded["ber"]):
        if ber_c > 0 and ber_u > 0:
            gain = 10 * np.log10(ber_u / ber_c)
            gain_str = f"{gain:+.2f} dB"
        else:
            gain_str = "—"
        print(f"{snr:<10}{ber_u:<18.5f}{ber_c:<18.5f}{gain_str}")
    print("=" * 60)

    # 绘图
    out_dir = ROOT / "outputs" / "task2"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_path = out_dir / "coding_ber_curve.png"

    plt.figure(figsize=(8, 6))
    plt.semilogy(res_uncoded["snr_db"], res_uncoded["ber"],
                 'o--', label='Uncoded (BPSK)', color='steelblue')
    plt.semilogy(res_coded["snr_db"], res_coded["ber"],
                 's-', label='Convolutional (K=3, R=1/2)', color='coral')
    plt.xlabel("SNR (dB)")
    plt.ylabel("Bit Error Rate (BER)")
    plt.title("BER vs SNR: (2,1,3) Convolutional Code")
    plt.grid(True, which='both', alpha=0.3)
    plt.legend()
    plt.ylim(1e-4, 0.5)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print(f"\n[output] {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())