"""Task3 方向二: 标准 3GPP TR 38.901 TDL 信道验证"""
import sys
import yaml
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.simulation.ofdm_generator import OFDMGenerator
from src.simulation.pulse_compression import (
    matched_filter_fft, generate_time_domain_echo,
    add_awgn, range_axis_from_compression, find_peaks_1d
)
from src.simulation.tdl_model import apply_tdl_channel, print_tdl_summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mu", type=int, default=3)
    parser.add_argument("--model", type=str, default="TDL-A",
                        choices=["TDL-A", "TDL-B", "TDL-C", "TDL-D"])
    parser.add_argument("--snr", type=float, default=20.0)
    args = parser.parse_args()

    np.random.seed(42)

    params_path = ROOT / "outputs" / "task1" / "params.yaml"
    with open(params_path, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    num = next((n for n in params["numerology"] if n["mu"] == args.mu), None)
    if not num:
        print(f"错误: 未找到 mu={args.mu}")
        return 1

    scs_khz = num["scs_khz"]
    cp_dur = num["cp_duration_us"]

    print(f"[Task3-TDL] 标准 3GPP TDL 信道仿真 (mu={args.mu}, "
          f"SCS={scs_khz}kHz, {args.model}, SNR={args.snr}dB)")
    print_tdl_summary(args.model)

    # 生成发射信号
    gen = OFDMGenerator(scs_khz=scs_khz, cp_duration_us=cp_dur, fft_size=1024)
    fs = gen.sampling_rate
    tx = gen.generate(num_symbols=1)

    # 目标
    targets = [
        {"range": 150.0, "velocity": 30.0, "rcs": 1.0},
        {"range": 300.0, "velocity": -20.0, "rcs": 0.5},
        {"range": 450.0, "velocity": 0.0, "rcs": 0.8},
    ]

    # 1. 生成干净回波 (仅时延 + 多普勒)
    rx_los = generate_time_domain_echo(tx, targets, fs=fs, fc=3.5e9)

    # 2. 应用 TDL 信道 (真实多径叠加)
    rx_tdl = apply_tdl_channel(rx_los, fs=fs, model=args.model)

    # 3. 加噪
    rx = add_awgn(rx_tdl, snr_db=args.snr)

    # 4. 脉冲压缩
    y_los = matched_filter_fft(tx, add_awgn(rx_los, args.snr), window='hamming')
    y_tdl = matched_filter_fft(tx, rx, window='hamming')
    range_axis = range_axis_from_compression(len(y_tdl), fs)

    peaks_los = find_peaks_1d(y_los, min_height_ratio=0.2)
    peaks_tdl = find_peaks_1d(y_tdl, min_height_ratio=0.15)

    print(f"\n  采样率: {fs/1e6:.2f} MHz, 距离分辨率: {3e8/(2*fs):.3f} m")

    print(f"\n  [对比] 仅 LOS 时延 (无多径):")
    for i, p in enumerate(peaks_los):
        if range_axis[p] <= 600:
            print(f"    [{i+1}] {range_axis[p]:.1f}m")

    print(f"\n  [对比] TDL-{args.model[-1]} 多径信道:")
    for i, p in enumerate(peaks_tdl):
        if range_axis[p] <= 600:
            print(f"    [{i+1}] {range_axis[p]:.1f}m")

    # 绘图
    out_dir = ROOT / "outputs" / "task3"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # 上图: LOS only
    ax = axes[0]
    mag = 20 * np.log10(np.abs(y_los) / np.max(np.abs(y_los)) + 1e-12)
    mask = range_axis <= 600
    ax.plot(range_axis[mask], mag[mask], color='steelblue', linewidth=1)
    for tgt in targets:
        ax.axvline(tgt["range"], color='red', linestyle='--', alpha=0.4)
    ax.set_xlabel("Range (m)")
    ax.set_ylabel("Normalized Magnitude (dB)")
    ax.set_title(f"LOS-only (mu={args.mu}, {args.snr}dB)")
    ax.set_ylim(-60, 5); ax.grid(True, alpha=0.3)

    # 下图: TDL
    ax = axes[1]
    mag = 20 * np.log10(np.abs(y_tdl) / np.max(np.abs(y_tdl)) + 1e-12)
    ax.plot(range_axis[mask], mag[mask], color='darkorange', linewidth=1)
    for tgt in targets:
        ax.axvline(tgt["range"], color='red', linestyle='--', alpha=0.4)
    ax.set_xlabel("Range (m)")
    ax.set_ylabel("Normalized Magnitude (dB)")
    ax.set_title(f"After {args.model} Multipath (mu={args.mu}, {args.snr}dB)")
    ax.set_ylim(-60, 5); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = out_dir / f"tdl_{args.model}_mu{args.mu}.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print(f"\n[output] {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())