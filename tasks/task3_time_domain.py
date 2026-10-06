"""Task3 方向一: 时域脉冲压缩验证 - 单目标 + 多目标 + 加窗对比"""
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


def get_numerology(params, mu):
    return next((n for n in params["numerology"] if n["mu"] == mu), None)


def run_single_target(scs_khz, cp_dur_us, mu, snr_db=20):
    """单目标验证：脉冲压缩峰值应精确落在目标距离。"""
    gen = OFDMGenerator(scs_khz=scs_khz, cp_duration_us=cp_dur_us, fft_size=1024)
    fs = gen.sampling_rate

    # 生成一个含 CP 的 OFDM 符号作为参考
    tx = gen.generate(num_symbols=1)

    # 单目标: 200m, 20m/s
    targets = [{"range": 200.0, "velocity": 20.0, "rcs": 1.0}]
    rx_clean = generate_time_domain_echo(tx, targets, fs=fs, fc=3.5e9)
    rx = add_awgn(rx_clean, snr_db=snr_db)

    # 脉冲压缩（不加窗 + 加汉明窗）
    y_no_win = matched_filter_fft(tx, rx, window=None)
    y_hamming = matched_filter_fft(tx, rx, window='hamming')

    range_axis = range_axis_from_compression(len(y_no_win), fs)

    # 峰值检测
    peaks_no_win = find_peaks_1d(y_no_win, min_height_ratio=0.3)
    peaks_hamming = find_peaks_1d(y_hamming, min_height_ratio=0.3)

    return {
        "mu": mu, "fs": fs, "targets": targets,
        "tx": tx, "rx": rx, "y_no_win": y_no_win, "y_hamming": y_hamming,
        "range_axis": range_axis,
        "peaks_no_win": peaks_no_win, "peaks_hamming": peaks_hamming,
    }


def run_multi_target(scs_khz, cp_dur_us, mu, snr_db=20):
    """多目标验证：3 个目标应产生 3 个清晰峰值。"""
    gen = OFDMGenerator(scs_khz=scs_khz, cp_duration_us=cp_dur_us, fft_size=1024)
    fs = gen.sampling_rate
    tx = gen.generate(num_symbols=1)

    targets = [
        {"range": 150.0, "velocity": 30.0, "rcs": 1.0},
        {"range": 300.0, "velocity": -20.0, "rcs": 0.5},
        {"range": 450.0, "velocity": 0.0, "rcs": 0.8},
    ]
    rx_clean = generate_time_domain_echo(tx, targets, fs=fs, fc=3.5e9)
    rx = add_awgn(rx_clean, snr_db=snr_db)

    y = matched_filter_fft(tx, rx, window=None)
    range_axis = range_axis_from_compression(len(y), fs)
    peaks = find_peaks_1d(y, min_height_ratio=0.2)

    return {
        "mu": mu, "fs": fs, "targets": targets,
        "y": y, "range_axis": range_axis, "peaks": peaks,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mu", type=int, default=3,
                        help="Numerology mu (建议 >= 2 保证距离分辨率)")
    parser.add_argument("--snr", type=float, default=20.0)
    args = parser.parse_args()

    np.random.seed(42)

    params_path = ROOT / "outputs" / "task1" / "params.yaml"
    with open(params_path, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    num = get_numerology(params, args.mu)
    if not num:
        print(f"错误: 未找到 mu={args.mu}")
        return 1

    scs_khz = num["scs_khz"]
    cp_dur = num["cp_duration_us"]

    print(f"[Task3-TimeDomain] 时域脉冲压缩验证 (mu={args.mu}, "
          f"SCS={scs_khz}kHz, SNR={args.snr}dB)")

    # ========== 单目标验证 ==========
    print("\n[1/2] 单目标验证 (200m, 20m/s)...")
    r1 = run_single_target(scs_khz, cp_dur, args.mu, snr_db=args.snr)
    fs = r1["fs"]
    print(f"  采样率: {fs/1e6:.2f} MHz, 距离分辨率: {3e8/(2*fs):.3f} m")

    # ★ 修复: 拆出列表推导, 避免 f-string 内嵌套引号
    peaks_no_win_m = [r1["range_axis"][p] for p in r1["peaks_no_win"]]
    peaks_hamming_m = [r1["range_axis"][p] for p in r1["peaks_hamming"]]
    str_no_win = ", ".join(f"{d:.1f}m" for d in peaks_no_win_m) if peaks_no_win_m else "未检测到"
    str_hamming = ", ".join(f"{d:.1f}m" for d in peaks_hamming_m) if peaks_hamming_m else "未检测到"
    print(f"  不加窗峰值距离: {str_no_win}")
    print(f"  汉明窗峰值距离: {str_hamming}")

    # ========== 多目标验证 ==========
    print("\n[2/2] 多目标验证 (3 个目标)...")
    r2 = run_multi_target(scs_khz, cp_dur, args.mu, snr_db=args.snr)
    peak_ranges = [r2["range_axis"][p] for p in r2["peaks"]]
    print(f"  检测到 {len(r2['peaks'])} 个峰值:")
    for i, pr in enumerate(peak_ranges):
        print(f"    [{i+1}] {pr:.1f}m")

    # ========== 绘图 ==========
    out_dir = ROOT / "outputs" / "task3"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # 单目标对比 (不加窗 vs 汉明窗)
    ax = axes[0]
    mag_no_win = 20 * np.log10(np.abs(r1["y_no_win"]) / np.max(np.abs(r1["y_no_win"])) + 1e-12)
    mag_hamming = 20 * np.log10(np.abs(r1["y_hamming"]) / np.max(np.abs(r1["y_hamming"])) + 1e-12)

    mask = r1["range_axis"] <= 500
    ax.plot(r1["range_axis"][mask], mag_no_win[mask],
            label="No Window", alpha=0.8, linewidth=1)
    ax.plot(r1["range_axis"][mask], mag_hamming[mask],
            label="Hamming Window", alpha=0.8, linewidth=1)
    ax.axvline(200, color='red', linestyle='--', alpha=0.5, label='True Range (200m)')
    ax.set_xlabel("Range (m)")
    ax.set_ylabel("Normalized Magnitude (dB)")
    ax.set_title(f"Single Target Pulse Compression (mu={args.mu}, SNR={args.snr}dB)")
    ax.set_ylim(-60, 5)
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 多目标
    ax = axes[1]
    mag_multi = 20 * np.log10(np.abs(r2["y"]) / np.max(np.abs(r2["y"])) + 1e-12)
    mask = r2["range_axis"] <= 600
    ax.plot(r2["range_axis"][mask], mag_multi[mask],
            label="Compressed Profile", color='steelblue', linewidth=1)

    for tgt in r2["targets"]:
        ax.axvline(tgt["range"], color='red', linestyle='--', alpha=0.5)

    for i, p in enumerate(r2["peaks"]):
        if r2["range_axis"][p] <= 600:
            ax.plot(r2["range_axis"][p], mag_multi[p], 'gv', markersize=10,
                    label='Detected Peak' if i == 0 else None)

    ax.set_xlabel("Range (m)")
    ax.set_ylabel("Normalized Magnitude (dB)")
    ax.set_title(f"Multi-Target Pulse Compression (mu={args.mu}, SNR={args.snr}dB)")
    ax.set_ylim(-60, 5)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plot_path = out_dir / f"pulse_compression_mu{args.mu}.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print(f"\n[output] {plot_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())