"""Task3-MIMO: 3D-FFT 距离-多普勒-角度 (Range-Doppler-Angle)。

升级 Task3: 从 2D RDM → 3D RDA (Range-Doppler-Angle)。
"""
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.simulation.radar_processing_mimo import MIMORadarSimulator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mu", type=int, default=3)
    parser.add_argument("--n-antennas", type=int, default=8)
    parser.add_argument("--angle-fft", type=int, default=64)
    args = parser.parse_args()

    np.random.seed(42)

    # 三个目标 + 不同角度
    targets = [
        {"range": 150.0, "velocity": 30.0, "rcs": 1.0, "az_deg": -30.0},
        {"range": 300.0, "velocity": -20.0, "rcs": 0.5, "az_deg": 0.0},
        {"range": 450.0, "velocity": 0.0, "rcs": 0.8, "az_deg": 40.0},
    ]

    # 读 numerology
    import yaml
    with open(ROOT / "outputs" / "task1" / "params.yaml", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    num = next((n for n in params["numerology"] if n["mu"] == args.mu), None)
    scs_khz = num["scs_khz"]

    print(f"[Task3-MIMO] 3D-FFT 距离-多普勒-角度 (mu={args.mu}, "
          f"ULA={args.n_antennas} 元)")
    print(f"  目标: {[(t['range'], t['velocity'], t['az_deg']) for t in targets]}")

    # 初始化 MIMO 仿真器
    num_sym = 64 if scs_khz <= 60 else 512
    sim = MIMORadarSimulator(
        scs_khz=scs_khz, fft_size=1024, num_symbols=num_sym,
        fc_ghz=3.5, n_antennas=args.n_antennas, angle_fft_size=args.angle_fft
    )

    print(f"  带宽: {sim.bandwidth/1e6:.2f} MHz, 距离分辨率: {sim.c/(2*sim.bandwidth):.2f} m")
    rayleigh_res = np.degrees(np.arcsin(sim.array.wavelength / (args.n_antennas * sim.array.spacing)))
    print(f"  角度分辨率 (Rayleigh): {rayleigh_res:.1f} deg")

    # 生成 MIMO 回波
    print("\n  [1/2] 生成 MIMO 回波...")
    X, Y_mimo = sim.generate_echo_mimo(
        targets=targets, snr_db=20,
        use_swerling=True, use_multipath=True,
        tdl_model="TDL-A", rms_delay_ns=5.0,
    )

    # 3D-FFT
    print("  [2/2] 3D-FFT (Range × Doppler × Angle)...")
    cube, range_axis, velocity_axis, angle_axis = sim.compute_3d_cube(X, Y_mimo)
    print(f"  Cube shape: {cube.shape}")

    # 找每个目标的峰值位置 (在 cube 里)
    print("\n  峰值检测结果:")
    detections = []
    for tgt in targets:
        # 找该目标真值附近的最大值
        r_idx = np.argmin(np.abs(range_axis - tgt["range"]))
        v_idx = np.argmin(np.abs(velocity_axis - tgt["velocity"]))
        a_idx = np.argmin(np.abs(angle_axis - tgt["az_deg"]))
        # 在真值附近 ±5 格内搜索
        r_slice = slice(max(0, r_idx - 5), min(cube.shape[0], r_idx + 6))
        v_slice = slice(max(0, v_idx - 3), min(cube.shape[1], v_idx + 4))
        a_slice = slice(max(0, a_idx - 3), min(cube.shape[2], a_idx + 4))
        sub = cube[r_slice, v_slice, a_slice]
        idx = np.unravel_index(np.argmax(sub), sub.shape)
        peak_r = range_axis[r_slice][idx[0]]
        peak_v = velocity_axis[v_slice][idx[1]]
        peak_a = angle_axis[a_slice][idx[2]]
        detections.append((peak_r, peak_v, peak_a))
        print(f"    真值 ({tgt['range']:.0f}m, {tgt['velocity']:.0f}m/s, {tgt['az_deg']:.0f}deg) "
              f"→ 检测 ({peak_r:.1f}m, {peak_v:.1f}m/s, {peak_a:.1f}deg)")

    # ========== 绘图 ==========
    out_dir = ROOT / "outputs" / "task3"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_path = out_dir / f"rda_cube_mu{args.mu}.png"

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 视图 1: 距离-多普勒 (所有角度求和)
    rdm_2d = cube.sum(axis=-1)
    rdm_db = 20 * np.log10(rdm_2d / np.max(rdm_2d) + 1e-12)
    ax = axes[0]
    im = ax.imshow(rdm_db, aspect='auto', cmap='jet',
                    extent=[velocity_axis[0], velocity_axis[-1],
                            range_axis[0], range_axis[-1]],
                    origin='lower', vmin=-40, vmax=0)
    ax.set_xlabel("Velocity (m/s)"); ax.set_ylabel("Range (m)")
    ax.set_title("Range-Doppler (Angle-summed)")
    plt.colorbar(im, ax=ax)
    ax.set_xlim(-60, 60); ax.set_ylim(0, 600)

    # 视图 2: 距离-角度 (在目标速度处切面)
    v_center_idx = np.argmin(np.abs(velocity_axis - 0))
    ra_slice = cube[:, v_center_idx - 1:v_center_idx + 2, :].sum(axis=1)
    ra_db = 20 * np.log10(ra_slice / np.max(ra_slice) + 1e-12)
    ax = axes[1]
    im = ax.imshow(ra_db, aspect='auto', cmap='jet',
                    extent=[angle_axis[0], angle_axis[-1],
                            range_axis[0], range_axis[-1]],
                    origin='lower', vmin=-40, vmax=0)
    ax.set_xlabel("Azimuth Angle (deg)"); ax.set_ylabel("Range (m)")
    ax.set_title("Range-Angle (Doppler≈0)")
    plt.colorbar(im, ax=ax)
    ax.set_xlim(-90, 90); ax.set_ylim(0, 600)

    # 视图 3: 速度-角度 (在目标2距离处切面)
    r_center_idx = np.argmin(np.abs(range_axis - 300))
    va_slice = cube[r_center_idx - 1:r_center_idx + 2, :, :].sum(axis=0)
    va_db = 20 * np.log10(va_slice / np.max(va_slice) + 1e-12)
    ax = axes[2]
    im = ax.imshow(va_db.T, aspect='auto', cmap='jet',
                    extent=[velocity_axis[0], velocity_axis[-1],
                            angle_axis[0], angle_axis[-1]],
                    origin='lower', vmin=-40, vmax=0)
    ax.set_xlabel("Velocity (m/s)"); ax.set_ylabel("Azimuth Angle (deg)")
    ax.set_title("Velocity-Angle (Range≈300m)")
    plt.colorbar(im, ax=ax)
    ax.set_xlim(-60, 60); ax.set_ylim(-90, 90)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print(f"\n[output] {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())