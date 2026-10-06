"""Task3 补充: CP-OFDM 全链路验证 (时域 vs 频域等效性)。"""
import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt

from src.simulation.cp_ofdm_link import CPOFDMLink
from src.simulation.time_domain_channel import TimeDomainTDLChannel
from src.simulation.modulation import modulate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mu", type=int, default=3)
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--n-sym", type=int, default=8)
    parser.add_argument("--rms-delay-ns", type=float, default=30.0)
    args = parser.parse_args()

    np.random.seed(42)

    import yaml
    with open(ROOT / "outputs" / "task1" / "params.yaml", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    num = next((n for n in params["numerology"] if n["mu"] == args.mu), None)
    scs_khz = num["scs_khz"]
    cp_us = num["cp_duration_us"]

    fs_hz = args.n_fft * scs_khz * 1e3
    cp_samples = int(round(cp_us * 1e-6 * fs_hz))

    print(f"[Task3-CPOFDM] CP-OFDM 全链路验证")
    print(f"  mu={args.mu}, SCS={scs_khz}kHz")
    print(f"  n_fft={args.n_fft}, cp_samples={cp_samples}, fs={fs_hz/1e6:.2f} MHz")
    print(f"  TDL 模型: {args.rms_delay_ns}ns RMS, n_sym={args.n_sym}\n")

    # ========== 1. 构造链路 ==========
    link = CPOFDMLink(n_fft=args.n_fft, cp_len=cp_samples)
    channel = TimeDomainTDLChannel(
        model="TDL-A", rms_delay_ns=args.rms_delay_ns,
        fs_hz=fs_hz, normalize_power=True
    )
    max_delay_samples = channel.maximum_delay_samples()
    print(f"  [准备] 信道最大时延: {max_delay_samples} 采样 "
          f"({channel.maximum_delay_ns():.1f} ns)")
    cp_ok = "PASS" if cp_samples >= max_delay_samples else "FAIL"
    print(f"  [验证] CP长度 ({cp_samples}) >= 最大时延 ({max_delay_samples}): {cp_ok}\n")

    # ========== 2. 频域数据 ==========
    n_total = args.n_fft * args.n_sym
    qpsk_flat = modulate(n_total, "qpsk")
    X_tx = qpsk_flat.reshape(args.n_fft, args.n_sym)

    # 离散 FIR 的频响（正确的参考模型）
    H_ref = channel.frequency_response(args.n_fft).flatten()

    # ========== 3. 测试 1: 无信道理想回环 ==========
    print("  [测试 1] 无信道理想回环...")
    tx_signal = link.transmit(X_tx)
    rx_signal = link.receive(tx_signal, args.n_sym)
    err_ideal = np.mean(np.abs(rx_signal - X_tx) ** 2) / np.mean(np.abs(X_tx) ** 2)
    print(f"    相对误差 (MSE): {err_ideal:.2e}")
    assert err_ideal < 1e-20

    # ========== 4. 测试 2: 时域信道 → 频域估计 (低频段) ==========
    print("\n  [测试 2] 时域 TDL 信道 → 频域信道估计 (低频段)...")
    rx_clean = channel.apply(tx_signal)
    snr_db = 30
    sig_pow = np.mean(np.abs(rx_clean) ** 2)
    noise_pow = sig_pow / (10 ** (snr_db / 10))
    rx_noisy = rx_clean + (np.random.randn(*rx_clean.shape) +
                            1j * np.random.randn(*rx_clean.shape)) * np.sqrt(noise_pow / 2)

    X_rx = link.receive(rx_noisy, args.n_sym)
    H_est = X_rx * np.conj(X_tx) / (np.abs(X_tx) ** 2 + 1e-12)
    H_est_mean = H_est.mean(axis=1)

    # 只对比低频段 (量化误差小)
       # P4.1: 全频段对比 (频率轴修复后应对齐)
    n_low = args.n_fft
    corr = np.corrcoef(np.abs(H_est_mean[:n_low]),
                       np.abs(H_ref[:n_low]))[0, 1]
    print(f"    全频段 ({n_low} 子载波) 幅度相关系数: {corr:.4f}")
    corr = np.corrcoef(np.abs(H_est_mean[:n_low]),
                       np.abs(H_ref[:n_low]))[0, 1]
    print(f"    低频段 (前 {n_low} 子载波) 幅度相关系数: {corr:.4f}")

    # ========== 5. 测试 3: CP 长度对 ISI 的影响 ==========
    print("\n  [测试 3] CP 长度对 ISI 的影响...")

    def measure_isi_err(cp_samples_local, link_obj=None):
        if link_obj is None:
            link_obj = CPOFDMLink(n_fft=args.n_fft, cp_len=cp_samples_local)
        tx = link_obj.transmit(X_tx)
        rx = channel.apply(tx)
        X_rx_local = link_obj.receive(rx, args.n_sym)
        # 用离散 FIR 的频响作参考
        err = np.mean(np.abs(X_rx_local[:, 1:-1] -
                              X_tx[:, 1:-1] * H_ref.reshape(-1, 1)) ** 2) \
              / np.mean(np.abs(X_tx) ** 2)
        return float(err)

    err_with_cp = measure_isi_err(cp_samples)
    err_no_cp = measure_isi_err(0)   # CP = 0

    print(f"    有 CP (n={cp_samples}):  err = {err_with_cp:.4e}")
    print(f"    无 CP (n=0):             err = {err_no_cp:.4e}")
    improvement = err_no_cp / max(err_with_cp, 1e-12)
    print(f"    CP 改善倍数: {improvement:.1f}×")

    # ========== 6. 可视化 ==========
    out_dir = ROOT / "outputs" / "task3"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_path = out_dir / f"cp_ofdm_link_mu{args.mu}.png"

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # 图 1: 频响对比 (仅低频段)
    ax = axes[0, 0]
    f_axis = np.arange(n_low) * scs_khz * 1e-3
    ax.plot(f_axis, 20*np.log10(np.abs(H_est_mean[:n_low]) + 1e-12),
            'b-', alpha=0.7, label='Estimated')
    ax.plot(f_axis, 20*np.log10(np.abs(H_ref[:n_low]) + 1e-12),
            'r--', alpha=0.9, label='Discrete FIR')
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("|H(f)| (dB)")
    ax.set_title(f"Channel Response (low-band, corr={corr:.3f})")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 图 2: 接收星座图
    ax = axes[0, 1]
    scale = np.abs(H_est_mean).mean()
    ax.scatter(X_rx.real.flatten() / scale,
               X_rx.imag.flatten() / scale,
               s=2, alpha=0.3, color='steelblue')
    ax.set_xlabel("In-phase"); ax.set_ylabel("Quadrature")
    ax.set_title("Received Constellation")
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

    # 图 3: 均衡后星座图
    ax = axes[1, 0]
    X_eq = X_rx * np.conj(H_ref.reshape(-1, 1)) / \
           (np.abs(H_ref.reshape(-1, 1)) ** 2 + 1e-12)
    ax.scatter(X_eq.real.flatten(), X_eq.imag.flatten(),
               s=2, alpha=0.3, color='coral')
    ax.set_xlabel("In-phase"); ax.set_ylabel("Quadrature")
    ax.set_title("After Zero-Forcing EQ")
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

    # 图 4: 时域信号
    ax = axes[1, 1]
    t_axis = np.arange(link.sym_len) / fs_hz * 1e6
    ax.plot(t_axis, np.abs(tx_signal[:link.sym_len]), 'b-',
            alpha=0.7, label='TX')
    ax.plot(t_axis, np.abs(rx_clean[:link.sym_len]), 'r-',
            alpha=0.7, label='RX')
    ax.axvline(cp_samples / fs_hz * 1e6, color='k', linestyle='--',
               alpha=0.5, label='CP ends')
    ax.set_xlabel("Time (us)"); ax.set_ylabel("|Signal|")
    ax.set_title("Time Domain (1 OFDM symbol)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print(f"\n[output] {plot_path}")
    print(f"\n关键结论:")
    print(f"  1. CP 长度 ({cp_samples}) >= 信道时延 ({max_delay_samples}): {cp_ok}")
    print(f"  2. 有 CP 时误差 {err_with_cp:.2e} << 无 CP 时误差 {err_no_cp:.2e}")
    print(f"  3. CP 改善 ISI: {improvement:.0f}× (说明 CP 是 OFDM 抗多径的关键)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
