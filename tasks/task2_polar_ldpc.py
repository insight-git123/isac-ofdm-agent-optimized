"""Task2-Polar-LDPC: 三种编码方案 BER 曲线对比。"""
import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt

from src.simulation.polar import polar_encode, polar_decode
from src.simulation.ldpc import RegularLDPC
from src.simulation.scl_decoder import scl_decode_with_crc_selection


def _bpsk_awgn_llr(bits, snr_db, rng):
    """BPSK + AWGN, 返回 LLR。"""
    symbols = 1 - 2 * np.asarray(bits, dtype=float)
    sigma = 10 ** (-snr_db / 20)
    noise = rng.normal(0, sigma, size=symbols.shape)
    rx = symbols + noise
    return 2 * rx / sigma**2


def simulate_polar(K: int, N: int, snr_range: list,
                   n_bits_per_point: int = 5000, seed: int = 42,
                   use_scl: bool = False) -> dict:
    """Polar 码 BER 仿真。

    Args:
        use_scl: True = SCL(L=8)+CRC-11; False = SC (无 CRC)
    """
    from src.simulation.scl_decoder import scl_decode_with_crc_selection

    rng = np.random.default_rng(seed)
    bers = []
    for snr in snr_range:
        n_blocks = max(1, n_bits_per_point // K)
        total_err = 0
        total_bits = 0
        for _ in range(n_blocks):
            info = rng.integers(0, 2, size=K)
            if use_scl:
                codeword, frozen, _ = polar_encode(info, N, crc_type="CRC11")
                llr = _bpsk_awgn_llr(codeword, snr, rng)
                decoded = scl_decode_with_crc_selection(
                    llr, frozen, K, L=8, crc_type="CRC11"
                )
            else:
                codeword, frozen, _ = polar_encode(info, N, crc_type=None)
                llr = _bpsk_awgn_llr(codeword, snr, rng)
                decoded = polar_decode(llr, frozen, K, crc_type=None)

            if len(decoded) < K:
                total_err += K
            else:
                total_err += int(np.sum(info != decoded[:K]))
            total_bits += K
        bers.append(total_err / total_bits)

    return {"snr": snr_range, "ber": bers,
            "scheme": "Polar-SCL" if use_scl else "Polar-SC"}


def simulate_ldpc(snr_range: list, n_bits_per_point: int = 5000,
                  seed: int = 42) -> dict:
    """LDPC 码 BER 仿真 (大块数据)。"""
    rng = np.random.default_rng(seed)
    code = RegularLDPC(n_var=96, dv=3, dc=6)
    K_actual = code.K

    bers = []
    for snr in snr_range:
        n_blocks = max(1, n_bits_per_point // K_actual)
        total_err = 0
        total_bits = 0
        for _ in range(n_blocks):
            info = rng.integers(0, 2, size=K_actual)
            codeword = code.encode(info)
            llr = _bpsk_awgn_llr(codeword, snr, rng)
            decoded = code.decode(llr, n_iter=30)
            if len(decoded) != K_actual:
                total_err += K_actual
            else:
                total_err += int(np.sum(info != decoded))
            total_bits += K_actual
        bers.append(total_err / total_bits)
    return {"snr": snr_range, "ber": bers,
            "scheme": f"LDPC (N={code.N}, K={K_actual})"}


def simulate_uncoded(K: int, snr_range: list, n_bits_per_point: int = 5000,
                     seed: int = 42) -> dict:
    """未编码 BPSK 的 BER (大块数据)。"""
    rng = np.random.default_rng(seed)
    bers = []
    for snr in snr_range:
        info = rng.integers(0, 2, size=n_bits_per_point)
        llr = _bpsk_awgn_llr(info, snr, rng)
        decoded = (llr < 0).astype(int)
        bers.append(float(np.mean(info != decoded)))
    return {"snr": snr_range, "ber": bers, "scheme": "Uncoded"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-bits", type=int, default=5000,
                        help="每个 SNR 点的比特数")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scl", action="store_true",
                        help="使用 SCL (L=8) + CRC-11")
    args = parser.parse_args()

    print(f"[Task2-Polar-LDPC] 编码方案 BER 对比")
    print(f"  Polar: N=128, K=64, SC 译码")
    print(f"  LDPC:  N=96,  K=48, Min-Sum 译码")
    print(f"  每点: {args.n_bits} 比特\n")

    snr_range = [0, 1, 2, 3, 4, 5, 6, 7, 8]

    mode_str = "SCL L=8+CRC" if args.scl else "SC"
    print(f"  [1/3] Polar 码 ({mode_str})...")
    res_polar = simulate_polar(K=64, N=128, snr_range=snr_range,
                                n_bits_per_point=args.n_bits, seed=args.seed,
                                use_scl=args.scl)

    print("  [2/3] LDPC 码...")
    res_ldpc = simulate_ldpc(snr_range=snr_range,
                              n_bits_per_point=args.n_bits, seed=args.seed)

    print("  [3/3] 未编码...")
    res_uncoded = simulate_uncoded(K=64, snr_range=snr_range,
                                    n_bits_per_point=args.n_bits, seed=args.seed)

    print("\n" + "=" * 70)
    print(f"{'SNR(dB)':<10}{'Uncoded':<15}{'Polar':<15}{'LDPC':<15}")
    print("-" * 70)
    for i, snr in enumerate(snr_range):
        print(f"{snr:<10}{res_uncoded['ber'][i]:<15.4e}"
              f"{res_polar['ber'][i]:<15.4e}{res_ldpc['ber'][i]:<15.4e}")
    print("=" * 70)

    out_dir = ROOT / "outputs" / "task2"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_path = out_dir / "polar_ldpc_ber.png"

    plt.figure(figsize=(9, 6))
    plt.semilogy(res_uncoded["snr"], res_uncoded["ber"],
                 'o--', label="Uncoded BPSK", color='gray')
    plt.semilogy(res_polar["snr"], res_polar["ber"],
                 's-', label="Polar (N=128, K=64, SC)", color='steelblue')
    plt.semilogy(res_ldpc["snr"], res_ldpc["ber"],
                 '^-', label="LDPC (N=96, K=48, Min-Sum)", color='coral')
    plt.xlabel("SNR (dB)")
    plt.ylabel("Bit Error Rate (BER)")
    plt.title("BER Comparison: Polar vs LDPC vs Uncoded")
    plt.grid(True, which='both', alpha=0.3)
    plt.legend()
    plt.ylim(1e-4, 1)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print(f"\n[output] {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())