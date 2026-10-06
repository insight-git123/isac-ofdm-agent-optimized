"""Task4: 遍历所有 numerology，计算 CP 开销并对比。"""
import sys
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.simulation.ofdm_generator import OFDMGenerator

def run_sweep(params: dict, fft_size: int = 1024):
    results = []
    for num in params["numerology"]:
        gen = OFDMGenerator(
            scs_khz=num["scs_khz"],
            cp_duration_us=num["cp_duration_us"],
            fft_size=fft_size
        )
        # CP 开销 = CP样本数 / 总符号样本数
        overhead = gen.cp_samples / (gen.cp_samples + gen.fft_size)
        results.append({
            "mu": num["mu"],
            "scs_khz": num["scs_khz"],
            "cp_duration_us": num["cp_duration_us"],
            "cp_samples": gen.cp_samples,
            "cp_overhead": round(overhead, 4)
        })
    return results

def plot_results(results: list, out_path: Path):
    mus = [r["mu"] for r in results]
    overheads = [r["cp_overhead"] * 100 for r in results]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(mus, overheads, color='steelblue', width=0.5)
    plt.xlabel("Numerology (μ)")
    plt.ylabel("CP Overhead (%)")
    plt.title("CP Overhead vs Numerology (FFT=1024)")
    plt.xticks(mus)
    plt.grid(axis='y', alpha=0.3)

    # 在柱子上标注数值
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.1, f"{yval:.2f}%", ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()