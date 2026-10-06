"""统计工具：Bootstrap + Wilson 置信区间。

小样本场景下，正态近似的 95% CI 会低估实际范围。
Bootstrap 通过重采样更准确地估计统计量的分布。

Reference:
- Efron, B. (1979). "Bootstrap Methods: Another Look at the Jackknife"
- Wilson, E.B. (1927). "Probable inference, the law of succession..."
"""
import numpy as np


def bootstrap_ci(values, n_bootstrap: int = 2000,
                 confidence: float = 0.95, seed: int = 42) -> dict:
    """Bootstrap 置信区间 (非参数, 适用任意统计量)。

    Args:
        values: 样本列表 (例如每次 MC 的命中数)
        n_bootstrap: 重采样次数
        confidence: 置信水平 (默认 0.95)
        seed: 随机种子 (可复现)

    Returns:
        dict: {"mean", "ci_lo", "ci_hi", "ci_half_width", "n_samples"}
    """
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    if n == 0:
        return {"mean": 0.0, "ci_lo": 0.0, "ci_hi": 0.0,
                "ci_half_width": 0.0, "n_samples": 0}

    rng = np.random.default_rng(seed)
    boot_means = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(arr, size=n, replace=True)
        boot_means[i] = np.mean(sample)

    alpha = (1 - confidence) / 2
    lo = float(np.percentile(boot_means, alpha * 100))
    hi = float(np.percentile(boot_means, (1 - alpha) * 100))
    mean = float(np.mean(arr))
    return {
        "mean": mean,
        "ci_lo": lo,
        "ci_hi": hi,
        "ci_half_width": (hi - lo) / 2,
        "n_samples": n,
    }


def wilson_ci(successes: int, n: int, confidence: float = 0.95) -> dict:
    """Wilson score interval (适用于二项分布比例)。

    比正态近似更准确, 特别是小样本和极端比例。

    Args:
        successes: 成功次数 (0 <= k <= n)
        n: 总试验次数
        confidence: 置信水平

    Returns:
        dict: {"p_hat", "ci_lo", "ci_hi", "ci_half_width"}
    """
    if n <= 0:
        return {"p_hat": 0.0, "ci_lo": 0.0, "ci_hi": 0.0, "ci_half_width": 0.0}

    from scipy.stats import norm
    z = norm.ppf(1 - (1 - confidence) / 2)

    p_hat = successes / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    margin = z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / denom

    lo = max(0.0, center - margin)
    hi = min(1.0, center + margin)
    return {
        "p_hat": p_hat,
        "ci_lo": lo,
        "ci_hi": hi,
        "ci_half_width": (hi - lo) / 2,
    }


def wilson_ci_no_scipy(successes: int, n: int,
                       confidence: float = 0.95) -> dict:
    """Wilson 区间 (纯 numpy 实现, 不依赖 scipy)。

    使用正态分布近似 z 值: z ≈ 1.96 (95%), 2.576 (99%), 1.645 (90%)。
    """
    if n <= 0:
        return {"p_hat": 0.0, "ci_lo": 0.0, "ci_hi": 0.0, "ci_half_width": 0.0}

    z_table = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
    z = z_table.get(confidence, 1.96)

    p_hat = successes / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    margin = z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / denom

    lo = max(0.0, center - margin)
    hi = min(1.0, center + margin)
    # 浮点精度: 当 p_hat 恰好为 0 或 1 时, 强制边界精确
    if p_hat <= 1e-12:
        lo = 0.0
    if p_hat >= 1.0 - 1e-12:
        hi = 1.0
    return {
        "p_hat": p_hat,
        "ci_lo": lo,
        "ci_hi": hi,
        "ci_half_width": (hi - lo) / 2,
    }