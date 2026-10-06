"""多径鬼影抑制：幅度约束 + 正向距离 + 收紧指纹范围。"""
import numpy as np
from collections import Counter


def suppress_ghosts(det_ranges, det_velocities, det_mags=None,
                    r_tol=3.0, v_tol=5.0, mag_ratio=0.6, verbose=True):
    """
    多径鬼影抑制 (稳健版)。

    核心物理约束:
    1. 正向多径: 鬼影距离 > 真实目标距离 (LOS 最先到)
    2. 幅度衰减: 鬼影幅度 < 真实目标幅度 × mag_ratio
    3. 指纹范围: Δr ∈ [2, 60]m, Δv ∈ [0, 40]m/s
    """
    n = len(det_ranges)
    if n < 2:
        return list(range(n)), []

    # 归一化幅度
    if det_mags is None:
        mags_norm = np.ones(n)
    else:
        mags_norm = np.array(det_mags) / (max(det_mags) + 1e-12)

    # 1. 收集候选指纹 (i=鬼影候选, j=真实候选)
    fp_counter = Counter()
    fp_pairs = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            dr = det_ranges[i] - det_ranges[j]
            dv = abs(det_velocities[i] - det_velocities[j])

            # 约束1: 距离必须为正 (鬼影更远)
            if dr <= 0:
                continue
            # 约束2: 幅度必须明显更弱
            if mags_norm[i] >= mags_norm[j] * mag_ratio:
                continue
            # 约束3: 指纹物理范围
            if not (2.0 <= dr <= 200.0):
                continue
            if dv > 40.0:
                continue

            key = (round(dr / r_tol), round(dv / v_tol))
            fp_counter[key] += 1
            fp_pairs.append((key, i, j))

    if not fp_counter:
        if verbose:
            print("  [Ghost] 未检测到满足约束的指纹，跳过抑制")
        return list(range(n)), []

    # 2. 取票数最高的指纹
    top_key, votes = fp_counter.most_common(1)[0]
    if votes < 2:
        if verbose:
            print(f"  [Ghost] 指纹票数不足 ({votes}), 跳过抑制")
        return list(range(n)), []

    dr_fp = top_key[0] * r_tol
    dv_fp = top_key[1] * v_tol
    if verbose:
        print(f"  [Ghost] 主指纹: Δr≈{dr_fp:.1f}m, Δv≈{dv_fp:.1f}m/s (票数={votes})")

    # 3. 标记鬼影
    ghost_mask = [False] * n
    for key, i, j in fp_pairs:
        if key == top_key:
            ghost_mask[i] = True

    true_idx = [i for i in range(n) if not ghost_mask[i]]
    ghost_idx = [i for i in range(n) if ghost_mask[i]]
    return true_idx, ghost_idx