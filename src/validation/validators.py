"""验证任务1提取结果是否符合 3GPP Rel-18。"""
from typing import Dict, List

# 依据 TS 38.211 v18.4.0 Table 4.2-1
# 依据 TS 38.211 v18.4.0 Table 4.2-1
EXPECTED_NUMEROLOGY = {
    0: {"scs_khz": 15, "cp_types": ["normal"]},
    1: {"scs_khz": 30, "cp_types": ["normal"]},
    2: {"scs_khz": 60, "cp_types": ["normal", "extended"]},
    3: {"scs_khz": 120, "cp_types": ["normal"]},
    4: {"scs_khz": 240, "cp_types": ["normal"]},   # Rel-18 新增
    5: {"scs_khz": 480, "cp_types": ["normal"]},   # Rel-18 新增
    6: {"scs_khz": 960, "cp_types": ["normal"]},   # Rel-18 新增
}

# T_CP = 144 * kappa * 2^(-mu) * T_c, kappa=64, T_c ≈ 0.509 ns
KAPPA = 64
T_C_SEC = 0.509e-9


def compute_normal_cp_us(mu: int) -> float:
    return 144 * KAPPA * (2 ** (-mu)) * T_C_SEC * 1e6


def validate_numerology(extracted: List[Dict]) -> Dict:
    errors, warnings = [], []
    seen_mu = set()
    for entry in extracted:
        mu = entry.get("mu")
        if mu not in EXPECTED_NUMEROLOGY:
            errors.append(f"未识别的 mu={mu}")
            continue
        if mu in seen_mu:
            warnings.append(f"mu={mu} 重复出现")
        seen_mu.add(mu)
        exp = EXPECTED_NUMEROLOGY[mu]
        if entry.get("scs_khz") != exp["scs_khz"]:
            errors.append(
                f"mu={mu}: SCS={entry.get('scs_khz')} 与期望 {exp['scs_khz']} 不符"
            )
        got_cp = set(entry.get("cp_types", []))
        exp_cp = set(exp["cp_types"])
        if got_cp != exp_cp:
            errors.append(
                f"mu={mu}: CP={sorted(got_cp)} 与期望 {sorted(exp_cp)} 不符"
            )
    missing = set(EXPECTED_NUMEROLOGY) - seen_mu
    if missing:
        errors.append(f"缺失 mu: {sorted(missing)}")
    return {
        "passed": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "expected": EXPECTED_NUMEROLOGY,
    }


def validate_cp_duration(extracted: List[Dict], tol: float = 0.02) -> Dict:
    """验证正常CP时长计算，相对误差 ≤ tol (默认2%)。"""
    errors = []
    for entry in extracted:
        mu = entry.get("mu")
        if mu is None or "normal" not in entry.get("cp_types", []):
            continue
        expected = compute_normal_cp_us(mu)
        got = entry.get("cp_duration_us")
        if got is None:
            errors.append(f"mu={mu}: 未提供 cp_duration_us")
            continue
        rel = abs(got - expected) / expected
        if rel > tol:
            errors.append(
                f"mu={mu}: cp_duration={got:.4f} µs, 期望={expected:.4f} µs, 相对误差={rel:.2%}"
            )
    return {"passed": len(errors) == 0, "errors": errors}