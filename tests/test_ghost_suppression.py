"""鬼影抑制单元测试 + Precision/Recall 评估 (P4.4)。"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.simulation.ghost_suppression import suppress_ghosts


def _compute_metrics(true_idx, ghost_idx, det_ranges, det_velocities, targets,
                     r_tol=5.0, v_tol=15.0):
    """计算 precision/recall (用于测试)。"""
    def is_true(r, v):
        return any(abs(r - t[0]) <= r_tol and abs(v - t[1]) <= v_tol
                   for t in targets)

    TP = FP = FN = TN = 0
    for i in true_idx:
        if is_true(det_ranges[i], det_velocities[i]):
            TP += 1
        else:
            FP += 1
    for i in ghost_idx:
        if is_true(det_ranges[i], det_velocities[i]):
            FN += 1
        else:
            TN += 1

    precision = TP / max(TP + FP, 1)
    recall = TP / max(TP + FN, 1)
    return {"TP": TP, "FP": FP, "FN": FN, "TN": TN,
            "precision": precision, "recall": recall}


# ============ 基础功能测试 (原有) ============

def test_ghost_suppression_basic():
    """3 真实 + 2 鬼影场景, 应正确分类。"""
    det_ranges = [150.0, 300.0, 450.0, 172.5, 322.5]
    det_velocities = [30.0, -20.0, 0.0, 20.0, -30.0]
    det_mags = [1.0, 0.7, 0.9, 0.2, 0.15]

    true_idx, ghost_idx = suppress_ghosts(
        det_ranges, det_velocities, det_mags,
        r_tol=3.0, v_tol=5.0, mag_ratio=0.6, verbose=False
    )

    assert len(ghost_idx) >= 1
    assert 0 in true_idx or 2 in true_idx


def test_ghost_suppression_single_point():
    """单点时不应报错。"""
    true_idx, ghost_idx = suppress_ghosts(
        [150.0], [30.0], [1.0], verbose=False
    )
    assert true_idx == [0]
    assert ghost_idx == []


# ============ P4.4: Precision/Recall 评估 ============

def test_suppression_metrics_perfect_case():
    """理想场景: 3 真 2 鬼, 完美区分。"""
    det_ranges = [150.0, 300.0, 450.0, 172.5, 322.5]
    det_velocities = [30.0, -20.0, 0.0, 20.0, -30.0]
    det_mags = [1.0, 0.7, 0.9, 0.2, 0.15]
    targets = [(150.0, 30.0), (300.0, -20.0), (450.0, 0.0)]

    true_idx, ghost_idx = suppress_ghosts(
        det_ranges, det_velocities, det_mags,
        r_tol=3.0, v_tol=5.0, mag_ratio=0.6, verbose=False
    )

    m = _compute_metrics(true_idx, ghost_idx, det_ranges, det_velocities, targets)
    # 理想场景: precision 和 recall 都应很高
    assert m["precision"] >= 0.9, f"precision={m['precision']}"
    assert m["recall"] >= 0.9, f"recall={m['recall']}"
    assert m["FN"] == 0, f"不应误抑制真实目标, 但 FN={m['FN']}"


def test_suppression_metrics_no_ghosts():
    """无鬼影场景: precision 和 recall 都应为 1。"""
    det_ranges = [150.0, 300.0, 450.0]
    det_velocities = [30.0, -20.0, 0.0]
    det_mags = [1.0, 0.7, 0.9]
    targets = [(150.0, 30.0), (300.0, -20.0), (450.0, 0.0)]

    true_idx, ghost_idx = suppress_ghosts(
        det_ranges, det_velocities, det_mags, verbose=False
    )

    m = _compute_metrics(true_idx, ghost_idx, det_ranges, det_velocities, targets)
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0


def test_suppression_metrics_with_false_positive():
    """含误保留场景: precision < 1。"""
    # 5 个检测点, 只有 3 个是真值, 其余是噪声
    det_ranges = [150.0, 300.0, 450.0, 200.0, 500.0]
    det_velocities = [30.0, -20.0, 0.0, 10.0, 5.0]
    det_mags = [1.0, 0.9, 0.8, 0.7, 0.6]   # 幅度都差不多, 无法通过幅度区分
    targets = [(150.0, 30.0), (300.0, -20.0), (450.0, 0.0)]

    true_idx, ghost_idx = suppress_ghosts(
        det_ranges, det_velocities, det_mags, verbose=False
    )

    m = _compute_metrics(true_idx, ghost_idx, det_ranges, det_velocities, targets)
    # 三个真值都在 → recall 应为 1
    assert m["recall"] == 1.0
    # 但保留了 2 个噪声点 → precision < 1
    assert m["precision"] < 1.0
    assert m["FP"] >= 1


def test_suppression_metrics_with_false_negative():
    """含误抑制场景: recall < 1。"""
    # 目标 1 的幅度被错误设成最弱 (真实场景可能是深衰落)
    det_ranges = [150.0, 300.0, 450.0, 172.5, 322.5]
    det_velocities = [30.0, -20.0, 0.0, 20.0, -30.0]
    det_mags = [0.05, 0.7, 0.9, 0.2, 0.15]   # 目标 1 幅度异常低
    targets = [(150.0, 30.0), (300.0, -20.0), (450.0, 0.0)]

    true_idx, ghost_idx = suppress_ghosts(
        det_ranges, det_velocities, det_mags,
        r_tol=3.0, v_tol=5.0, mag_ratio=0.6, verbose=False
    )

    m = _compute_metrics(true_idx, ghost_idx, det_ranges, det_velocities, targets)
    # 目标 1 可能被误抑制 → recall < 1 或 FP > 0
    # 不断言具体数值, 只确认指标能正常计算
    assert "precision" in m
    assert "recall" in m
    assert m["TP"] + m["FP"] + m["FN"] + m["TN"] == len(det_ranges)


def test_metrics_sum_conservation():
    """混淆矩阵元素之和应等于检测点总数。"""
    det_ranges = [150.0, 300.0, 450.0, 172.5, 322.5, 500.0]
    det_velocities = [30.0, -20.0, 0.0, 20.0, -30.0, 5.0]
    det_mags = [1.0, 0.7, 0.9, 0.2, 0.15, 0.1]
    targets = [(150.0, 30.0), (300.0, -20.0), (450.0, 0.0)]

    true_idx, ghost_idx = suppress_ghosts(
        det_ranges, det_velocities, det_mags, verbose=False
    )

    m = _compute_metrics(true_idx, ghost_idx, det_ranges, det_velocities, targets)
    assert m["TP"] + m["FP"] + m["FN"] + m["TN"] == 6