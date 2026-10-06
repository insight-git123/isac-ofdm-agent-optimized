"""CFAR 检测器单元测试，含 Pfa 校准。"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.simulation.cfar import ca_cfar_2d, ca_cfar_2d_from_magnitude


def test_cfar_detects_strong_target():
    """人工构造单目标 RDM，CFAR 应能检测到。"""
    rdm_power = np.random.exponential(1.0, (50, 50))   # 指数分布噪声
    rdm_power[25, 25] = 1e6                             # 强目标
    detected = ca_cfar_2d(rdm_power, guard_cells=2, train_cells=4, pfa=1e-2)
    assert detected[25, 25]


def test_cfar_pfa_calibration():
    """纯噪声下的实测 Pfa 应接近设定值（Monte Carlo）。"""
    np.random.seed(42)
    pfa_set = 1e-2
    total_false = 0
    total_cells = 0
    for _ in range(5):
        # 指数分布噪声（功率域正确分布）
        noise = np.random.exponential(1.0, (200, 200))
        detected = ca_cfar_2d(noise, guard_cells=2, train_cells=4, pfa=pfa_set)
        total_false += detected.sum()
        total_cells += detected.size

    pfa_actual = total_false / total_cells
    # 允许 2 倍误差（Monte Carlo 采样有限）
    assert pfa_actual < 3 * pfa_set, \
        f"实测 Pfa = {pfa_actual:.4f}, 期望 ≈ {pfa_set}"
    # 也不应太低（说明检测器过严）
    assert pfa_actual > pfa_set / 10, \
        f"实测 Pfa = {pfa_actual:.4f} 过低，检测器过严"


def test_cfar_handles_boundary():
    """边界目标也应被检测到（不应只扫描内部区域）。"""
    rdm_power = np.random.exponential(1.0, (50, 50))
    rdm_power[0, 0] = 1e6      # 左上角边界目标
    rdm_power[49, 49] = 1e6    # 右下角边界目标
    detected = ca_cfar_2d(rdm_power, guard_cells=2, train_cells=4, pfa=1e-2)
    assert detected[0, 0], "左上角边界目标未检测"
    assert detected[49, 49], "右下角边界目标未检测"


def test_cfar_from_magnitude_wrapper():
    """从幅度矩阵出发的封装应等价于功率域。"""
    np.random.seed(0)
    rdm_mag = np.random.exponential(1.0, (50, 50))
    rdm_mag[25, 25] = 100.0
    d1 = ca_cfar_2d_from_magnitude(rdm_mag, guard_cells=2, train_cells=4, pfa=1e-2)
    d2 = ca_cfar_2d(np.abs(rdm_mag) ** 2, guard_cells=2, train_cells=4, pfa=1e-2)
    assert np.array_equal(d1, d2)


def test_cfar_does_not_detect_pure_noise_peak():
    """纯噪声中不应有大量虚警。"""
    np.random.seed(0)
    noise = np.random.exponential(1.0, (100, 100))
    detected = ca_cfar_2d(noise, guard_cells=2, train_cells=4, pfa=1e-4)
    # 允许少量虚警（< 5%）
    assert detected.sum() < 100 * 100 * 0.05