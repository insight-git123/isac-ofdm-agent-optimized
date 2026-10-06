"""MIMO 阵列单元测试。"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.simulation.mimo_array import (
    UniformLinearArray, UniformPlanarArray, generate_multipath_angles
)


def test_ula_steering_vector_shape():
    ula = UniformLinearArray(n_elements=8, fc_ghz=3.5)
    a = ula.steering_vector(0.0)
    assert a.shape == (8, 1)


def test_ula_broadside():
    """0 度入射时转向矢量全为 1。"""
    ula = UniformLinearArray(n_elements=8, fc_ghz=3.5)
    a = ula.steering_vector(0.0).flatten()
    assert np.allclose(a, np.ones(8))


def test_ula_endfire():
    """90 度入射时相位差最大。"""
    ula = UniformLinearArray(n_elements=8, fc_ghz=3.5)
    a = ula.steering_vector(np.pi / 2).flatten()
    # 相邻相位差 = 2*pi*d/lambda = pi
    phase_diff = np.angle(a[1] / a[0])
    assert abs(abs(phase_diff) - np.pi) < 0.01


def test_ula_angle_axis_range():
    ula = UniformLinearArray(n_elements=8, fc_ghz=3.5)
    axis = ula.angle_axis(n_fft=64)
    assert len(axis) == 64
    assert np.all(axis >= -90)
    assert np.all(axis <= 90)


def test_upa_steering_vector_shape():
    upa = UniformPlanarArray(n_h=4, n_v=4, fc_ghz=3.5)
    a = upa.steering_vector(0.0, 0.0)
    assert a.shape == (16, 1)


def test_upa_broadside():
    """垂直入射时转向矢量全为 1。"""
    upa = UniformPlanarArray(n_h=4, n_v=4, fc_ghz=3.5)
    a = upa.steering_vector(0.0, 0.0).flatten()
    assert np.allclose(a, np.ones(16))


def test_multipath_angles_structure():
    targets = [{"range": 150, "velocity": 30, "az_deg": 0}]
    paths = generate_multipath_angles(1, targets, n_paths=3)
    assert len(paths) == 3
    # 第一条是 LOS
    assert paths[0]["path_idx"] == 0
    assert paths[0]["atten_db"] == 0.0
    assert paths[0]["az_deg"] == 0.0