"""标准 2D CA-CFAR（功率域实现，纯 numpy）。

修复:
1. 明确接收功率矩阵（不是幅度矩阵）
2. 正确计算训练单元数量
3. 使用标准 CA-CFAR 阈值公式 alpha = N_train * (pfa^(-1/N_train) - 1)
4. 用镜像 padding 处理边界
5. 不含 scipy 依赖，纯 numpy 实现
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def ca_cfar_2d(rdm_power: np.ndarray,
               guard_cells: int = 2,
               train_cells: int = 4,
               pfa: float = 1e-3) -> np.ndarray:
    """2D CA-CFAR，功率域。

    Args:
        rdm_power: 功率矩阵 (|rdm|^2)，2D
        guard_cells: 保护单元数（每侧）
        train_cells: 训练单元数（每侧）
        pfa: 设计虚警率

    Returns:
        detected: 布尔矩阵，True 表示检测到
    """
    rdm_power = np.asarray(rdm_power, dtype=float)
    rows, cols = rdm_power.shape

    # 训练单元数：完整矩形环 - 中心 CUT + 保护环
    N_total = (2 * train_cells + 2 * guard_cells + 1) ** 2
    N_excluded = (2 * guard_cells + 1) ** 2
    N_train = N_total - N_excluded

    if N_train <= 0:
        raise ValueError("训练单元数必须 > 0")

    # 标准 CA-CFAR 阈值因子
    alpha = N_train * (pfa ** (-1.0 / N_train) - 1.0)

    # 边界镜像 padding
    pad = train_cells + guard_cells
    padded = np.pad(rdm_power, pad, mode='reflect')

    # 完整窗口大小
    window_size = 2 * train_cells + 2 * guard_cells + 1
    exclude_size = 2 * guard_cells + 1

    # 用 sliding_window_view 提取所有窗口（不复制数据）
    full_windows = sliding_window_view(padded, (window_size, window_size))
    # full_windows shape: (rows, cols, window_size, window_size)

    # 每个窗口的"外部环"求和 = 窗口总和 - 中心排除区域总和
    # 中心排除区域是整个窗口的中间 exclude_size x exclude_size 部分
    # 在 window_size x window_size 坐标系里
    g0 = train_cells
    g1 = train_cells + exclude_size
    center_sum = full_windows[:, :, g0:g1, g0:g1].sum(axis=(-2, -1))
    total_sum = full_windows.sum(axis=(-2, -1))
    train_sum = total_sum - center_sum

    noise_est = train_sum / N_train
    threshold = noise_est * alpha
    detected = rdm_power > threshold

    return detected


def ca_cfar_2d_from_magnitude(rdm_mag: np.ndarray,
                              guard_cells: int = 2,
                              train_cells: int = 4,
                              pfa: float = 1e-3) -> np.ndarray:
    """从幅度矩阵出发，转为功率后调用 ca_cfar_2d。"""
    return ca_cfar_2d(np.abs(rdm_mag) ** 2,
                      guard_cells=guard_cells,
                      train_cells=train_cells,
                      pfa=pfa)