"""CP-OFDM 收发链：真实时域处理 + 循环前缀。

参考 3GPP TS 38.211 Section 5.3 (OFDM baseband signal generation)。

发送链: freq symbols → IFFT → insert CP → time signal
接收链: time signal → remove CP → FFT → freq symbols
"""
import numpy as np


class CPOFDMLink:
    """CP-OFDM 收发链。

    Args:
        n_fft: IFFT 长度 (频域子载波数)
        cp_len: 循环前缀长度 (采样点数)
    """
    def __init__(self, n_fft: int = 1024, cp_len: int = 72):
        self.n_fft = n_fft
        self.cp_len = cp_len
        self.sym_len = n_fft + cp_len

    def transmit(self, freq_grid: np.ndarray) -> np.ndarray:
        """发送：频域 → 时域。

        Args:
            freq_grid: (n_fft, n_symbols) complex 频域数据

        Returns:
            time_signal: (n_symbols * sym_len,) 时域信号 (串行)
        """
        n_fft, n_sym = freq_grid.shape
        assert n_fft == self.n_fft, f"期望 n_fft={self.n_fft}, 收到 {n_fft}"

        # IFFT (每列 = 一个 OFDM 符号)
        time_data = np.fft.ifft(freq_grid, axis=0) * np.sqrt(self.n_fft)
        # (n_fft, n_sym)

        # 插入 CP: 复制最后 cp_len 个采样到前面
        cp = time_data[-self.cp_len:, :]
        time_with_cp = np.concatenate([cp, time_data], axis=0)
        # (sym_len, n_sym)

        # 串行化
        return time_with_cp.flatten(order='F')

    def receive(self, time_signal: np.ndarray, n_symbols: int) -> np.ndarray:
        """接收：时域 → 频域。

        Args:
            time_signal: (>= n_symbols * sym_len,) 时域信号
            n_symbols: 期望的符号数

        Returns:
            freq_grid: (n_fft, n_symbols) complex 频域数据
        """
        needed = n_symbols * self.sym_len
        if len(time_signal) < needed:
            time_signal = np.pad(time_signal, (0, needed - len(time_signal)),
                                 mode='constant')

        # 反串行化
        time_mat = time_signal[:needed].reshape(n_symbols, self.sym_len).T
        # (sym_len, n_symbols)

        # 去 CP
        time_no_cp = time_mat[self.cp_len:, :]
        # (n_fft, n_symbols)

        # FFT
        freq = np.fft.fft(time_no_cp, axis=0) / np.sqrt(self.n_fft)
        return freq

    def circular_conv_property(self):
        """返回一个说明: CP 使信道成为循环卷积。"""
        return (f"CP-OFDM 核心原理:\n"
                f"  若 CP长度 ({self.cp_len} 采样) >= 信道最大时延扩展,\n"
                f"  则时域信道 h 与 OFDM 符号的线性卷积\n"
                f"  等价于对频域数据的逐子载波乘法 Y[k] = H[k]·X[k]。")


class CPOFDMLinkOversampled:
    """带过采样的 CP-OFDM 链路 (支持部分子载波占用)。

    实际 NR 中, IFFT 长度 > 有效子载波数, 用于过采样和降低带外泄漏。
    """
    def __init__(self, n_fft: int = 2048, n_active_sc: int = 1024,
                 cp_len: int = 144):
        self.n_fft = n_fft
        self.n_active_sc = n_active_sc
        self.cp_len = cp_len
        self.sym_len = n_fft + cp_len

        # 有效子载波在 IFFT 网格里的位置 (居中)
        self.start_sc = (n_fft - n_active_sc) // 2

    def transmit(self, freq_grid: np.ndarray) -> np.ndarray:
        """发送：有效频域 → 时域。"""
        n_active, n_sym = freq_grid.shape
        assert n_active == self.n_active_sc

        # 零填充到 IFFT 长度
        full_grid = np.zeros((self.n_fft, n_sym), dtype=complex)
        full_grid[self.start_sc:self.start_sc + n_active, :] = freq_grid

        # IFFT
        time_data = np.fft.ifft(full_grid, axis=0) * np.sqrt(self.n_fft)

        # 插入 CP
        cp = time_data[-self.cp_len:, :]
        time_with_cp = np.concatenate([cp, time_data], axis=0)

        return time_with_cp.flatten(order='F')

    def receive(self, time_signal: np.ndarray, n_symbols: int) -> np.ndarray:
        """接收：时域 → 有效频域。"""
        needed = n_symbols * self.sym_len
        if len(time_signal) < needed:
            time_signal = np.pad(time_signal, (0, needed - len(time_signal)),
                                 mode='constant')

        time_mat = time_signal[:needed].reshape(n_symbols, self.sym_len).T
        time_no_cp = time_mat[self.cp_len:, :]
        freq = np.fft.fft(time_no_cp, axis=0) / np.sqrt(self.n_fft)

        return freq[self.start_sc:self.start_sc + self.n_active_sc, :]