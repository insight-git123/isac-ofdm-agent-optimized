"""NR 资源网格：DC 子载波、保护带、DMRS、PRS 映射。

参考 3GPP TS 38.211:
- Section 4.4: Physical resources (resource grid, DC, guard band)
- Section 7.4.1.1: PDSCH DMRS (Config Type 1)
- Section 7.4.1.7: PRS (Positioning Reference Signal)
"""
import numpy as np


class NRResourceGrid:
    """NR 资源网格 (频域 n_rb × 12 子载波，时域 n_symbols 个 OFDM 符号)。

    DC 子载波置零；保护带边缘置零。
    """
    def __init__(self, n_rb: int = 52, n_symbols: int = 14,
                 dc_null: bool = True, guard_rb: int = 2):
        self.n_rb = n_rb
        self.n_symbols = n_symbols
        self.n_sc = n_rb * 12
        self.dc_null = dc_null
        self.guard_rb = guard_rb

        # 网格: [n_sc, n_symbols]
        self.grid = np.zeros((self.n_sc, n_symbols), dtype=complex)
        # 每个 RE 的类型
        self.mask = np.full((self.n_sc, n_symbols), "data", dtype=object)

        # 1. DC 置零
        if dc_null:
            dc_idx = self.n_sc // 2
            self.mask[dc_idx, :] = "dc"
            self.grid[dc_idx, :] = 0

        # 2. 保护带
        if guard_rb > 0:
            guard_sc = guard_rb * 12
            self.mask[:guard_sc, :] = "guard"
            self.mask[-guard_sc:, :] = "guard"
            self.grid[:guard_sc, :] = 0
            self.grid[-guard_sc:, :] = 0

    def _qpsk(self):
        """随机 QPSK 符号 (归一化)。"""
        real = (1 - 2 * np.random.randint(0, 2)) / np.sqrt(2)
        imag = (1 - 2 * np.random.randint(0, 2)) / np.sqrt(2)
        return real + 1j * imag

    def map_dmrs(self, symbol_positions=None, dmrs_type: int = 1):
        """映射 PDSCH DMRS (Config Type 1 或 2)。

        TS 38.211 §7.4.1.1:
        - Type 1: 每 2 个子载波 1 个 DMRS
        - Type 2: 每 6 个子载波 1 个 DMRS
        """
        if symbol_positions is None:
            symbol_positions = [2]

        stride = 2 if dmrs_type == 1 else 6

        for sym in symbol_positions:
            if sym < 0 or sym >= self.n_symbols:
                continue
            for sc in range(0, self.n_sc, stride):
                if self.mask[sc, sym] in ("guard", "dc"):
                    continue
                self.mask[sc, sym] = "dmrs"
                self.grid[sc, sym] = self._qpsk()

    def map_prs(self, symbol_positions=None, comb_size: int = 4,
                re_offset: int = 0):
        """映射 PRS (定位参考信号)。

        TS 38.211 §7.4.1.7:
        - 时域: dl-PRS-NumSymbols ∈ {2,4,6,12}
        - 频域: comb 结构, comb_size ∈ {2,4,6,12}
        """
        if symbol_positions is None:
            symbol_positions = [5, 6, 7, 8]

        for sym in symbol_positions:
            if sym < 0 or sym >= self.n_symbols:
                continue
            for sc in range(re_offset, self.n_sc, comb_size):
                if self.mask[sc, sym] in ("guard", "dc"):
                    continue
                self.mask[sc, sym] = "prs"
                self.grid[sc, sym] = self._qpsk()

    def fill_data(self):
        """填充数据 RE（跳过 DC/guard/dmrs/prs）。"""
        data_mask = self.mask == "data"
        n_data = int(data_mask.sum())
        if n_data == 0:
            return
        real = (1 - 2 * np.random.randint(0, 2, n_data)) / np.sqrt(2)
        imag = (1 - 2 * np.random.randint(0, 2, n_data)) / np.sqrt(2)
        self.grid[data_mask] = real + 1j * imag

    def to_ofdm_symbols(self):
        """IFFT 变换到时域。返回 shape (n_sc, n_symbols)。"""
        return np.fft.ifft(self.grid, axis=0) * np.sqrt(self.n_sc)

    def occupancy_stats(self):
        """资源占用统计。"""
        stats = {}
        for tag in ["data", "dmrs", "prs", "dc", "guard"]:
            stats[tag] = int((self.mask == tag).sum())
        total = self.mask.size
        stats["total"] = total
        stats["data_ratio"] = round(stats["data"] / total, 4)
        return stats


def visualize_grid(grid_obj, save_path=None, title="NR Resource Grid"):
    """可视化资源网格。

    颜色: 蓝=data, 绿=DMRS, 红=PRS, 灰=guard, 黑=DC
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    color_map = {"data": 0, "dmrs": 1, "prs": 2, "guard": 3, "dc": 4}
    numeric = np.zeros_like(grid_obj.mask, dtype=int)
    for tag, val in color_map.items():
        numeric[grid_obj.mask == tag] = val

    cmap = ListedColormap(["#4878CF", "#6ACC64", "#D65F5F", "#B0B0B0", "#000000"])

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(numeric.T, aspect='auto', cmap=cmap, origin='lower')
    ax.set_xlabel("Subcarrier index")
    ax.set_ylabel("OFDM symbol index")
    ax.set_title(f"{title} ({grid_obj.n_rb} RB, {grid_obj.n_symbols} symbols)")

    legend_elements = [
        Patch(facecolor="#4878CF", label="Data"),
        Patch(facecolor="#6ACC64", label="DMRS"),
        Patch(facecolor="#D65F5F", label="PRS"),
        Patch(facecolor="#B0B0B0", label="Guard"),
        Patch(facecolor="#000000", label="DC"),
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.18, 1.0))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()