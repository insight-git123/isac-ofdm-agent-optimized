"""SS/PBCH block 生成 (3GPP TS 38.211 Section 7.4.2-7.4.3)。

SSB 结构:
- 时域: 4 个 OFDM 符号 (symbol 0~3)
- 频域: 240 个子载波 (20 RB)

映射 (Table 7.4.3.1-1):
- Symbol 0: PSS 在子载波 56~182, 其余置零
- Symbol 1: PBCH (0~47, 192~239), DMRS 每 4 个子载波
- Symbol 2: SSS 在 56~182, PBCH 同 Symbol 1
- Symbol 3: PBCH 同 Symbol 1

PSS: 127 长 m-sequence (TS 38.211 §7.4.2.2)
SSS: 127 长 m-sequence 组 (TS 38.211 §7.4.2.3)
"""
import numpy as np


# ============================================================
# PSS 序列 (TS 38.211 §7.4.2.2.1)
# ============================================================
def _generate_m_sequence(init: list, length: int,
                          taps: tuple) -> np.ndarray:
    """通用 m-sequence 生成器。

    Args:
        init: 初始寄存器值 (7 位)
        length: 输出长度
        taps: 反馈抽头 (i+a) 和 (i+b) 的偏移，例如 (4, 0) 表示 x(i+4)+x(i)
    """
    x = np.zeros(length + 7, dtype=int)
    x[:7] = init
    a, b = taps
    for i in range(length):
        x[i + 7] = (x[i + a] + x[i + b]) % 2
    return x[:length]


def generate_pss_sequence(n_id_2: int) -> np.ndarray:
    """生成 PSS 序列 (127 长复数)。

    Args:
        n_id_2: 物理层小区 ID 组内编号 (0, 1, 2)

    Returns:
        d_pss: (127,) complex 序列，取值 ±1
    """
    assert n_id_2 in (0, 1, 2), "n_id_2 must be 0, 1, or 2"

    # x(i+7) = (x(i+4) + x(i)) mod 2, 初始 [0 0 0 0 0 0 1]
    x = _generate_m_sequence([0, 0, 0, 0, 0, 0, 1], 127, taps=(4, 0))

    n = np.arange(127)
    m = (n + 43 * n_id_2) % 127
    d_pss = 1 - 2 * x[m]
    return d_pss.astype(complex)


# ============================================================
# SSS 序列 (TS 38.211 §7.4.2.3.1)
# ============================================================
def generate_sss_sequence(n_id_1: int, n_id_2: int) -> np.ndarray:
    """生成 SSS 序列 (127 长复数)。

    Args:
        n_id_1: 物理层小区 ID 组编号 (0~335)
        n_id_2: 组内编号 (0, 1, 2)

    Returns:
        d_sss: (127,) complex 序列，取值 ±1
    """
    assert 0 <= n_id_1 <= 335
    assert n_id_2 in (0, 1, 2)

    # x0(i+7) = (x0(i+4) + x0(i)) mod 2, 初始 [0 0 0 0 0 0 1]
    x0 = _generate_m_sequence([0, 0, 0, 0, 0, 0, 1], 127, taps=(4, 0))
    # x1(i+7) = (x1(i+1) + x1(i)) mod 2, 初始 [0 0 0 0 0 0 1]
    x1 = _generate_m_sequence([0, 0, 0, 0, 0, 0, 1], 127, taps=(1, 0))

    m0 = 15 * (n_id_1 // 112) + 5 * n_id_2
    m1 = n_id_1 % 112

    n = np.arange(127)
    d_sss = (1 - 2 * x0[(n + m0) % 127]) * (1 - 2 * x1[(n + m1) % 127])
    return d_sss.astype(complex)


# ============================================================
# PBCH DMRS (TS 38.211 §7.4.1.4) - Gold 序列简化版
# ============================================================
def _generate_gold_sequence(c_init: int, length: int) -> np.ndarray:
    """Gold 序列 (TS 38.211 §5.2.1)。

    c(n) = (x1(n+Nc) + x2(n+Nc)) mod 2
    Nc = 1600
    """
    Nc = 1600
    total = length + Nc + 31

    x1 = np.zeros(total, dtype=int)
    x2 = np.zeros(total, dtype=int)
    x1[0] = 1
    # x2 初始化为 c_init 的二进制
    for i in range(31):
        x2[i] = (c_init >> i) & 1

    for i in range(total - 31):
        x1[i + 31] = (x1[i + 3] + x1[i]) % 2
        x2[i + 31] = (x2[i + 3] + x2[i + 2] + x2[i + 1] + x2[i]) % 2

    c = (x1[Nc:Nc + length] + x2[Nc:Nc + length]) % 2
    return c


def generate_pbch_dmrs(n_id: int, n_hf: int, i_ssb: int,
                       n_symbol: int) -> np.ndarray:
    """生成 PBCH DMRS 序列 (144 长复数)。

    Args:
        n_id: 物理层小区 ID (0~1007)
        n_hf: 半帧编号 (0, 1)
        i_ssb: SSB 索引 (0~7)
        n_symbol: OFDM 符号编号 (1 或 3)

    Returns:
        r: (144,) complex 序列
    """
    c_init = 2**11 * (i_ssb + 1) * ((n_id >> 2) + 1) + 2**6 * (i_ssb + 1) + (n_id % 4)

    c = _generate_gold_sequence(c_init, 288)
    # r(m) = (1/sqrt(2))*(1-2c(2m)) + j*(1/sqrt(2))*(1-2c(2m+1))
    r = (1 / np.sqrt(2)) * (1 - 2 * c[0::2]) + \
        1j * (1 / np.sqrt(2)) * (1 - 2 * c[1::2])
    return r.astype(complex)


# ============================================================
# SS/PBCH block 资源网格
# ============================================================
class SSBGrid:
    """SS/PBCH block 资源网格 (240 子载波 × 4 OFDM 符号)。

    Reference: TS 38.211 Table 7.4.3.1-1
    """
    N_SC = 240
    N_SYMBOLS = 4

    def __init__(self, n_id: int, n_hf: int = 0, i_ssb: int = 0):
        self.n_id = n_id
        self.n_id_1 = n_id // 3
        self.n_id_2 = n_id % 3
        self.n_hf = n_hf
        self.i_ssb = i_ssb

        self.grid = np.zeros((self.N_SC, self.N_SYMBOLS), dtype=complex)
        self.mask = np.full((self.N_SC, self.N_SYMBOLS), "zero", dtype=object)

    def map_pss(self):
        """映射 PSS 到 symbol 0, 子载波 56~182。"""
        pss = generate_pss_sequence(self.n_id_2)
        self.grid[56:183, 0] = pss
        self.mask[56:183, 0] = "pss"

    def map_sss(self):
        """映射 SSS 到 symbol 2, 子载波 56~182。"""
        sss = generate_sss_sequence(self.n_id_1, self.n_id_2)
        self.grid[56:183, 2] = sss
        self.mask[56:183, 2] = "sss"

    def map_pbch_and_dmrs(self):
        """映射 PBCH 与 DMRS。

        PBCH 覆盖:
        - Symbol 1: 0~47, 192~239
        - Symbol 2: 0~47, 192~239
        - Symbol 3: 0~239 (含 DMRS 位置)
        
        DMRS: symbol 1, 3 每 4 个子载波一个 (offset 由 n_id 决定)
        """
        # PBCH 数据填充 (简化: 随机 QPSK)
        pbch_ranges = {
            1: [(0, 48), (192, 240)],
            2: [(0, 48), (192, 240)],
            3: [(0, 240)],
        }
        for sym, ranges in pbch_ranges.items():
            for (start, end) in ranges:
                self.mask[start:end, sym] = "pbch"
                n = end - start
                rng = np.random.default_rng(self.n_id + sym)
                real = (1 - 2 * rng.integers(0, 2, n)) / np.sqrt(2)
                imag = (1 - 2 * rng.integers(0, 2, n)) / np.sqrt(2)
                self.grid[start:end, sym] = real + 1j * imag

        # PBCH DMRS: symbol 1, 3，每 4 个 RE
        # 参考 TS 38.211 Table 7.4.3.1-1
        dmrs_offset = self.n_id % 4  # v = N_ID^(cell) mod 4

        for sym in [1, 3]:
            # DMRS 子载波位置: offset + 4k 和 offset + 4k + 4 (sublabel 决定)
            # 简化: 从 offset 开始每 4 个取一个
            dmrs_sc = np.arange(dmrs_offset, 240, 4)
            dmrs_seq = generate_pbch_dmrs(self.n_id, self.n_hf,
                                          self.i_ssb, sym)
            # 截断或补齐
            n_dmrs = min(len(dmrs_sc), len(dmrs_seq))
            self.grid[dmrs_sc[:n_dmrs], sym] = dmrs_seq[:n_dmrs]
            self.mask[dmrs_sc[:n_dmrs], sym] = "dmrs"

    def build(self):
        """完整构建 SSB。"""
        self.map_pss()
        self.map_sss()
        self.map_pbch_and_dmrs()
        return self


def visualize_ssb(ssb: SSBGrid, save_path=None):
    """可视化 SSB 资源网格。"""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    color_map = {"zero": 0, "pss": 1, "sss": 2, "pbch": 3, "dmrs": 4}
    numeric = np.zeros_like(ssb.mask, dtype=int)
    for tag, val in color_map.items():
        numeric[ssb.mask == tag] = val

    cmap = ListedColormap(["#FFFFFF", "#4878CF", "#6ACC64", "#D65F5F", "#F0C040"])

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.imshow(numeric.T, aspect='auto', cmap=cmap, origin='lower',
              interpolation='nearest')
    ax.set_xlabel("Subcarrier index (240)")
    ax.set_ylabel("OFDM symbol")
    ax.set_title(f"SS/PBCH Block (N_ID={ssb.n_id}, i_SSB={ssb.i_ssb})")
    ax.set_yticks([0, 1, 2, 3])

    legend_elements = [
        Patch(facecolor="#FFFFFF", edgecolor='gray', label='Zero'),
        Patch(facecolor="#4878CF", label='PSS (symbol 0, SC 56-182)'),
        Patch(facecolor="#6ACC64", label='SSS (symbol 2, SC 56-182)'),
        Patch(facecolor="#D65F5F", label='PBCH'),
        Patch(facecolor="#F0C040", label='PBCH DMRS'),
    ]
    ax.legend(handles=legend_elements, loc='upper left',
              bbox_to_anchor=(1.01, 1.0), fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def occupancy_stats(ssb: SSBGrid) -> dict:
    """统计各资源类型占用。"""
    stats = {}
    for tag in ["pss", "sss", "pbch", "dmrs", "zero"]:
        stats[tag] = int((ssb.mask == tag).sum())
    stats["total"] = ssb.mask.size
    return stats