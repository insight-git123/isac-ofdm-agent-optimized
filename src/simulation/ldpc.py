"""LDPC 码：(3,6)-正则 LDPC + Min-Sum 译码。

简化版本，用于教学/研究演示。
"""
import numpy as np


def _build_regular_ldpc(n_var: int = 96, dv: int = 3, dc: int = 6,
                        seed: int = 42) -> np.ndarray:
    """构造 (dv, dc)-正则 LDPC 的校验矩阵 H。"""
    rng = np.random.default_rng(seed)
    n_chk = n_var * dv // dc
    assert n_chk * dc == n_var * dv, "码率必须匹配"

    H = np.zeros((n_chk, n_var), dtype=int)

    # 变量节点存根: 每个变量节点出现 dv 次
    var_stubs = list(np.repeat(np.arange(n_var), dv))
    rng.shuffle(var_stubs)

    for chk_idx in range(n_chk):
        chosen = []
        for v in var_stubs:
            if v not in chosen:
                chosen.append(v)
                if len(chosen) == dc:
                    break
        # 从 var_stubs 里移除已选的
        for v in chosen:
            var_stubs.remove(v)
        for v in chosen:
            H[chk_idx, v] = 1

    return H


def _build_generator_from_H(H: np.ndarray):
    """从 H 构造系统生成矩阵 G。

    高斯消元到 RREF → 主元列 P、自由列 F。
    G 的构造: G[i, F[i]]=1, G[i, P[k]]=H[k, F[i]]

    Returns:
        (G, free_cols): G 是 (K, N) 生成矩阵, free_cols 是信息比特位置
    """
    H = H.copy() % 2
    M, N = H.shape

    row = 0
    pivot_cols = []
    for col in range(N):
        if row >= M:
            break
        pivot = None
        for r in range(row, M):
            if H[r, col] == 1:
                pivot = r
                break
        if pivot is None:
            continue
        H[[row, pivot]] = H[[pivot, row]]
        for r in range(M):
            if r != row and H[r, col] == 1:
                H[r] = (H[r] + H[row]) % 2
        pivot_cols.append(col)
        row += 1

    pivot_set = set(pivot_cols)
    free_cols = [c for c in range(N) if c not in pivot_set]
    K = len(free_cols)

    G = np.zeros((K, N), dtype=int)
    for i, free in enumerate(free_cols):
        G[i, free] = 1
        for pivot_row, pivot_col in enumerate(pivot_cols):
            G[i, pivot_col] = H[pivot_row, free]

    return G, free_cols


class RegularLDPC:
    """(dv, dc)-正则 LDPC 码。"""
    def __init__(self, n_var: int = 96, dv: int = 3, dc: int = 6,
                 seed: int = 42):
        self.N = n_var
        self.M = n_var * dv // dc

        self.H = _build_regular_ldpc(n_var, dv, dc, seed)
        self.G, self.free_cols = _build_generator_from_H(self.H)
        self.K = self.G.shape[0]

        # 预计算边列表
        self.var_edges = [np.where(self.H[:, j])[0].tolist()
                          for j in range(self.N)]
        self.chk_edges = [np.where(self.H[i, :])[0].tolist()
                          for i in range(self.M)]

    def encode(self, info_bits: np.ndarray) -> np.ndarray:
        """系统编码: codeword = info @ G mod 2。"""
        info_bits = np.asarray(info_bits, dtype=int)
        if len(info_bits) < self.K:
            info_bits = np.concatenate([info_bits,
                                        np.zeros(self.K - len(info_bits), dtype=int)])
        info_bits = info_bits[:self.K]
        return (info_bits @ self.G) % 2

    def decode(self, llr: np.ndarray, n_iter: int = 30) -> np.ndarray:
        """标准 LLR-BP Min-Sum 译码。

        Returns:
            信息比特 (按 free_cols 顺序提取)
        """
        llr = np.asarray(llr, dtype=float)
        assert len(llr) == self.N

        # 初始化 V2C
        V2C = {}
        for j in range(self.N):
            for i in self.var_edges[j]:
                V2C[(i, j)] = llr[j]

        for _ in range(n_iter):
            # 校验节点更新 (Min-Sum)
            C2V = {}
            for i in range(self.M):
                edges = self.chk_edges[i]
                for j_target in edges:
                    sign_prod = 1
                    min_mag = np.inf
                    for j in edges:
                        if j == j_target:
                            continue
                        msg = V2C[(i, j)]
                        sign_prod *= 1 if msg >= 0 else -1
                        mag = abs(msg)
                        if mag < min_mag:
                            min_mag = mag
                    if min_mag == np.inf:
                        min_mag = 0
                    C2V[(i, j_target)] = sign_prod * min_mag * 0.8

            # 变量节点更新
            for j in range(self.N):
                total = llr[j] + sum(C2V[(i, j)] for i in self.var_edges[j])
                for i in self.var_edges[j]:
                    V2C[(i, j)] = total - C2V[(i, j)]

        # 硬判决 (整个码字)
        decoded = np.zeros(self.N, dtype=int)
        for j in range(self.N):
            total = llr[j] + sum(V2C[(i, j)] for i in self.var_edges[j])
            decoded[j] = 0 if total > 0 else 1

        # 按 free_cols 提取信息比特
        return decoded[self.free_cols]


# 向后兼容别名
QCLDPC = RegularLDPC