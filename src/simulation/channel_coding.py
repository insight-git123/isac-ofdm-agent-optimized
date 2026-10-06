"""(7,5) 卷积编码 + Viterbi 译码。

K=3, rate=1/2 经典卷积码：
- G1 = 111 (octal 7)
- G2 = 101 (octal 5)
适用于 AWGN 信道的误码率验证。

参考: Proakis & Salehi, "Digital Communications", Ch. 8.
"""
import numpy as np


class ConvolutionalEncoder:
    """(2, 1, 3) 卷积编码器。"""
    def __init__(self):
        # 生成多项式 (octal 7 = 111, octal 5 = 101)
        self.G1 = np.array([1, 1, 1])
        self.G2 = np.array([1, 0, 1])
        self.memory = 2  # K-1

    def encode(self, bits: np.ndarray) -> np.ndarray:
        """编码。输入 0/1 位流，输出双倍长度的 0/1 位流。"""
        bits = np.asarray(bits, dtype=int)
        n = len(bits)
        # 尾部补 2 个零（清零寄存器）
        bits_padded = np.concatenate([bits, np.zeros(self.memory, dtype=int)])

        out1 = np.zeros(n + self.memory, dtype=int)
        out2 = np.zeros(n + self.memory, dtype=int)

        reg = np.zeros(3, dtype=int)  # 3 位移位寄存器
        for i, b in enumerate(bits_padded):
            reg = np.roll(reg, 1)
            reg[0] = b
            out1[i] = np.dot(reg, self.G1) % 2
            out2[i] = np.dot(reg, self.G2) % 2

        # 交织: [out1[0], out2[0], out1[1], out2[1], ...]
        encoded = np.zeros(2 * (n + self.memory), dtype=int)
        encoded[0::2] = out1
        encoded[1::2] = out2
        return encoded


class ViterbiDecoder:
    """(2, 1, 3) 卷积码 Viterbi 译码器。"""
    def __init__(self):
        self.G1 = np.array([1, 1, 1])
        self.G2 = np.array([1, 0, 1])
        self.n_states = 4         # 2^(K-1) = 4
        self.memory = 2
        self._build_trellis()

    def _build_trellis(self):
        """预计算状态转移表。"""
        self.next_state = np.zeros((self.n_states, 2), dtype=int)
        self.output = np.zeros((self.n_states, 2, 2), dtype=int)

        for state in range(self.n_states):
            # state bits: [s1, s0] (s0 是最近输入)
            s1 = (state >> 1) & 1
            s0 = state & 1
            for b in [0, 1]:
                # 新寄存器: [b, s1, s0]
                reg = np.array([b, s1, s0])
                o1 = np.dot(reg, self.G1) % 2
                o2 = np.dot(reg, self.G2) % 2
                new_state = ((b << 1) | s1) & 0b11
                self.next_state[state, b] = new_state
                self.output[state, b, 0] = o1
                self.output[state, b, 1] = o2

    def decode(self, received: np.ndarray) -> np.ndarray:
        """Viterbi 译码。输入硬判决 0/1 位流，输出译码位流。"""
        received = np.asarray(received, dtype=int)
        n_steps = len(received) // 2

        # 度量初始化
        INF = 1e9
        metrics = np.full(self.n_states, INF)
        metrics[0] = 0  # 初始状态 0
        traceback = np.zeros((n_steps, self.n_states), dtype=int)

        for t in range(n_steps):
            r1, r2 = received[2 * t], received[2 * t + 1]
            new_metrics = np.full(self.n_states, INF)
            new_traceback = np.zeros(self.n_states, dtype=int)

            for state in range(self.n_states):
                if metrics[state] >= INF:
                    continue
                for b in [0, 1]:
                    ns = self.next_state[state, b]
                    o1 = self.output[state, b, 0]
                    o2 = self.output[state, b, 1]
                    # 汉明距离
                    bm = (r1 != o1) + (r2 != o2)
                    cum = metrics[state] + bm
                    if cum < new_metrics[ns]:
                        new_metrics[ns] = cum
                        new_traceback[ns] = state

            metrics = new_metrics
            traceback[t] = new_traceback

        # Traceback
        state = int(np.argmin(metrics))
        decoded = np.zeros(n_steps, dtype=int)
        for t in range(n_steps - 1, -1, -1):
            prev_state = traceback[t, state]
            decoded[t] = (state >> 1) & 1   # ← 取 bit 1（即当前输入 bit）
            state = prev_state

        # 去掉尾部补零
        return decoded[:-self.memory]


def awgn_channel(bits: np.ndarray, snr_db: float,
                 rng: np.random.Generator = None) -> np.ndarray:
    """BPSK + AWGN 信道。输入 0/1，输出硬判决 0/1。"""
    if rng is None:
        rng = np.random.default_rng()
    # BPSK 调制: 0 -> -1, 1 -> +1
    symbols = 1 - 2 * np.asarray(bits, dtype=float)
    # 噪声功率
    sig_power = 1.0
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = rng.normal(0, np.sqrt(noise_power), size=symbols.shape)
    received = symbols + noise
    # 硬判决
    return (received < 0).astype(int)


def measure_ber(n_bits: int = 10000, snr_db_range: list = None,
                use_coding: bool = True, seed: int = 42) -> dict:
    """测量不同 SNR 下的误码率。

    Returns:
        dict: {"snr_db": [...], "ber": [...]}
    """
    if snr_db_range is None:
        snr_db_range = [0, 1, 2, 3, 4, 5, 6, 7, 8]

    rng = np.random.default_rng(seed)
    encoder = ConvolutionalEncoder()
    decoder = ViterbiDecoder()

    snrs, bers = [], []
    for snr_db in snr_db_range:
        # 随机数据
        data = rng.integers(0, 2, size=n_bits)
        if use_coding:
            tx = encoder.encode(data)
        else:
            tx = data

        rx = awgn_channel(tx, snr_db, rng)

        if use_coding:
            decoded = decoder.decode(rx)
        else:
            decoded = rx

        # 计算 BER (截断到相同长度)
        min_len = min(len(data), len(decoded))
        ber = np.mean(data[:min_len] != decoded[:min_len])
        snrs.append(snr_db)
        bers.append(float(ber))

    return {"snr_db": snrs, "ber": bers}