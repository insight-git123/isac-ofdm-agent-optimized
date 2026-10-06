# 已知局限 (Known Limitations)

本文档诚实列出项目当前的**技术边界**。

> **修复摘要 (P0~P4)**：
> - ✅ **P0**：输入路径动态解析 + 诚实定位
> - ✅ **P1**：TDL-A/B/C/D/E 抽头表补全；CFAR 重写为功率域 + Pfa 校准
> - ✅ **P2**：QPSK/16QAM/64QAM/256QAM 调制；NR 资源网格；报告动态生成；95% CI
> - ✅ **P3**：MIMO ULA/UPA + 3D-FFT；(2,1,3) 卷积码；SS/PBCH block；CP-OFDM 全链路
> - ✅ **P4**：TDL FFT 频率轴修正；Bootstrap/Wilson CI；MIMO RMSE；统一 CLI；Polar + LDPC

## 1. 输入可复现性
- 仓库只保留 `data/raw/sample_ts38211.txt`（脱敏样例）
- 真实 PDF 需从 3gpp.org 下载，通过 `python tasks/task1_run.py --raw <path>` 运行
- **没有 SHA256 校验**

## 2. TDL 信道模型
- **抽头表完整**：TDL-A/B/C/D/E 各 23/23/24/13/14 抽头
- **Task3 统一调用** `tdl_frequency_response()`
- **P4.1 修正 FFT 频率轴**：从 0.42 → 0.97+ 相关系数
- **仍存在的简化**：
  - Rician K-factor 记录但未实现独立 LOS 相位控制
  - 多普勒谱用单频近似，未做 Jakes 建模

## 3. OFDM 信号模型
- **已实现**：
  - NR 资源网格：DC、Guard、DMRS、PRS
  - QPSK/16QAM/64QAM/256QAM
  - **完整 CP-OFDM 收发链** + 时域/频域等效性验证
  - **P4.8: Polar (SC) + LDPC (Min-Sum) 编码**，BER 曲线验证
- **仍存在的简化**：
  - 未实现 SSB 与 PDSCH 的时频复用
  - 未实现预编码、层映射、码字扰码
  - Polar/LDPC 为**简化实现**，非 NR 标准 BG1/BG2

## 4. CFAR 检测
- 功率域实现 + Pfa 校准 + 边界 padding
- **仍缺少**：OS-CFAR / GO-CFAR 对比

## 5. 鬼影抑制
- **P4.4 新增 Precision/Recall/F1 + 误抑制率**
- 规则仍依赖"鬼影更远更弱"等先验
- 测试数据有循环论证风险

## 6. 报告生成
- **P2.2 完全动态化** + **P4.3 嵌入 commit/Python/NumPy 版本**

## 7. 测试覆盖
- **128 个单元测试**（P4.8 后）
- 分布：CFAR(5)、Ghost(6)、LDPC(4)、MIMO(7)、MIMO-Metrics(4)、
  Modulation(7)、NRGrid(11)、Polar(6)、PulseComp(3)、Report(3)、
  Scenario(7)、SSB(14)、Statistics(10)、Task1(3)、TDL(12)、CP-OFDM(11)、CLI(6)
- **没有真实 PDF 端到端测试**（CI 用脱敏样例）

## 8. 项目定位

| 层级 | 当前状态 |
|---|---|
| 教学演示 | ✅ 具备 |
| 研究原型 | ✅ 具备 |
| 3GPP 一致性工具 | ❌ 不具备（Polar/LDPC 为简化版） |

## 9. 数值精度

- **TDL 频响一致性**：时域 FIR 与频域连续模型的相关系数 ≈ 0.97
- 剩余 3% 来自时延量化误差（8.14 ns 采样间隔）
- 修复前因频率轴错误相关系数仅 0.42

**定位：教学/研究型 OFDM-ISAC 仿真原型，不声称 3GPP 标准一致性。**