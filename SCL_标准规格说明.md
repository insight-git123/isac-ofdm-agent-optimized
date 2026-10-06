# ISAC-OFDM Agent 标准 SCL

## 1. 文档目的
本 SCL（Standard Specification and Compliance List，标准规格与符合性清单）定义 `isac-ofdm-agent` 的功能边界、输入输出、接口约束、可复现实验要求和验收标准。它是项目的独立规格基线，不修改原项目代码或目录结构。

## 2. 项目定位
- 类型：教学/研究型 OFDM-ISAC 仿真原型。
- 参考：3GPP TS 38.211 Rel-18 参数提取、3GPP TR 38.901 TDL 信道模型。
- 约束：不得宣称完整 3GPP 一致性；Polar/LDPC、预编码、层映射和 SSB/PDSCH 复用仍为简化或未实现部分。

## 3. 标准目录与模块边界
| 层级 | 目录/模块 | 责任 | 验收 |
|---|---|---|---|
| 数据摄取 | `src/ingestion` | PDF/文本解析、索引与参数提取 | 能生成带来源定位的参数文件 |
| 检索引用 | `src/retrieval` | 参数检索、引用和证据链 | 每个关键参数包含来源信息 |
| 波形仿真 | `src/simulation` | 调制、资源网格、SSB、CP-OFDM | 时域/频域等效误差在项目阈值内 |
| 感知处理 | `src/simulation` | 脉冲压缩、TDL、CFAR、MIMO/RDA | 输出距离/多普勒/角度及检测指标 |
| 扫描分析 | `src/sweep` | numerology/性能扫描和置信区间 | 结果可复现并含 CI |
| 任务编排 | `tasks` | Task1–Task5 CLI 入口 | 各任务可单独运行 |
| 验证 | `tests` | 单元与接口测试 | 测试全部通过 |
| 报告 | `src/report` | 从 JSON 动态生成报告 | 报告包含环境和版本信息 |

## 4. 输入输出契约
- 输入：`config/params.yaml`、`config/sources.yaml`、`data/raw/` 中的标准资料或脱敏样例。
- 输出：`outputs/task1` 至 `outputs/task5`，包括 JSON、PNG 和 Markdown 报告。
- 随机实验必须显式记录 seed、参数、Python/NumPy 版本和 Git commit（若可用）。
- 不得覆盖原始输入；生成文件只能写入 `outputs/` 或用户指定目录。

## 5. 标准运行流程
```text
Task1 参数摄取 → Task2 NR 波形/资源网格/编码 → Task3 感知链路
→ Task4 扫描与统计 → Task5 自动报告
```
推荐命令：`python -m pytest -q`；全流程使用仓库提供的 `run_all.sh` 或对应 `tasks/task*_run.py`。

## 6. 符合性等级
- L0：目录、配置和 CLI 可用。
- L1：单元测试通过，关键输出可生成。
- L2：实验可复现，引用和环境元数据完整。
- L3：与真实 3GPP 文档逐项核验；当前项目不默认声称达到 L3。

## 7. 已知限制与变更控制
所有偏离项登记在 `LIMITATIONS.md`。标准变更必须新增版本号、变更原因、受影响任务和回归测试；禁止直接改写本 SCL 作为临时记录。

## 8. 验收清单
- [ ] 原项目目录与代码未被修改
- [ ] Task1–Task5 入口可发现
- [ ] 关键输出可生成且路径稳定
- [ ] 测试通过并记录运行环境
- [ ] 结果与限制说明一致
