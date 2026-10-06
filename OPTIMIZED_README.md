# isac-ofdm-agent 优化版

这是基于桌面原项目的独立优化副本。原项目 `Desktop\isac-ofdm-agent` 保持原样。

## 已加入的优化
1. `src/optimization.py`：确定性随机数、批量 Monte Carlo 扫描、轻量结果缓存和 NumPy 向量化统计工具。
2. `config/optimized_params.yaml`：面向快速迭代的默认配置，减少重复实验开销。
3. `OPTIMIZATION_CHANGELOG.md`：优化点、兼容性和验证记录。

优化模块以独立 API 提供，不强制改动现有 Task1–Task5 接口；可逐步接入已有任务。
