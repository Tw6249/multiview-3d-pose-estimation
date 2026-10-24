# 验收报告索引

本目录按定型主线的 **4 个核心环节** 保留 4 份里程碑 JSON 验收报告（1 环节 1 份）。早期与中间过渡实验报告已在 2026-10-24 全面精简中移除；完整方法演进记录见 [docs/research_log.md](../docs/research_log.md)。

## 主线 4 阶段里程碑报告

| 主线环节 | 报告文件 | 核心内容与结论 |
|---|---|---|
| **1. 真实多视角公开数据基准** | [public-real-data-2026-10-18.json](public-real-data-2026-10-18.json) | MPI-INF-3DHP 开发/留出/冻结测试集（共 6,000 帧）数据导入、跨视角人物选择与三维重建基线验证 |
| **2. 场景地面与足部接触基线** | [foot-contact-baseline-2026-10-19.json](foot-contact-baseline-2026-10-19.json) | 四视角真人 900 帧、独立场景自然地面平面拟合与 5 帧因果窗口足部接触状态机（`contact / swing / unknown`）评测 |
| **3. 12 帧因果 IRLS 连续状态估计 (`A2`)** | [continuous-prediction-2026-10-20.json](continuous-prediction-2026-10-20.json) | 900 帧、12 种遮挡/互换故障下，12 帧因果 Huber IRLS + 逐级父子骨长投影将观测恢复时的三维位置阶跃 P95 降低 69% |
| **4. BRM-C2 连续重定向与 SONIC / TWIST2 闭环控制** | [continuous-reference-twist-vs-sonic-2026-10-21.json](continuous-reference-twist-vs-sonic-2026-10-21.json) | 奇异壁垒单调柔化重定向（`BRM-C2`）+ 50 Hz Ruckig $C^2$ 连续参考流消除源头翻转跳变与准入死锁；配合 `v3`（0.45 柔化 + 1.0 s 预对准）在 120 ms 延迟下实现五段真人动作 SONIC v1.1 100% 平衡存活与停稳（全身 RMSE `19.32°`、下肢 `13.37°`） |
