# PMPose 同人物框实验（2026-10-18）

目标：验证 ProbPose/BBoxMaskPose 的分布式观测是否适合现有多视角人体链路。当前实验只接入 PMPose-S 姿态头；未接入分割器和完整 BMP 迭代，不改默认 RTMPose-M、人体状态或控制接口。

## 实测结果与决定

4 人、2,500 帧、9,472 个有效相机裁剪。下表的平均/P95 为**同一共同有效点集合**、世界坐标绝对误差，没有对齐、缩放或移时。覆盖率与 ≤100 mm 比例以全部参考目标为分母。详见[机器可读验收](../reports/pmpose-observation-2026-10-18.json)。

| 指标，RTMPose-M → PMPose-S | 开发 S1/S2，1,000 帧 | 已查看过的复验 S4/S6，1,500 帧 |
|---|---:|---:|
| 平均位置误差 | 66.69 → **64.26 mm** | 49.51 → **46.96 mm** |
| P95 位置误差 | 130.93 → **124.87 mm** | 97.07 → **90.87 mm** |
| 全目标 ≤100 mm | 73.56% → **75.06%** | 94.99% → **95.86%** |
| 输出覆盖率 | 85.60% → **85.29%** | 99.47% → **99.42%** |
| >500 mm 输出点错误 | 31 → **11** | 0 → **0** |

位置有可重复的局部收益，但**未通过预先规定的覆盖率不下降条件，继续保留默认 M**。S2 的输出点平均误差基本持平；S2/S6 的躯干方向诊断均值变差，且各方法有效样本不同，尚未证明旋转整体改善。脚掌/手掌完整旋转、接触、脚滑仍无独立真值。

90% 最高密度区域的实际抽样覆盖：S1 **56.64%**、S2 **57.60%**、S4 **64.81%**、S6 **55.92%**；说明归一化位置分布在当前近似关节对应和图像域下并不可靠。裁剪内存在事件超过 99.5% 为正例，低 Brier 分数不能证明对出框/遮挡有辨别力；可见性准确率保持未知。概率图不能直接放行机器人动作。

固定相同四个裁剪、预热后交替测量 20 次：M 平均 **15.65 ms**；PMPose 逐视角 **64.24 ms**、批处理 **23.19 ms**，批处理 P95 **25.87 ms**。初次 4 帧批处理检查最大像素差仅 **0.000122 px**。早期连续测量曾达到 160–190 ms/组，随后在模型未变时明显下降，原因未确认；不能挑最快一段宣称稳定实时。连续阶段还有后台 CPU 评分，所有时间均不是物理相机端到端延迟。

回归：新接口 4 项；全仓视觉环境共 257 项，231 通过、26 条件跳过。默认环境、算法、控制门不变，无提交、推送或实机命令。只保留约 **4.06 MB** 当前数值证据/抽样概率图/日志；不复制视频和完整状态轨迹。

下一步先查清覆盖损失与关节定义偏差，在保留的开发块中校准概率；再把预测掩膜、分布级多视角融合分别做消融。需要新人物作为真正独立评测，不能继续把已查看的 S4/S6 当未见数据。

## 模块边界

- `pose_sim/probabilistic_pose.py`：当前 BGR `ExposureFrame`＋已选人物 `xyxy` 像素框 → COCO17 图像坐标、预测 OKS、裁剪内存在概率、可见性、64×48 联合位置概率质量。保留相机、曝光纳秒与时钟域；图像 x 向右、y 向下；不去畸变、不补历史帧。
- 23 个原始关节中的前 17 个必须与 COCO17 名称逐项一致。官方权重严格匹配全部参数，固定源码版本与校验和。
- 概率图由原始 Sparsemax 头生成，检查每个关节质量和为 1。归一化不等于在 MPI 上校准；存在概率表示关节位于模型裁剪内，不能解释为“人体存在”。预测 OKS 与可见性不互换。
- 官方 `PMPose.predict` 在字段缺失时退回普通关键点分数，且在 `pred_instances` 中查找实际位于 `pred_fields` 的热图。适配器直接读取真实 MultiHead 字段，缺失即报错。`output_heatmaps` 在构造模型时设置。
- 官方测试配置的 `MaskBackground` 要求掩膜；本轮明确设置 `continue_on_failure=True` 且传入空掩膜，图像保持原样。不是用真值掩膜，也不是完整 BMP 的效果复现。
- 低维几何和人体状态仍运行 CPU。四视角当前裁剪批处理运行 CUDA；保留原接口作数值一致性检查。缺失视角不补零观测；格式错误停止实验；实时接入仍须执行原同步、过期和未知处理。

## 固定对照与验收

1. S1/S2 各前 500 帧开发，冻结实现后 S4/S6 各前 750 帧复验。S4/S6 已在历史轮次查看，不能称为独立未见测试集。
2. 复用原始 RGB、标定及已冻结的 M 人物关联，输入同一个 `xyxy` 框。两模型各用官方裁剪预处理，不能称为像素完全相同的网络输入。RTMW 足手细节不变。
3. PMPose 使用官方预测 OKS 作为原重建器分数；不依据参考数据调整阈值，也不把可见性直接乘入分数。保留存在概率/可见性供独立审计。
4. `infer` 不读取参考；先保存哈希绑定的二维结果，`score` 才重建并读参考评分。验证源视频哈希、选人结果哈希、曝光、相机顺序及框。基线 M 是此前已验证与新推理逐值相同的缓存，不冒称本轮重新测量 M 耗时。
5. 检查所有目标 ≤100 mm 比例、覆盖率、平均/P95 误差、>500 mm 错误、相位/速度/恢复及推理耗时；不能用丢弃困难关节改善均值。开发收益不足以默认替换。
6. 12 个人体对应点只是近似参考。二维误差来自三维参考投影；不代表独立二维遮挡标注。存在事件只审计图像内关节是否落在裁剪支持域。每 25 帧保存一次概率图，报告 50/90/95% 最高密度区域的覆盖和面积；不据此声称校准保证。缺少真实可见性标签，故不报告其准确率。

## 环境与复现

默认 `.runtime.json` / `.robot-runtime.json` 保持原路径。独立 `venv-py311-pmpose` 位于相同外部运行时根目录；CPython 3.11.9、PyTorch 2.1.2+cu121、MMCV 2.1.0 Windows CUDA 轮子。完整依赖见 `requirements-pmpose-lock.txt`，不要安装到默认环境。

创建独立环境后，使用 `uv pip install --python <独立环境的python.exe> --index-strategy unsafe-best-match -r requirements-pmpose-lock.txt` 安装锁定依赖，再安装下述固定源码；索引仅为锁定文件中的官方 PyTorch 和 PyPI。

官方源码：[BBoxMaskPose](https://github.com/MiraPurkrabek/BBoxMaskPose/tree/49a070a6f8396147323b1c4959474077dbe2ce8e)，GPL-3.0。外部源码使用 `git config core.longpaths true` 支持 Windows 路径，再以 `uv pip install --no-deps --no-build-isolation -e <repo>` 安装；无需 SAM2/3D 模型。`setuptools==69.5.1` 保留官方依赖所需的 `pkg_resources`。

权重：[PMPose-S-1.0.0](https://huggingface.co/vrg-prague/BBoxMaskPose/blob/main/PMPose/PMPose-s-1.0.0.pth)，161,604,468 bytes，SHA256 `7255986ab80a4e2c7b60f135fd4b0ffb9438991f21ac1c23a7c54cd1ecf79bee`。源码、权重、环境均在仓库外；不重复下载视频。

在项目根目录运行（`$base` 为 `.runtime.json` 中 Python 路径向上三级的外部运行时根目录）：

```powershell
$rt = Get-Content .runtime.json | ConvertFrom-Json
$base = Split-Path (Split-Path (Split-Path $rt.python))
$out = 'outputs/frontier_research_20261018/pmpose'
& "$base\venv-py311-pmpose\Scripts\python.exe" scripts/benchmark_pmpose.py infer --output "$out/development" --split development --repository "$base/research-pmpose/BBoxMaskPose" --checkpoint "$base/research-pmpose/PMPose-s-1.0.0.pth"
& $rt.python scripts/benchmark_pmpose.py score --output "$out/development"
& "$base\venv-py311-pmpose\Scripts\python.exe" scripts/benchmark_pmpose.py infer --output "$out/heldout" --split heldout --expected-freeze "$out/development/freeze.json" --repository "$base/research-pmpose/BBoxMaskPose" --checkpoint "$base/research-pmpose/PMPose-s-1.0.0.pth"
& $rt.python scripts/benchmark_pmpose.py score --output "$out/heldout"
```

结果目录必须是新的；复验请使用新目录，成功后按项目清理约定保留必要证据。代码变更使冻结校验失效，需要重新推理。耗时是离线处理时间，不能当作相机曝光到状态输出延迟。

`scripts/audit_pmpose_latency.py --method pmpose_s --output <新JSON路径>` 在独立环境运行；同脚本的 `--method medium` 在默认视觉环境运行。它固定 S1 首帧四个当前框，执行预热和 20 次交替测量，保存全部耗时及像素一致性、GPU 状态和源码哈希；不读取参考标签。
