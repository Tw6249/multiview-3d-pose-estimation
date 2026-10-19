# 安装、复现与历史结果

本文由 README 迁移，内容保持原文，仅调整标题层级和相对链接。最新进展以 [研究日志](research_log.md) 为准。

## README 历次状态摘要

**纠正先前的范围错误：** Panda 单臂到双腕的映射、抬手触发行走，不满足本项目的全身动作模仿要求。此前 Panda 仿真报告不能作为本任务验收。已有的 CUDA 人体感知和时间戳回归继续复用；原固定基座双臂示例仅作为历史模块回归。

当前控制工作以 29 自由度 G1 模型为目标；实际 G1 EDU 的关节配置仍需与设备核对。尚未进行物理 G1 联调，也不把运动学动画称为动力学验证。

本轮完成视觉四项首轮实现与配对评测：五帧联合估计、更大二维前端、旋转/速度可观测性和经验误差范围。联合拟合、速度候选和 RTMPose-X 均未通过完整接受条件，保留原默认姿态算法。已修复文件回放源积压，并将本机回放设为经过逐值一致性检查的 4 路检测/4 路姿态推理；直行实测约 11.3 Hz，未达到 24 Hz。机器人环境 249 项回归通过。见 [验收](../reports/vision-joint-state-2026-10-18.json)、[视觉模块、研究依据与复现](vision_joint_state.md)。

上一轮修复足部支撑语义并审计源—机器人触地相位；接触仍缺独立真值，已知 10 段完整模仿仍为 0/10，本轮没有重跑或改善动力学控制。见 [足部证据记录](../reports/contact-observation-2026-10-18.json)。

上一轮已修复人体朝向速度到机器人当前朝向速度的坐标转换，固定 `support_aware` 的平衡通过 3/10→5/10、停止仍 4/10；路径在 6 段退化。落脚/角动量规划器保持诊断状态，过渡默认仍为 `legacy`，见 [速度坐标与规划记录](../reports/landing-momentum-2026-10-18.json)。

上一轮 4 秒分支预测曾在旧速度坐标路径上把末段停止 4/10→5/10，但新增成功方案在模型偏差下跌倒，规划耗时约 6.8–9.1 秒，保持离线实验。该历史结果不与新坐标路径混用，见 [退出规划记录](../reports/recovery-rollout-2026-10-18.json)。

此前接触约束和瞬时制动力检查的负面结果继续保留：[接触约束](../reports/contact-retargeting-2026-10-18.json)、[停止过渡](../reports/stop-transition-2026-10-18.json)。不能把参考足底几何改善、延迟倒地或倒地后静止称为成功。

### 视觉优化与调研记录

针对错误/缺失观测与恢复跳变，新增[研究核查及下一轮验证方案](observation_recovery_research_2026-10-18.md)：区分观测风险、独立恢复候选和机器人参考接续，补充 TAPNext++、CUPS、MEOM 等依据及在线限制。本轮为调研，未改变默认算法。

随后建立带标签的恢复事件评测，并用此前未使用的 S3/S5/S7/S8 组成冻结测试集。可选的轨迹确认未通过覆盖率和误接受门槛，默认不变；主要失效是所有视角一致的左右互换，下一步做双假设身份恢复。见 [恢复事件验收](../reports/recovery-events-2026-10-18.json)。

最新补充：[2026-10-18 前沿论文、公开效果与迁移顺序](vision_frontier_2026-10-18.md)。随后完成 [PMPose 同框对照](pmpose_experiment.md)：4 人、2,500 帧、9,472 个裁剪；复验共同点误差 49.51→46.96 mm，但覆盖率略降，标称 90% 位置区域实际仅覆盖约 56%–65%。保留为研究候选，默认 M 与控制门不变。见 [本轮验收](../reports/pmpose-observation-2026-10-18.json)。此前的 [原始 SimCC 接口](../reports/frontier-research-2026-10-18.json)已验证保留分布不改变原解码。

第三轮在无真实相机数据条件下修正了关节关联、尺寸约束及速度状态一致性，增加共同集合评测和随机扰动测试，并优化 CPU 几何及 GPU 调度。原渲染 RGB 平均代理误差由第二轮 49.96 mm 恢复到 47.54 mm，随机合成测试平均位置误差由兼容基线 24.72 mm 降到 14.81 mm。**仍为实验路径，不代表真人精度、前沿排名或实机就绪；G1 默认输入保持兼容。** 详细结果见 [第三轮说明](vision_pipeline.md#第三轮无真实相机数据条件下的修正) 和 [第三轮报告](../reports/vision-revision-2026-10-17.json)，历史见 [第二轮报告](../reports/vision-motion-2026-10-17.json)。

```powershell
# 真实 CUDA 计算的定时文件回放；不是物理曝光延迟验收
.\run.ps1 -Stage stream-vision -InputSession outputs/human_walk_g1 -Session outputs/my_stream -DetectorWorkers 4 -PoseWorkers 4
# 对已保存的二维观测运行新重建与人体状态模块
.\run.ps1 -Stage refine-motion -Session outputs/my_motion
```

[研究调研](vision_research_2026-10-17.md)记录当前问题、近期研究来源和模块化方案；[视觉模块与复现](vision_pipeline.md)记录已实施的接口与限制。

[简短研究日志](research_log.md)持续记录对话决策、实验得失及下一步。2026-10-18 已新增跨视角重复肢体/弱实例校验，完成另外两人的独立片段测试；状态恢复实验未通过成功率验收，保持关闭。见 [本轮验收](../reports/vision-consistency-2026-10-18.json)。

随后扩展到四段各 30 秒的连续真实视频，完成 3,000 帧对照；持续观测恢复仍未通过验收，默认保持关闭。长片暴露出可用视角不足及前端对应问题，见 [连续恢复记录](../reports/vision-recovery-continuous-2026-10-18.json)。

新视觉配置保留 RTMPose 身体点，用 RTMW 补足掌部点，输出 33 点；加入独立因果状态估计、可修订的尺寸约束、最多 100 ms 的显式预测及过期处理。RTMW 足掌推理合批，身体网络保留逐图推理，检测框 NMS 转移到 CPU，三角化复用相机矩阵与解析导数。

同一行走 RGB 上，身体代理误差平均 60.87 → 47.63 mm，P95 131.54 → 113.39 mm，右小腿长度变异系数 20.77% → 1.32%。含预测的状态覆盖不等于直接观测覆盖。完整视觉处理约 8.82 组/秒，尚未达到 24 FPS；也尚未完成真实人体精度和物理 G1 验证。见 [视觉验收报告](../reports/vision-optimization-2026-10-17.json)。

```powershell
.\run.ps1 -Stage vision-demo -InputSession outputs/human_walk_g1 -Session outputs/my_vision_run
```

当前输出 `outputs/vision_optimized/comparison.mp4` 展示四路 RGB、原始骨架与状态估计。G1 默认消费原始身体观测，新的预测状态和足掌朝向尚未直接成为控制指令。

## 当前人体到 G1 示例

```powershell
python scripts/bootstrap.py --device cuda
python scripts/bootstrap_robot.py
.\run.ps1 -Stage robot-test
.\run.ps1 -Stage robot-demo -Session outputs/my_human_g1 -Device cuda
```

当前输出：`outputs/human_walk_g1/imitation/preview.mp4`。画面依次为四相机人体 RGB、按尺寸重定向的 G1 参考、自由基座物理仿真中的实际 G1。
使用 96 帧/384 张真实推理的渲染图像；86 帧允许腿部控制，手臂遮挡单独处理。全身参考驱动 29 关节 TWIST2 跟踪器，无手势触发行走。人体动画仅生成 RGB；感知与控制不读取动画骨架或真值。

**完整验收尚未通过。** 当前通过平衡与主要腿部动作一致性检查；骨盆路径 P95 跟踪误差约 0.77 m，超过 0.25 m 阈值。脚掌/手掌完整朝向尚不可由 COCO17 观测。程序保留失败报告并返回非零状态，不能把该视频作为物理机器人就绪证明。见 [当前报告](../reports/human-g1-2026-10-17.json)及[模块、科学问题与硬件边界](source_robot_to_g1.md)。

`g1_commands.jsonl` 包含具名 29 关节目标、PD 参数、力矩限幅、模型哈希、序号和有效期；目前只进入模拟接收器。物理 G1 的 DDS/CRC、反馈/里程计、机型核对和独立停止机制尚未联调。

## 人体输入与通用双臂回归示例

本项目将 **Blender 虚拟相机**与 **真实图像可复用的姿态估计管线**分开。
目标是在本机跑通单人、多相机感知到机器人动作的模块化链路，再替换真实相机与机器人驱动。

**人体输入示例：`outputs/robot_demo`。** 4 视角 RGB 经 CUDA 姿态估计、三维重建、质量筛选、
因果滤波、双臂 IK、轨迹约束，生成 50 Hz 关节指令，并通过模拟执行器反馈闭环验证。
默认机器人是可配置的固定基座 8 自由度双臂，尚未接入物理机器人。
模块职责、关键科学问题、未解决项与硬件接入条件见 [架构与硬件迁移](architecture.md)。
本次结果与源码指纹见 [控制链路验收报告](../reports/robot-control-2026-10-17.json)。

本次生成 92 条指令；12 帧观测中 9 帧通过控制准入，1 帧几何不确定性过高、2 帧骨长异常被拒绝。
指令位置、速度、离散加速度、时间有效性和静止结束检查全部通过；34 项回归测试通过。
RTX 4090 实测推理平均 22.12 ms/图像，当前动作处理与 IK 平均约 4.6 ms/帧。
50 Hz 是控制回放的虚拟时钟频率，输入延迟显式注入 120 ms，**不是实机端到端实时性能结论**。

```powershell
.\run.ps1 -Stage demo -Session outputs/my_robot_session -Device cuda
# 对新格式 poses3d 单独生成指令（目标 session/control 不得已存在）
.\run.ps1 -Stage control -Session outputs/my_pose_session
```

`control/commands.jsonl` 为带模型哈希、关节名、位置/速度、时钟域、序号和有效期的指令；
`feedback.jsonl` 为模拟反馈，`ros2_trajectory.json` 为 ROS 2 消息布局导出，
`control_preview.gif` 展示执行过程，`control_report.json` 保存验收结果。
当前 ROS 导出供仿真回放，实际驱动接入、连续轨迹重新定时、碰撞/动力学验证仍取决于机器人型号。
旧 `poses3d` 没有不确定性字段时，先运行 `-Stage reconstruct`。

## 视觉基线

本机重新复现的正式结果在 **`outputs/gpu_session02`**（2026-10-16，RTX 4090）。
输出目录、模型与运行时不随 Git 提交；下方命令可重建结果。中断的渲染和被替代的基准实验已清理。

4 视角 × 12 帧 = 48 张图像全部检测到单人；重建 199/204 个 COCO17 点（97.55%）。
12 个身体骨骼代理点的有效覆盖率 99.31%，平均代理误差 53.89 mm、中位数 38.01 mm、
P95 142.92 mm；平均内点重投影误差 2.93 px。这不是人体真实标注精度，见下方代理真值限制。
相机导出与 Blender 投影最大差异 0.000054 px。11 项测试通过，覆盖几何、真实输入格式、
空场景、时间戳匹配回归及 CUDA 静默回退拒绝。

OptiX 渲染与降噪共 31.76 秒；正式 session 的 CUDA 预热后平均推理 22.75 ms/图像，
四视角连同图像读取平均 110.42 ms/组。当前仍是串行离线处理，未宣称达到 12 fps
真实采集、重建与显示的端到端性能。旧 CPU 示例与本次 OptiX 示例的渲染图像不同，
不能直接用旧数字计算加速比；同图像对照使用 `scripts/benchmark.py`。

```text
Blender：人体动画 + 4 台同步虚拟相机
       ↓ RGB + 相机标定 + 时间戳
YOLOX 人体检测 → RTMPose COCO17 二维关键点
       ↓
置信度筛选 → 多视角 RANSAC / 加权 DLT → 鲁棒重投影优化
       ↓
世界坐标中的 17 个关节点 + 有效性 + 质量指标
       ↓
质量门控 / 因果滤波 → 人体坐标系 → 机器人运动学映射
       ↓
位置 / 速度 / 加速度约束 → 具名关节指令 → 模拟执行反馈
```

**真实图像迁移时保留 YOLOX、RTMPose、三角化和结果格式。**
Blender 仅负责生成测试输入。模型真值只由独立评测程序读取，绝不充当二维检测结果。

## 人体回归插件范围

- 单人，四台固定相机，640×640，默认 12 fps / 12 帧的短动作片段。
- 输出 COCO17；置信度不足、视角不足或几何退化的关节写为 `null`，不填入真值或伪造坐标。
- 实际检测渲染出来的 RGB，ONNX Runtime CUDA/CPU 可选；本机默认 CUDA。
- Blender Cycles 支持 OptiX/CUDA/CPU；本机使用 OptiX 渲染与降噪，运行时与系统 Python 隔离。
- 真实图像与仿真图像共享 `cameras.json`、`frames.jsonl` 数据接口。
- 示例 CesiumMan 是风格化的公开人体动画，适合功能验收，不是写实人体精度基准。
- 示例模型仅有 12 个身体骨骼关节的评测代理值。面部 5 个点没有真值；骨骼中心与 COCO 标注定义也不完全相同。`body12_proxy_mpjpe_mm` 必须按此解释。
- 默认理想针孔、全局快门、精确同步。重建代码支持真实 Brown-Conrady 畸变；本例不模拟卷帘快门、真实曝光、网络延迟或相机触发电路。
- 已包含双臂方向重定向、关节指令约束和模拟执行反馈；不含多人跨视角身份关联、SMPL 拟合、真实机器人驱动、全身平衡或动力学控制。

## 安装与启动（Windows PowerShell）

```powershell
python scripts/bootstrap.py --device cuda
.\run.ps1 -Stage test
.\run.ps1 -Stage demo -Session outputs/my_new_session -Frames 12 -Device cuda
```

安装脚本使用固定 uv 0.9.7 下载独立 Python 3.11.9，不再沿用系统 Python 版本；
从带 SHA256 的完整 CPU/GPU 依赖锁文件同步独立环境。GPU 环境固定 ONNX Runtime GPU 1.22.0、cuDNN 9.10.2.21，
CUDA/cuDNN DLL 由 pip 包提供，不需要另装完整 CUDA Toolkit；需要可用的 NVIDIA 驱动。
Blender 4.5.3 压缩包与固定提交的 CesiumMan 均校验 SHA256；balanced 模型也核对
`assets/model_checksums.json` 中的 SHA256。当前只对 balanced 模式完成全流程复现。
运行时位于 `%LOCALAPPDATA%/multiview-pose-sim`，不修改系统软件关联或基础 Python。
首次姿态检测下载官方模型到 RTMLib 缓存。Windows 的 CUDA/cuDNN DLL 搜索路径仅在当前进程配置。
CUDA 模式会检查两个实际模型会话并拒绝静默 CPU 回退；部分形状算子仍由 ONNX Runtime 分配给 CPU。
DLL 预加载按 [ONNX Runtime 官方说明](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#preload-dlls)实现。
两种后端均固定 ORT 线程数（intra-op 4 / inter-op 1），关闭线程忙等。
CUDA 默认请求 `use_tf32=0`，使相同图像的 CPU/CUDA 结果更接近。
首次预热可能出现 `No execution plans support the graph`：ORT 1.22 的 cuDNN 前端会在候选计划
不受支持时于 CUDA 内部重新选择计划，并可能重新允许 Tensor Core 运算。
这是 [ORT 1.22 卷积实现](https://github.com/microsoft/onnxruntime/blob/v1.22.0/onnxruntime/core/providers/cuda/nn/conv.cc#L208-L220)
中的重试逻辑；本机完整运行与算子记录均验证成功，不能据此宣称所有卷积严格禁用 TF32 或结果逐位一致。

无 NVIDIA GPU 时可用 `python scripts/bootstrap.py --device cpu`，随后以 `-Device cpu` 运行。
CPU/GPU 环境分开保存；每次 bootstrap 会让 `.runtime.json` 指向选中的环境。

同图像 CPU/CUDA 对照及独立算子分析（需要 GPU 环境）：

```powershell
$rt = Get-Content .runtime.json -Raw | ConvertFrom-Json
& $rt.python scripts/benchmark.py --session outputs/gpu_session02 --output outputs/my_benchmark
```

基准测试在独立进程中保留各自检测/重建结果，检查模型哈希相同，记录预热后的平均/P95 耗时及三维点差异。
CUDA 算子分析单独运行一帧，不计入 CPU/GPU 测速。`comparison.json` 汇总结果，
`cuda_profile/profiles` 保存原始算子事件。`run.ps1 -Stage detect -Source coco17 ... -Profile` 也可单独开启分析，
但此时耗时包含分析开销。

本次最终对照在 `outputs/benchmark03`：CPU 1128.90 ms/图像，CUDA 22.54 ms/图像，
实测加速 50.09 倍；两者身体 12 点平均差异 0.0207 mm、最大 2.57 mm。
全部可比关节点的最大差异 12.98 mm，包含没有代理真值的面部点。两种后端均重建 199/204 点。
YOLOX / RTMPose 的 CUDA 算子事件分别为 2191 / 1295 次，另有少量 CPU 形状算子。
可随代码查看的摘要见 [本机复现报告](../reports/rtx4090-2026-10-16.json)；该摘要不是完整图像数据集。

已存在的仿真 session 不会被渲染器覆盖。重复实验请换输出目录。
也可以对已有图像单独运行：

```powershell
.\run.ps1 -Stage detect -Source coco17 -Session outputs/gpu_session02 -Device cuda
.\run.ps1 -Stage reconstruct -Session outputs/gpu_session02
.\run.ps1 -Stage evaluate -Session outputs/gpu_session02
.\run.ps1 -Stage preview -Source coco17 -Session outputs/gpu_session02
.\run.ps1 -Stage open-scene -Session outputs/gpu_session02
```

可在 `config/demo.json` 修改相机方位、距离、高度、焦距、分辨率、帧数、同步容差及重建阈值。
相机朝向统一看向人体中心。渲染器逐相机拍摄同一动画帧，期间不推进动作时间。

## 结果文件

| 文件 | 含义 |
|---|---|
| `scene.blend` | 可编辑的人体、灯光、地面、四相机场景 |
| `images/camXX/000000.png` | 原始虚拟相机 RGB 图像 |
| `cameras.json` | 内参、畸变、世界到相机的外参，米制 |
| `frames.jsonl` | 每个同步图像包及逐相机时间戳 |
| `detections.jsonl` | 实际图像检测得到的 17 个二维点与置信度 |
| `poses3d.jsonl` | 三维坐标、内点相机、重投影误差、三角化角度 |
| `poses3d.quality` 中的协方差字段 | 假设 2 px 独立噪声的局部几何不确定性；不含标定/同步误差 |
| `ground_truth.jsonl` | 只用于仿真评测的骨骼代理真值及其投影 |
| `metrics.json` | 实际检测重建的误差、有效率；以及独立几何自检结果 |
| `render_validation.json` | 导出相机与 Blender 投影是否一致 |
| `render_runtime.json` | 实际渲染/降噪后端、GPU 名称、渲染耗时 |
| `inference_timing.json` | 实际模型执行后端与哈希、预热、平均/P95 推理耗时 |
| `reproduction.json` | 完整性检查、环境版本、源码/图像/输入与结果文件 SHA256 |
| `preview.png / .gif / .mp4` | 四视角检测结果与三维重建对照 |
| `control/observations.jsonl` | 动作观测、拒绝原因、人体坐标系、IK 目标与残差 |
| `control/commands.jsonl / feedback.jsonl` | 实际生成的指令及模拟反馈 |
| `control/control_report.json / control_preview.gif` | 控制验收指标与动画 |

灰色为骨骼代理真值，蓝色为根据 RGB 重建的三维点。指标只统计有效重建点，因此必须同时检查有效率。
`oracle_geometry_max_error_mm` 使用理想投影点，只验证几何实现，不能作为视觉系统精度。

## 迁移到真实设备

1. 保持每台相机焦距、对焦和分辨率固定，完成内参及畸变标定。
2. 联合标定外参到同一个世界坐标系。需要地面坐标时明确地面平面与 Z 轴方向。
3. 用设备 SDK 获得图像及共同时间基准下的**曝光时间戳**。主机收到帧的时间不是曝光同步证据。
4. 保存以下标定格式，以及 `camera_id,timestamp_ns,image` CSV 清单。
5. 用 `scripts/import_real.py` 组包；之后运行完全相同的 detect / reconstruct / preview。

示例相机（这里只示范结构，数值必须替换成真实标定结果）：

```json
{
  "schema_version": 1,
  "units": "metre",
  "world_axes": "right-handed, Z-up",
  "camera_axes": "OpenCV: X-right Y-down Z-forward",
  "cameras": {
    "cam00": {
      "K": [[750,0,320],[0,750,320],[0,0,1]],
      "R_world_to_camera": [[1,0,0],[0,1,0],[0,0,1]],
      "t_world_to_camera_m": [0,0,0],
      "distortion": [0,0,0,0,0],
      "width": 640,
      "height": 640
    }
  }
}
```

`X_camera = R_world_to_camera @ X_world + t_world_to_camera_m`。
`t` 不是相机位置；相机中心为 `-R.T @ t`。畸变系数顺序为 OpenCV 的 `k1,k2,p1,p2,k3`。
仿真世界为右手系、Z 向上，所有长度为米。真实世界坐标可以另选，但标定文件需描述清楚且各相机一致。

```csv
camera_id,timestamp_ns,image
cam00,1000000000,cam00/000000.png
cam01,1000001000,cam01/000000.png
```

CSV 的图片路径相对于 CSV 文件，必须包含至少两台已标定相机。使用隔离环境：

```powershell
$rt = Get-Content .runtime.json -Raw | ConvertFrom-Json
& $rt.python scripts/import_real.py --csv real/frames.csv --calibration real/cameras.json --output outputs/real01 --max-skew-ms 5 --fps 30
.\run.ps1 -Stage detect -Source coco17 -Session outputs/real01
.\run.ps1 -Stage reconstruct -Session outputs/real01
.\run.ps1 -Stage preview -Source coco17 -Session outputs/real01
```

导入器以第一台相机的曝光时间为锚点，匹配容差内最近的未使用帧；距离相同时选较早曝光。
若最近帧已被占用，会继续考虑其他候选，不再直接丢掉整个视角。
门限限制的是各视角相对锚点的时间差，不是包内任意两台相机的最大时间差。
它不会建立硬件同步；运动越快，对时间误差要求越高。没有真实真值时跳过 evaluate。
设备采集 SDK、触发接线和标定结果仍需在具体硬件确定后接入，当前没有宣称已验证真实设备。

## 公开真实数据评测

2026-10-18 最新一轮增加 4×30 秒连续评测：跨视角人物候选选择在 1,500 帧留出片段中将 >500 mm 身体点错误从 4 降到 0，已启用；不能据此称感知完美。修复人体尺寸更新导致骨盆路径依赖世界原点的问题。新增轨迹约束、原始目标评分及多场景动力学检查；平滑方案未通过完整 G1 验收，未提升为默认控制。机器人环境 156 项回归通过。见 [简短研究日志](research_log.md)、[本轮证据](../reports/vision-person-trajectory-2026-10-18.json)、[轨迹接口与复现](source_robot_to_g1.md)。

已接入 MPI-INF-3DHP 的原始多视角 RGB、标定与逐帧三维参考。首轮完成 2 人、4 视角、6 个片段的 CUDA 基线测试，现已增加另外 2 人、6 段的留出测试，共 1,800 帧真实多视角状态。
数据参考来自多相机无标记动捕，不是独立 Vicon；本地测试使用官方训练部分，不能与论文官方 test 排行榜横比。
数据、模型和生成视频均在忽略目录中，感知过程不读取参考动作。

- [16 个公开数据资源、访问实测与复现方法](public_real_datasets_2026-10-17.md)
- [本轮真实数据验收记录](../reports/public-real-data-2026-10-18.json)
- 本地对照视频：`outputs/public_datasets/mpi_results/real_multiview_36s.mp4`

## 后续实验

建议先替换更接近真人的、具有明确关节定义的人体资产，再扩展动作、体型、服装、背景和光照。
每次扩展需保留独立真值导出及相机投影自检。可比较 2/4/6/8 相机、遮挡、失步与标定误差。
不要仅追求重投影误差小：错误的跨视角对应也可能给出看似一致的三维结果。

