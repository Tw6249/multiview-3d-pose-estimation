# 视觉姿态模块与复现

## 2026-10-18：独立视角选择人体候选

`pose_sim/person_selection.py` 输入带名称的二维像素候选、标定和曝光整数 ns，输出同时间戳的单一候选或原缺失视角。使用原本只有一个检测框的当前视角三角化肩/髋/膝，再投影到歧义相机；至少四个锚点，包含肩部及髋或膝。中位残差 ≤20 px、P80 ≤35 px，第二候选差距 ≥15 px。只允许 2–4 个检测框，不能用刚选出的相机去相互证明另一歧义相机。阈值是启发式，不是校准身份概率。

配置 `config/vision.json: cross_view_person_selection=true`，流式入口和 `vision_demo.py` 均接入。混合模型先以 RTMPose 身体点选择，仅为选中框补 RTMW 手脚；未计算细节点分数为零，不当作观测。CUDA 推理保留原模型与身体 batch-1 数值；选择和几何在 CPU。`person_selection_ms` 单列 CPU 选择时间，候选/细节推理计入 `pose_ms`。证据不足、失步或全视角歧义时保持缺失。

连续留出 1,500 帧的 >500 mm 身体点错误 4→0，覆盖率和全目标成功率均增加；渲染回归存在 1 个成功目标的损失。15 包真实 CUDA 集成检查与冻结缓存关键点、分数一致，平均检测 110.2 ms，P95 186.6 ms，尚不满足全部帧实时预算。数据与独立性限制、负面结果见 [研究日志](research_log.md) 和 [验收](../reports/vision-person-trajectory-2026-10-18.json)。

```powershell
$rt = Get-Content .runtime.json | ConvertFrom-Json
& $rt.python scripts/check_person_selection_stream.py --input outputs/public_datasets/mpi_long/inputs/s1_long --detections outputs/public_datasets/mpi_long_detections/s1_long/detections.jsonl --candidates outputs/public_datasets/mpi_long_person_candidates/s1_long.jsonl --output outputs/person_selection_integration.json
```

冻结评分后仅修改推理调度、计时与裁切类型保持；没有调整选择阈值或留出评分算法。原冻结指纹与最终实现指纹均保留，不能用最终指纹冒充先前冻结版本。

## 2026-10-18：真实数据驱动的观测校验

过程与负面实验见 [简短研究日志](research_log.md)，数值与指纹见 [本轮验收](../reports/vision-consistency-2026-10-18.json)。

| 模块 | 输入与输出 | 失败行为与验证 |
| --- | --- | --- |
| `observation_consistency.py`（默认启用） | 已标定、同步的具名 2D 像素/置信度；相机采用世界到相机变换，世界长度为米，曝光时间为整数纳秒。输出复制的观测和拒绝证据，只撤回分数，不生成新坐标。 | 其他视角独立投影否定重复左右肢体或弱实例；证据不足不认证身份。真实重叠、过期视角和无历史输入均有测试；全局左右对称仍无法解决。 |
| `reacquisition.py`（实验关闭） | 当前三维观测（世界米）、几何质量和曝光纳秒时间；持续保存最近三次受支持的观测，包括滤波器已接受的观测。三视角或两视角人体结构支持仍是启发式证据。 | 缺失、失步、来源变更和不连续候选清空。创新门拒绝时，连续证据可重建轨迹；过期后的新几何仍可显示，但标为 `reacquiring`，不提供旋转依据、不参与下一帧身份先验、不直接驱动机器人。候选尚未通过真实数据验收。 |

新默认由 `config/vision_motion.json` 控制，接入流式处理、离线 `refine_motion.py` 和展示脚本。研究类 `benchmark_motion.py` 的 `joint` 保留此前基线，`consistency` 是本轮选择，`candidate` 为含恢复的实验组合。测试使用官方训练分区中预留的 S4/S6，未按其结果调参；该参考仍来自无标记动捕，尚无独立测量保证。

```powershell
$rt = Get-Content .runtime.json | ConvertFrom-Json
& $rt.python -X utf8 scripts/evaluate_observation_recovery.py --dataset outputs/public_datasets/mpi_heldout --baseline outputs/public_datasets/mpi_heldout_baseline --selected-method consistency --output outputs/my_consistency_check
& $rt.python -X utf8 -m unittest tests.test_observation_recovery -v
# refine_motion 增加 --no-observation-consistency 可复现此前观测基线。
```

### 连续片段恢复实验（2026-10-18）

`config/public_mpi3dhp_long.json` 固定四段各 30 秒的四相机视频窗口，S1/S2 开发、S4/S6 冻结后复核。复用本地原视频，不复制图像；CUDA 检测只读取 RGB、标定和时间，参考只在推理结束后进入评分。S4/S6 是此前留出人物的新窗口，不是新人物或独立数据集。

`scripts/benchmark_recovery.py` 将 GPU 检测缓存与 CPU 算法对照分开。输入沿用源曝光时间及标定世界坐标；输出缓存、SHA256 和简短评测报告。缓存变化、缺帧、曝光键不一致或冻结代码变化时明确失败，不补帧。`baseline` 和 `recovery` 都启用已验收的观测一致性，仅比较状态恢复开关。

```powershell
$rt = Get-Content .runtime.json | ConvertFrom-Json
& $rt.python -X utf8 scripts/import_mpi3dhp.py --config config/public_mpi3dhp_long.json --output outputs/my_long_inputs
& $rt.python -X utf8 scripts/benchmark_recovery.py cache --dataset outputs/my_long_inputs --output outputs/my_long_detections
& $rt.python -X utf8 scripts/benchmark_recovery.py replay --dataset outputs/my_long_inputs --cache outputs/my_long_detections --output outputs/my_long_development --split development
# 记录代码指纹和参数、作出选择后再评分 heldout；--expected-hash 校验代码指纹。
```

除共同目标位置误差、覆盖率与全目标成功率外，新增与状态名称无关的位移误差，包含 `reacquiring` 帧及缺失后的再出现，防止把大跳变移到“过渡”状态就误报改善。它仍不是带人工遮挡标签的恢复指标。完整解剖旋转、脚掌接触和脚滑缺少相应参考，保留为空。短片与连续片段的负面结果也保留；详见 [连续恢复验收](../reports/vision-recovery-continuous-2026-10-18.json)。

误差报告同时保留全部目标分母、覆盖率、两两共同集合、大误差、恢复/缺失后重现跳变与相位。过渡状态单列，不伪装成当前观测或预测。未使用参考动作进行推理，未改模型或环境，未向 G1 发布动作。

本轮实现依据 [研究调研](vision_research_2026-10-17.md)。当前配置为 CUDA YOLOX-M 人体检测、RTMPose-M 身体关键点、RTMW-L 足部和手掌关键点、多视角几何重建、独立因果人体状态估计。保留两个姿态模型，是因为同一行走样本上 RTMW 单独替换身体模型虽然平均代理误差更低，但髋部观测覆盖和原控制器准入率下降。该选择只在当前合成片段上验证。

## 运行

已有 `.runtime.json` 和 `.robot-runtime.json` 的本机环境可直接执行：

```powershell
.\run.ps1 -Stage vision-demo -InputSession outputs/human_walk_g1 -Session outputs/my_vision_run
```

输入 session 必须含 `frames.jsonl`、`cameras.json`、`config.json` 及其中引用的 RGB。只在独立评估时读取可选的 `ground_truth.jsonl`。输出目录必须不存在，不复制输入图像、不覆盖基线。

`vision-demo` 使用固定机器人环境中的 ONNX 工具，将检测器中 NMS 之前的输出提取到模型缓存，再由固定 CUDA 环境推理。原始权重不修改；派生模型校验原始及自身 SHA256。没有给视觉环境安装额外依赖。

```powershell
$rt = Get-Content .runtime.json -Raw | ConvertFrom-Json
& $rt.python scripts/vision_preview.py outputs/human_walk_g1 --output outputs/my_vision_run --baseline outputs/human_walk_g1
```

当前验收结果位于 `outputs/vision_optimized`：`comparison.mp4` 为按输入时间播放的离线对比，`comparison.png` 为骨长稳定性图。播放帧率不代表处理速度。

独立阶段入口：`pose_sim.pipeline detect/reconstruct`、`pose_sim.human_state`、`pose_sim.vision_metrics`。检测器、关键点模型和批处理由 `config/vision.json` 选择；人体状态参数在 `config/vision_state.json`，不含 G1 参数。

## 数据接口

| 输出 | 含义及失败行为 |
| --- | --- |
| `detections.jsonl` | 各视角的具名二维点、分数、曝光时间和模型来源。只接受一个人；图像外预测分数置零，不当成可见观测。模型签名随记录保存 |
| `poses3d.jsonl` | 独立三角化的原始米制世界坐标及几何质量，缺失点为 null；不以状态估计覆盖原始观测 |
| `states3d.jsonl` | `human_state_v1`：位置、速度、每点状态、最后观测时间、观测年龄、协方差近似、骨长和可观测的足掌坐标系。绑定原始姿态文件 SHA256 |
| `vision_metrics.json` | 原始与状态输出的分项误差、覆盖、骨长波动、异常原因、朝向可用帧数及各阶段耗时 |
| `run_manifest.json` | 输入清单、RGB 序列和源码哈希、模型哈希、环境版本、配置和性能摘要 |

`human33` 的顺序是 COCO17、左右各三个足部点（大脚趾、小脚趾、脚跟）、左右各五个掌部点（手腕、拇指 MCP、食指 MCP、中指 MCP、小指 MCP）。RTMW 内部输出 133 点，但本阶段只提取动作模仿需要的部分；没有宣称恢复全部手指或面部三维动作。下游按名字取点。

`joint_states.status`：

- `filtered`：本帧存在通过质量和创新量检查的三维观测，经过运动估计和结构约束。
- `predicted`：本帧未接受新观测，仅由历史状态短时外推。最长 100 ms，`last_observed_timestamp_ns` 不更新，`quality` 为 null。
- `missing`：没有可用历史或预测过期，位置为 null。

时间必须是严格递增整数 ns。人物、标定、关键点模式或模型签名变化，以及长时间间隔，都会清空运动和骨长历史。骨长由最初 12 个被接受的样本取中位数，之后固定；初始估计仍可能有模型域偏差，不能当作实测人体尺寸。

状态估计是恒速度 Kalman 模型和带权骨长约束，没有未来帧输入。协方差含启发式观测噪声下限和运动模型假设，未校准真实覆盖概率。修正后重投影误差重新计算，不复用原始点的误差值。

足部坐标系 X 从脚跟指向脚趾、Y 指向人体左侧，Z 为叉积；手掌坐标系 X 从手腕指向中指 MCP、Y 从小指指向食指。这些是关键点定义的坐标系，不是完整解剖关节旋转。点不足或接近共线时输出 null。轴向扭转和骨盆与腰部倾斜分离仍不可唯一观测。

地面高度默认为 null，接触估计因此也为 null。只有显式提供标定的地面高度后，才根据同一个足部点的离地高度及速度产生静止接触评分。它不是经过标注校准的概率，也不是接触力测量。

## 已测量的改善

同一 96 帧、384 张 RGB；旧模型和新配置均关闭性能分析后测量。精确数值及实验选择记录见 [验收报告](../reports/vision-optimization-2026-10-17.json)。

| 指标 | 原始 RTMPose17 | 新人体状态 |
| --- | ---: | ---: |
| 身体代理点平均误差 | 60.87 mm | 47.63 mm |
| 身体代理点 P95 误差 | 131.54 mm | 113.39 mm |
| 右小腿长度变异系数 | 20.77% | 1.32% |
| 身体 12 点可用覆盖 | 99.39% 原始观测 | 99.48% 含短时预测 |

新状态中身体 12 点共 1152 个时刻：1106 个 filtered、40 个 predicted、6 个 missing。平滑与预测不能被当成新增的直接观测。上述误差使用风格化模型骨骼中心代理，包含关节定义偏差，不代表实拍精度；足部和掌部朝向还没有真值验收。

新配置图像处理平均约 82.1 ms/四路组，旧配置约 92.2 ms；新配置额外输出足掌点。新三维重建约 26.8 ms/帧，状态估计约 4.6 ms/帧，完整离线视觉处理约 113.4 ms/组，即 8.82 组/秒。**24 组/秒目标尚未达到**，也未测量真实曝光到机器人执行延迟。

GPU 算子证据单独采集，不计入上述测速。检测网络、RTMW 和 RTMPose 均实际运行 CUDA；仅检测框筛选与 NMS、低维几何和状态估计在 CPU 上。相同图像的检测框对比验证了 NMS 优化，且显式仅接受 person 类，避免把 COCO 其他类别当作人。

未采用的实验：轻量检测器漏检更多；直接 RTMW 身体输出的髋部连续性不足；当前 ORT 转换得到的 FP16 模型更慢；身体网络合批会在少量歧义点切换 SimCC 峰值和三角化相机对，导致 G1 参考关节出现约 24.5° 差异。因此当前仅 RTMW 足掌网络合批，RTMPose 身体逐图推理。最终身体三维坐标与基线最大差异约 0.00014 mm，公共准入帧的 G1 参考关节最大差异约 0.0000005 rad。对应摘要保留在报告，临时模型和重复输出清理。

## G1 接口与下一步

新的 `robot-demo` 生成 33 点视觉观测和独立人体状态。G1 默认重定向仍使用有当前图像来源的 `poses3d` 身体点及原有滤波；当前样本准入仍为 86/96 帧。没有把预测点自动伪装成新观测，也没有把新的足掌朝向直接接入电机控制。显式读取状态文件时检查来源哈希；现有门控对状态文件的阈值尚需独立校准。

下一步优先获取多动作和实拍同步数据，评估人体尺度初始化、髋部和交叉腿错误、遮挡恢复与动态延迟，再验证基于状态估计的控制接口。多人身份关联、学习式跨视角特征融合、闭环在线采集调度和 24 FPS 实时运行仍是未完成项。现有全身机器人路径跟踪验收仍未通过，本轮不改变这一结论。

## 第二轮联合估计与在线调度

第二轮结果见 [验收报告](../reports/vision-motion-2026-10-17.json)。新算法作为可选择的实验路径保留；旧 `vision-demo` 与 G1 默认原始观测路径继续作为兼容基线。新模块均不读取真值。`benchmark_data` 负责独立生成测试观测和标签，`motion_evaluation` 在推理结束后才读取标签。

### 模块接口与科学边界

| 模块 | 输入 / 输出 | 失败和不可观测情况 |
| --- | --- | --- |
| `multiview_temporal.MultiViewTemporal` | 标定、带曝光时间的具名二维点、上一人体状态 → 当前多视角三维观测和分配后的二维记录 | 左右臂/腿分别枚举视角标签；对称极线误差与历史投影共同评估。只在几何改善和假设间隔足够时改标签；歧义局部拒绝。全视角共同互换无法由几何确定，标为歧义；不补伪相机证据 |
| `body_shape.BodyShape` | 当前高质量、至少三视角观测 → 可修正骨长、样本量、跨度、姿态变化、描述性误差尺度 | 使用滚动稳健统计，需时间和方向变化才参与约束；尺寸学习不受旧尺寸的拒绝门控控制。共同系统偏差仍不可识别，结果不是实测人体尺寸 |
| `human_motion.HumanMotionEstimator` | 当前三维观测 → `human_state_v2` | 延续 100 ms 预测上限、来源切换重置。增加根部、局部速度拟合、足掌/躯干标志点旋转、可观察的膝肘屈曲和接触滞回；轴向扭转、骨盆与腰部独立旋转、接触力仍未知 |
| `motion_evaluation` 的校准函数 | 仅校准集标签和估计 → 域及代码配置绑定的经验误差球 | 共享生成器和人物造成样本相关，不能声称严格覆盖保证。域或实现不匹配时不提供校准半径；合成校准绝不直接用于 RGB/实机 |
| `online.ExposureSynchronizer` / `LatestMailbox` | SDK 映射到共同单调时钟的曝光帧 → 有界同步组/最新待处理组 | 拒绝时钟混用、逆序、重复、过期和过大视角时间差；不复用帧，待处理队列只有一组 |
| `streaming_vision.StreamingVision` | BGR 图像组 → 因果人体状态 | 真实 CUDA 网络推理；身体网络仍 batch 1；可选择 4 个并发检测调用。多人输入继续拒绝，未实现人物身份关联 |
| `perception_gate.PerceptionGate` | 人体状态和当前共同时间 → follow / transition / stop 请求 | 只允许当前、不过期、具有本域误差校准的腿部状态启动新动作；连续 3 帧恢复。控制器须持续轮询无数据看门狗。此模块不执行机器人命令，也不证明平衡或轨迹连续性 |

单位：位置/骨长米、线速度米每秒、角度弧度、角速度弧度每秒、时间整数纳秒。坐标为标定世界坐标，旋转矩阵为世界从局部标志点系的旋转。`orientation_quality.observable` 表示局部几何和误差尺度检查，未校准旋转覆盖概率。局部速度来自至多 4 帧因果拟合，可能有相位偏差；缺少速度窗口时明确标为模型预测。接触是基于地面高度和足点速度的运动学候选，不等于物理支撑力。

### 评测拆分及复现

`config/vision_benchmark.json` 固定 7 种人体比例、6 类动作、42 段、3780 组四视角观测。调参 1 人物配置 × 6 动作；校准 4 × 6；测试 2 × 6。人物配置不跨集合。数据源是程序化运动投影加噪声、局部遮挡及标签扰动，**不经过 RGB 检测器，也不是真人精度证据**。校准与测试参考文件分开，生成清单记录哈希，修改输入会报错。事件时刻仅评测读取。

```powershell
$rt = Get-Content .runtime.json -Raw | ConvertFrom-Json
& $rt.python scripts/benchmark_motion.py --dataset outputs/my_benchmark/data --generate --output outputs/my_benchmark/tune --split tune --methods baseline joint no_association with_structure
& $rt.python scripts/benchmark_motion.py --dataset outputs/my_benchmark/data --output outputs/my_benchmark/calibration --split calibration --methods joint
& $rt.python scripts/benchmark_motion.py --dataset outputs/my_benchmark/data --output outputs/my_benchmark/test --split test --calibration outputs/my_benchmark/calibration/uncertainty_calibration.json
```

指标包含世界位置误差及全目标 100 mm 覆盖、足掌旋转误差、躯干旋转（单列）、速度误差、骨长误差、遮挡结束时额外跳变、真值接触时的足滑、接触混淆/未知比例、离散相位滞后和误差球覆盖。相位分辨率为一帧约 33 ms，0 帧不等于零延迟。接触标签在该生成器中也是高度/速度定义的运动学参考，不能当作力传感器真值。没有正例也没有预测正例的序列，接触 F1 留空；汇总使用混淆矩阵而非把空指标算零。

### 在线回放与 G1 接入边界

```powershell
.\run.ps1 -Stage stream-vision -InputSession outputs/human_walk_g1 -Session outputs/my_online -DetectorWorkers 4
.\run.ps1 -Stage refine-motion -InputSession outputs/human_walk_g1 -Detections outputs/vision_optimized/detections.jsonl -Session outputs/my_refined
```

在线回放按源帧率在主机单调时钟上安排测试曝光时间，实际运行解码、GPU 推理、几何和状态处理。记录真实墙钟处理时间、排队年龄、丢弃数及曝光代理到输出时间；超过配置的输出期限不能成为新动作。此测量**不含真实传感器曝光、驱动或设备时钟误差**。实机必须使用已有 `ClockMapping` 与实际 SDK 的曝光时间，不能将接收时间冒充曝光时间。

G1 读取 `human_state_v2` 时沿用当前观测检查，预测点不能绕过准入；状态文件必须绑定原始观测哈希。人体状态到 G1 的新增尺寸、足掌旋转、支撑候选尚未完成动力学接入验证，默认控制仍不改。实拍域没有误差校准时 `PerceptionGate` 不会进入 follow；这属于明确未完成的验收项，不能通过放宽阈值伪装成功。

### 真实 RGB 挑战样例

`scripts/import_mamma_example.py` 读取 MAMMA 官方四相机样例和固定提交的标定，显式将相机世界位姿求逆，并同步缩放内参和图像。只解析该已核查标定的数值数组布局，不通用执行外部 YAML。示例帧索引对齐来自提供方演示约定，未验证硬件同步；无随附三维真值，仅作输入和拒绝行为检查。

```powershell
& $rt.python scripts/import_mamma_example.py --cache outputs/public_mamma --output outputs/my_public_case
& $rt.python scripts/vision_demo.py outputs/my_public_case --output outputs/my_public_audit
```

本地样例限非商业科研用途，来源/许可和 SHA256 保存在忽略目录。多人场景拒绝是当前能力限制，不能把零错误接入率当作成功动捕。仍需覆盖完整动作、不同真人、标注语义和实际相机误差的定量独立测试集。

### 第二轮实测结果与发布决定

| 检查 | 旧方法 | 实验方法 | 结论 |
| --- | ---: | ---: | --- |
| 独立程序化观测测试，平均位置误差 | 44.66 mm | 13.97 mm | 此扰动分布上改善，非 RGB 精度 |
| 同测试平均速度误差 | 0.583 m/s | 约 0.454 m/s | 改善，速度不确定性仍未校准 |
| 同测试接触 micro-F1 | 0.901 | 0.921 | 仅运动学接触参考，非真实支撑力 |
| 原 RGB 行走平均代理误差 | 47.63 mm | 49.96 mm | 回归，不推广为默认 |
| 原 RGB 的 G1 参考准入 | 86/96 | 77/96 | 未改善控制准入 |
| 定时 RGB 回放平均处理时间 | 串行 113.91 ms | 并发检测 101.27 ms | 仍不满足 24 Hz |
| 回放曝光代理到输出 P95 | 177.76 ms | 163.66 ms | 真实相机曝光延迟未测 |

并发检测对同一 96 帧的关键点和分数逐点差异为零；仅并发检测，身体网络仍逐图推理。最终 150 ms 输出期限下，并发运行处理 40 组，替换 56 组旧数据，14 组输出超时；有效及时结果只有 26 组。队列有界不会自动提升算法吞吐量。

经验误差球在 12 条测试序列中完全覆盖 11 条，但半径较保守；这不是“91.7% 的真人可靠性”。同一动作生成器和关联序列不满足真实样本的交换性条件。足滑与足掌旋转的改进很有限，尚不能认为相关科学问题已解决。

最终回归：机器人环境 92/92 通过；视觉环境 92 项中 86 通过、6 项机器人专用跳过。一次视觉测试进程未产生完成摘要，后续完整详细运行通过，原退出原因未确定。状态接口、时间、左右标签歧义、尺寸自锁恢复和预测不能绕过 G1 门控均有回归覆盖。

自动审批拒绝了本轮被替代实验目录的清理（仅返回 `blocked by policy`），因此没有删除。待清理目录和大小记录于 `outputs/motion_benchmark/cleanup_pending.json`；当前验收产物与基线已分别列明，未提交或推送。

## 第三轮：无真实相机数据条件下的修正

本轮使用已保存的 CUDA RGB 检测、独立程序化扰动以及固定 G1 仿真控制器，不新增真人数据。完整结果见 [第三轮报告](../reports/vision-revision-2026-10-17.json)。第二轮的整肢体交换及速度覆盖逻辑被以下实现替代；旧 `vision-demo` / G1 默认原始观测入口仍保留为兼容路径。

### 已实施的变化与边界

- **按关节对关联**：膝、踝、髋以及肩、肘、腕独立提出左右候选，防止一次膝部错误把正确的髋和踝一起交换。只有几何和历史投影同时支持才换标签；共同的全视角语义歧义只拒绝受影响关节对。输出前三个候选及相对代价，代价不是校准概率，也尚未实现多候选轨迹的长期联合滤波。人体结构仍通过独立状态模块参与约束。
- **尺寸软约束**：至少 6 个合格样本后可以使用暂定尺寸，但以较宽误差尺度施加软约束；未成熟尺寸不能硬拒绝观测。稳定尺寸仍可从独立的原始多视角证据修正。尺寸与共同检测偏差尚不能完全辨识。
- **速度与状态一致**：主输出速度保持在联合位置/速度滤波器中，暴露其未校准协方差，单位 m²/s²。局部拟合速度仅保留为诊断，不再覆盖滤波器速度而沿用旧协方差。朝向丢失后清除角速度历史，避免跨缺失段直接差分。
- **CPU 几何优化**：针孔归一化批量计算，相机子集矩阵跨帧复用；三维鲁棒重投影采用小规模 soft-L1 求解和单调线搜索，未收敛回退 SciPy。原始几何入口默认仍使用 SciPy，新融合入口采用快速求解器。保持同一残差、置信权重、内点规则和坐标系。复用的几何缓存只属于固定标定的融合实例。
- **GPU 调度**：可同时提交四个独立 batch-1 身体网络调用，并与足掌网络调度重叠；不改变已验证身体模型的批大小。RTMLib 有状态的预处理串行执行，只有无状态的 ORT 推理并发。相机检测并发独立设置。
- **长期运行内存**：采集队列仍只保留最新一组；时延和阶段计时统计最多保留最近 4096 组，总处理/丢弃计数不截断。超过窗口时报告 `metrics_window.truncated`，不会将窗口分位数冒充全部历史分位数。

位置单位 m、旋转矩阵和世界坐标约定、时间纳秒及过期规则沿用前文。新增候选和速度协方差均未取得真实域概率校准，不能直接授权硬件动作。

### 复现和更严格的比较

```powershell
$rt = Get-Content .runtime.json -Raw | ConvertFrom-Json
# 固定原二维观测做四组消融，并保存 joint 路径供下游参考检查
& $rt.python scripts/diagnose_vision_regression.py outputs/human_walk_g1 --baseline outputs/vision_optimized --output outputs/my_revision_rgb
# 在完全相同观测上比较快速几何求解与 SciPy
& $rt.python scripts/check_geometry_parity.py outputs/human_walk_g1 --detections outputs/my_revision_rgb/assigned_detections.jsonl --output outputs/my_geometry_check.json
# 检测和身体推理并发，身体网络每次仍只有一张图
& $rt.python scripts/check_stream_parity.py outputs/human_walk_g1 --pose-workers 4 --output outputs/my_gpu_check.json
.\run.ps1 -Stage stream-vision -Session outputs/my_revision_stream -DetectorWorkers 4 -PoseWorkers 4
```

新增 `compare_methods` 同时计算共同有效集合、共同当前观测集合、全部目标分母下的正确率、缺失和预测数量，旋转也仅在共同有效集合比较。支持原 COCO 代理参考中省略 `joint_names` 的历史格式和空目标。所有误差在标定世界坐标计算，不通过刚性对齐消除根部误差。

`config/vision_benchmark_randomized.json` 固定另一组 7 种人体比例、6 动作、42 段、5040 帧。随机化相机方位、噪声、初始化偏差持续时间、交换的关节/视角、遮挡位置和持续时间，并注入确实对应上一帧图像的过期相机观测。4 种比例的 24 段用于校准，2 种未见比例的 12 段用于测试。它仍共享程序化运动生成器，不能代表不同真人或真实 RGB 泛化。

```powershell
& $rt.python scripts/benchmark_motion.py --dataset outputs/my_random_data --generate --config config/vision_benchmark_randomized.json --output outputs/my_random_tune --split tune --methods baseline joint no_association
& $rt.python scripts/benchmark_motion.py --dataset outputs/my_random_data --output outputs/my_random_calibration --split calibration --methods joint
& $rt.python scripts/benchmark_motion.py --dataset outputs/my_random_data --output outputs/my_random_test --split test --methods baseline joint --calibration outputs/my_random_calibration/uncertainty_calibration.json
```

基准默认只保存报告和校准文件，明确要求逐帧复查时再加 `--save-states`，避免每个实验复制大量状态文件。`outputs/motion_revision/freeze.json` 记录了随机测试前冻结的实现指纹。原程序化测试的复跑只作回归，不能当作第二份独立证据。

本轮随机测试平均位置误差 24.72 → 14.81 mm，共同有效点上也改善；速度误差 0.545 → 0.428 m/s，接触 micro-F1 0.878 → 0.916。足掌旋转与脚滑基本未变，遮挡恢复跳变改善很小。经验误差球平均半径约 0.112 m，12 段中 11 段全覆盖：这说明尾部风险导致范围较保守，不代表真实场景的覆盖保证。实拍域的动作门控仍保持关闭。

### 第三轮最终验收与控制回退

| 指标 | 对照 | 本轮 |
| --- | ---: | ---: |
| 渲染 RGB 代理平均误差 | 原 v1 47.63 / 第二轮 v2 49.96 mm | 47.54 mm |
| 同一片段最大代理误差 | 第二轮 291.06 mm | 203.57 mm |
| 身体当前观测点 / 预测点 / 缺失点 | 第二轮 1092 / 59 / 1 | 1138 / 13 / 1 |
| 原程序化测试平均位置误差 | 第二轮 13.97 mm | 13.54 mm（仅复跑回归） |
| CPU 重建平均处理时间 | SciPy 25.92 ms | 17.86 ms |
| 最终在线回放平均处理时间 | 身体串行约 96.3 ms | 身体并发 85.51 ms |
| 最终并发回放曝光代理至输出 P95 | — | 149.16 ms |
| G1 参考准入 | 默认原始点 86/96；v1 状态 77/96 | v2 状态 90/96 |
| 固定控制器仿真根部路径 P95 | 原始点路径 0.773 m | v2 状态路径 1.379 m，变差 |

快速几何求解在 3127 个相同观测点上最大差异约 0.0091 mm，内点与覆盖一致，测试片段无回退；噪声、畸变、异常视角和低视差另有回归。四路并发身体推理在 96 帧上的二维点和分数差异为零。

最终并发回放处理 48 组，其中 46 组在 150 ms 输出期限内，状态吞吐约 11.47 组/秒，仍未达到 24 Hz。由于真实/RGB 域没有误差校准，在线 `PerceptionGate` 的 follow 数仍为零。

G1 对照保持同一机器人模型、跟踪策略和 120 ms 注入延迟，只更换视觉输入和相应参考生成路径。新状态的平衡与停止检查通过，腿部模仿和根部路径检查未通过；当前状态虽然更准确，也不能视为控制策略已经适配。新增诊断中，直接使用新原始三维点的根部路径 P95 为 1.113 m；对新状态补加原有重定向平滑后为 1.130 m，均未通过。参考准入数增加不能代替动力学验收。仿真使用离线参考门控，不是已校准的在线状态到实机闭环。

最终测试：机器人环境 101/101 通过；视觉环境 101 项中 95 通过、6 项机器人专用跳过。绘图见 `outputs/motion_revision/summary.png`。本轮改进保留在视觉实验路径，不替换 G1 默认输入。

本轮已保存简短开发诊断，拟删除的三个替代产物目录被自动审批拒绝（仅返回 `blocked by policy`），共 25,816,857 字节，实际删除为零。路径记录在 `outputs/motion_revision/cleanup_current.json`；此前被拒绝的旧目录仍以原待清理清单记录。未提交或推送。

### 多场景姿态估计视频

`config/vision_showcase.json` 定义 6 段各 6 秒、24 fps 的四相机片段：直行、连续转向、局部遮挡、弱光、统一缩放至 1.50 m 的人物、加快步速。使用同一个 CesiumMan 模型与步行动画，改变根部路径、动画速度、尺寸和环境；不能当作多人物、多动作真实评测集。

```powershell
$rt = Get-Content .runtime.json | ConvertFrom-Json
& $rt.python scripts/vision_showcase.py --output outputs/vision_showcase
```

支持 `--stage render|infer|compose` 分阶段复现。Blender / OptiX 生成 RGB，CUDA 检测器逐帧处理全部四路图像，CPU 完成多视角与因果时序估计。每次场景切换重新初始化关联、体型与历史状态。输入时间为独立片段的 `synthetic_exposure` 纳秒时间，几何为米、世界 Z 轴向上。渲染真值另存用于核对相机投影，检测、重建和视频画面均不读取真值文件。

输出视频在原始 24 fps 下离线播放，不代表在线吞吐。左侧显示原始二维检测，右侧用两个固定正交投影显示根部居中的估计骨架；没有额外插帧或展示专用平滑。橙色 / 蓝色是模型输出的左右标签，紫色为历史预测，缺失关节留空。足掌与手掌的完整旋转、实际支撑和实机执行均未获此视频验证。

逐帧记录在 `sources/` 与 `estimates/`，最终视频为 `pose_estimation_36s.mp4`。`video_validation.json` 记录编码、帧数、时长与时间对齐检查；六个场景的中间帧保存在预览图和联系表中。输出在忽略目录内，不复制模型或运行环境。渲染投影校验失败、RGB 读取失败、时间戳错位、缺帧或编码数量不符时显式报错，不通过截短、补帧或真值替换隐藏失败。

### 第四轮：独立细节关联与附着一致性（2026-10-17）

本轮接受手脚一致性模块，拒绝将多候选时序关联设为默认。简短验收报告为 `reports/vision-attachment-2026-10-17.json`。当前结果在 `outputs/vision_attachment/accepted/`；更新后的六场景视频仍为 36 秒、864 帧。复用原四路 RGB，没有重渲染、引入真值或额外画面平滑。

模块契约：

| 模块 | 输入与输出 | 失败行为与验证 |
| --- | --- | --- |
| `auxiliary.associate_details` | 具名像素点、分数、原曝光纳秒时间；输出独立手/脚整组置换及来源索引，身体点不变 | 两侧身体锚点可靠、分离充分、置换有明确距离收益才纠正；交叉/缺失时不强行决定。脚组需 RTMW 原脚踝锚点，旧记录缺少时跳过脚组置换。 |
| `auxiliary.filter_measurements` | 标定世界坐标中的米制点、几何质量；输出副本和逐组附着诊断 | 用前臂/小腿当前尺度和不确定性放宽的上界检查手腕、MCP、足点；拒绝明显脱节的观测，不拉回、不修改原始三维记录。身体锚点不可靠时保留未知。 |
| `HumanMotionEstimator` 附着检查 | 因果滤波后的当前观测；增加 `attachment_measurements`、`attachment_state` | 再查独立滤波造成的脱节；缺少身体依据时不输出相应旋转/支撑。被拒观测不刷新观测时间，历史预测仍最多 100 ms。 |
| `AssociationHistory`（实验） | 每视角左右关节像素轨迹、曝光时间；保留最多 4 条候选、最长 150 ms | 相机当前几何仍是必要依据；历史不是额外相机、分数不是后验概率。换源、模型、标定或时间断层清空；全局左右对称保持未知。RGB 退化，默认关闭。 |

默认参数是宽松的合理性界限，不是学习得到的精确人体尺寸，也不能保证语义身份正确。未知量不被补成可执行自由度。当前 `human_state_v2` 增加诊断字段，既有坐标系、单位与时间基准保持不变。`config/vision_motion.json` 中 `association_history=false`；在线路径读取 `auxiliary_association` 与 `state.auxiliary_consistency`。算法级消融使用 `MultiViewTemporal(auxiliary=False, history=False)` 和 `HumanMotionEstimator({'auxiliary_consistency':False})`。

六场景共重新运行 3456 张 RGB 的 CUDA 检测，原 33 点坐标、分数与基线差异均为零；新增保存了 RTMW 原脚踝锚点。接受方案的身体 12 点轨迹与基线一致，代理平均误差均为 47.055 mm。以下距离只反映一致性，不是手脚真值误差：

| 六场景指标 | 第三轮基线 | 本轮默认 |
| --- | ---: | ---: |
| 当前手腕—掌根间距 P95 | 154.46 mm | 90.82 mm |
| 当前手腕—掌根间距超过 250 mm | 45 / 1538 | 2 / 1455 |
| 当前细节点覆盖率 | 97.38% | 92.19% |
| 含历史预测的细节点覆盖率 | 99.86% | 98.21% |
| 可输出足掌/手掌框架数 | 2972 / 3456 | 2780 / 3456 |
| 身体 12 点全为当前观测的帧数 | 680 / 864 | 680 / 864 |

保守拒绝会降低细节可用率，不能称为已经恢复遮挡。少量较大偏差仍可在身体锚点高不确定性时通过宽松界限；RGB 没有足掌/手掌真值，不能据此证明细节旋转准确。

新增 `config/vision_benchmark_attachment.json`：独立开发人物 6 段、2 种未见比例 12 段测试，每段 120 帧，包含六种程序化动作。除原有噪声/遮挡外，注入独立手脚标签交换和跨全部相机一致的错误三维簇。后者可通过重投影检查，专门检验结构约束。测试前冻结核心实现和参数指纹，真值只进入评测；这仍是投影观测压力集，不是 RGB 或真实人物泛化。

| 留出集指标 | 第三轮 `legacy_joint` | 本轮 `joint` |
| --- | ---: | ---: |
| 新细节压力集：共同有效点误差 | 41.86 mm | 17.61 mm |
| 新细节压力集：共同当前观测误差 | 25.17 mm | 15.13 mm |
| 新细节压力集：全部目标中误差 ≤100 mm | 92.07% | 96.78% |
| 新细节压力集：可用点覆盖率 | 99.92% | 97.55% |
| 新细节压力集：共同旋转误差 | 9.35° | 7.50° |
| 原随机测试集：平均序列误差 | 14.81 mm | 14.84 mm |
| 原随机测试集：当前且误差 ≤100 mm 的全部目标比例 | 99.54% | 99.43% |

新压力集全部可用点的均值为 58.74 → 17.60 mm，但集合不同，必须与共同集合及全部目标分母一起阅读。遮挡恢复额外跳变为 11.27 → 11.75 mm，未改善，统计样本也从 454 降至 384。脚滑统计约 0.646 → 0.272 m/s，但样本数 1056 → 992，不是相同样本比较，不作为已解决脚滑的证据。相位延迟逐序列保存在报告中；本轮没有解决长期遮挡、可观测性或真实域不确定性校准。

多候选实验在 RGB 共同点均值 47.05 → 47.60 mm、最大值 473 → 723 mm，故不晋升默认。开发对照与最差片段状态保存在 `rgb_comparison_initial.json` 和 `rejected_history_evidence.json`。保留实验入口供后续改进，不因单个单元测试能恢复交换就宣布总体更优。

RTX 4090、四路检测及身体 batch-1 并发的顺序短回放：平均处理 84.80 → 88.35 ms，状态率 11.56 → 11.11 Hz，曝光代理至输出 P95 146.04 → 155.49 ms。队列容量仍为 1，超过 150 ms 的结果不能成为新动作；本轮 46 次输出中 41 次及时。两个短回放的调度与处理帧不同，不能解释为严格的算法耗时统计。RGB 校准无效，`planner_follow_count=0`；没有更改 G1 默认控制，也没有新增动力学或实机验收。

```powershell
$rt = Get-Content .runtime.json | ConvertFrom-Json
& $rt.python -m unittest discover -s tests
& $rt.python scripts/benchmark_motion.py --dataset outputs/my_detail_data --generate --config config/vision_benchmark_attachment.json --output outputs/my_detail_tune --split tune --methods legacy_joint joint with_history
& $rt.python scripts/benchmark_motion.py --dataset outputs/my_detail_data --output outputs/my_detail_test --split test --methods legacy_joint joint
& $rt.python scripts/refine_motion.py outputs/vision_showcase/sources/01_straight --detections outputs/vision_attachment/rgb/estimates/01_straight/detections.jsonl --output outputs/my_detail_replay
& $rt.python scripts/stream_vision.py outputs/human_walk_g1 --output outputs/my_detail_stream --detector-workers 4 --pose-workers 4
# 对照耗时增加 --legacy-details；实验关联回放增加 --history（refine_motion）。
```

最终视觉环境 114 项中 108 通过、6 项机器人专用跳过；机器人环境 114/114 通过。视频已逐帧验证时间、数量和全片解码。模型、环境和 RGB 未复制。拟清理本轮被替代的 18 个派生 JSONL，共 82,941,845 字节；路径及 SHA256 均先核验，但自动审批拒绝执行，仅返回 `blocked by policy`，实际删除为零。清单在 `outputs/vision_attachment/cleanup_generated.json`；未重试此前被拒绝的旧清理项，未提交或推送。

### 恢复事件评测与轨迹确认（2026-10-18）

人体状态新增 `track_confirmation_frames`（默认 1）：新建或过期重建的关节先输出 `tentative`，连续接受 N 帧后才是 `filtered`；未确认关节不进入骨长、朝向和接触估计。下游门控只认 `filtered` 且当前曝光有观测的关节，所以 `tentative` 不会成为新动作。开发集上 N=3 未通过覆盖率与误接受门槛，默认不变，详见[调研与结果](observation_recovery_research_2026-10-18.md#第一轮实施恢复事件评测与轨迹确认2026-10-18)。

[故障注入](../pose_sim/fault_injection.py)只改缓存的二维检测，带事件标签；[事件评测](../scripts/benchmark_recovery_events.py)每种方法配对干净/注入两次运行，报告误接受、可用性损失、恢复耗时、身份互换、控制可见跳变和相位延迟。MPI-INF-3DHP 的 50 Hz 序列（S3/S5）导入时按步长 2 降到 25 Hz，时间戳仍取源帧时钟。

```powershell
$rt = Get-Content .runtime.json | ConvertFrom-Json
& $rt.python scripts/benchmark_recovery_events.py --dataset outputs/public_datasets/mpi_long --cache outputs/public_datasets/mpi_long_detections --output outputs/recovery_events/my_dev --split development
# 冻结测试集：导入、缓存检测，再用冻结哈希评测一次
& $rt.python scripts/import_mpi3dhp.py --output outputs/public_datasets/mpi_test --config config/public_mpi3dhp_test.json
& $rt.python scripts/benchmark_recovery.py cache --dataset outputs/public_datasets/mpi_test --output outputs/public_datasets/mpi_test_detections
& $rt.python scripts/benchmark_recovery_events.py --dataset outputs/public_datasets/mpi_test --cache outputs/public_datasets/mpi_test_detections --output outputs/recovery_events/my_test --split test --methods default confirm3 --expected-hash <冻结哈希>
```
