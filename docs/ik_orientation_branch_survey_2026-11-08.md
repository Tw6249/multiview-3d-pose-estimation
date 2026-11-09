# IK 朝向约束与分支选择调研：2026-11-08

## 结论

**有成熟可借鉴的解法，不是无解；但没有找到可直接替换本项目视觉→G1 全身链路的“一键分支修复”。** 文献把 7-DoF 手臂冗余明确参数化成 elbow swivel / SEW angle，使用人的肘部构型或学习到的人体构型先验选分支；另一类工作做带关节位置/速度限制与跨帧连续性代价的轨迹优化。部分方向/朝向约束文档允许只固定可观测旋转轴，把轴向自旋释放出来。

本项目在同一 G1+SONIC 闭环基准上的首轮试验也给出直接证据：泛化的多初值重启会选到更坏的手臂分支；将骨盆俯仰轴降权后只有个别腿部和倾角指标改善，路径/手臂等指标不一致。因此，下一步不该把每帧“最低残差 seed”投入默认，而应先把分支显式写成可测的肢体构型参数，并让不确定观测控制对应任务权重。

## 研究方法分类

| 方法 | 文献/项目做法 | 对本项目的适用性 |
|---|---|---|
| Swivel / SEW 角显式分支变量 | 7-DoF 冗余手臂的解析 IK 将肩-肘-腕配置由全局构型参数和 arm angle/SEW angle 表示；改变 SEW angle 可在末端相同或近似不变时沿自运动流形移动肘部。 | 最接近 G1 手臂的几何问题。源人的肩、肘、腕三点能给出手臂平面/肘位；当肘伸直或点置信度低时，swivel 本身不可观，应保持上一可信值或退回弱姿态先验。 |
| 肘平面匹配的全身 IK | WARP 用闭式 SEW 几何求解器匹配末端位姿，并把 SEW swivel 与躯干/头部朝向等目标放入全身重定向目标。 | 这是目前检索到最直接针对“腕点正确但肘走错分支”的全身重定向方法。论文处理离线人体示范/移动操作；必须在本项目实时相机关键点、G1 关节拓扑和下肢任务下重做置信度、奇异点与留出集验证。 |
| 闭式 SEW-Mimic 几何重定向 | 将肩、肘、腕的方向对齐分解成几何旋转约束，按肩到肘再到腕逐段求解，并显式匹配 SEW swivel。 | 提供可独立于全身 IK 验证的上肢候选，且有公开 Python/C++ 实现；需检查仓库支持的机器人拓扑、左右坐标和关节限位，不能假设直接兼容 G1。 |
| 在线观测肘点约束 | SEW-Mimic 一类方法把观测到的肩、肘、腕三点用于构造 SEW swivel/肘平面约束，以解析几何或零空间 IK 选择人类相似的肘构型。 | 可作为低成本上肢参照，机制比多初值搜索更直接；公开 Python 实现所列机器人不包含 G1，需适配并验证目标 G1 的平行腕/关节拓扑，不能拿来证明骨盆/腿部重定向有效。 |
| 肘构型先验 | HL-IK 将历史人体 retarget 数据训练出的肘部构型预测作为低权重残差，与末端跟踪和平滑项一起优化。 | 可当研发方向，不应直接照搬：项目要求当前无训练任务，且本项目有真人多视角肘点，可先用几何 swivel 观测验证；HL-IK 的公开检索结果未确认可直接部署的通用 checkpoint/代码。 |
| 跨帧轨迹优化 | TORM 在冗余机械臂路径优化中显式加入构型连续性；SPARK 将 IK 输出送入 kinodynamic trajectory optimization；HOI-Retarget 对短时间窗联合优化跟踪、接触、支撑与平滑。 | 可作为轨迹质量上限和短窗对照。完整窗口需要未来帧，加入在线延迟；先以因果 q/dq 连续性代价或小窗已知延迟实验，不冒称零延迟。 |
| 任务优先级/差分 IK | GMR/Mink/Pink 风格每步解局部 QP，给任务不同权重并约束关节限位。它们不自动解决多个局部解间的全局 branch identity；从上一帧 warm-start 只是隐式延续。 | 与当前 S45c 最接近。需要显式记分和分支迟滞，并报告约束冲突；不可只调高某个姿态 cost。 |
| 部分朝向约束 | CoppeliaSim IK 文档提供 full / 2-axis / 1-axis / no orientation constraints；Drake 的操作空间规划例子在圆柱抓取中释放绕对称轴的旋转。 | 支持“只约束被观测到的轴”。骨段方向只能测到 $S^2$ 的两个自由度，不能把绕骨段的 twist 当作测量。COCO 手部 MCP 不稳定时不应用掌朝向约束。 |
| 全局/多初值搜索 | IKSel 等工作讨论为数值 IK 选择质量更好的种子；解析全局 IK 常枚举多个解，再由构型/路径代价选解。 | 通用随机重启不是本项目分支解法。刚做的本地负结果显示，若代价没有匹配人类肢体构型，重启只会更快找到错误的局部解。 |

## 来源

- Araujo et al., **HL-IK: A Lightweight Implementation of Human-Like Inverse Kinematics in Humanoid Arms**, arXiv 2509.20263。搜索来源：[arXiv](https://arxiv.org/abs/2509.20263)、[项目 PDF](https://hl-ik.github.io/static/pdf/HL-IK.pdf)。摘要/项目资料描述学习肘部先验与 LM IK；本次没有找到确认可复用的官方推理仓库或本项目可用权重。
- **WARP: Whole-Body Retargeting for Learning from Offline Human Demonstrations**, arXiv 2606.29940。[论文](https://arxiv.org/html/2606.29940v1)、[项目](https://warp-retargeting.github.io/)、[代码](https://github.com/Euler-Rodrigues-Lab/WARP_retargeting)。使用闭式 SEW solver 并在 whole-body 目标中匹配 swivel；与本项目观测到的肩肘腕点最匹配，但论文针对离线示范/移动操作，不等于因果相机系统验证。
- **A Closed-Form Geometric Retargeting Solver for Upper Body Humanoid Robot Teleoperation (SEW-Mimic)**, arXiv 2602.01632。[论文](https://arxiv.org/abs/2602.01632)、[Python/C++ 实现](https://github.com/chase6305/SEW-mimic-retargeting)。适合作为上肢几何映射的独立候选；报告其论文结果前仍需按 G1 运动学链与本项目数据复现。
- **A Closed-Form Inverse Kinematic Analytical Method for a Humanoid Seven-DOF Redundant Manipulator**, *Machines* 14(4), 395 (2026)。[论文](https://www.mdpi.com/2075-1702/14/4/395)。用全局构型变量描述 IK branch；检索文本还指出配置连续性仍是后续工作，故不能把论文误说成已有完整迟滞解法。
- **Redundancy Parameterization and Inverse Kinematics of 7-DOF Revolute Manipulators**, arXiv 2307.13122。[论文](https://arxiv.org/pdf/2307.13122)。讨论 7-DoF 冗余参数化和 arm-angle 分支。
- **TORM: Fast and Accurate Trajectory Optimization of Redundant Manipulator given an End-Effector Path**, arXiv 1909.12517。[论文](https://arxiv.org/pdf/1909.12517)。将冗余解选择放到轨迹级优化，加入配置连续性。
- **Constrained Whole-Body Tracking for Humanoid Robots**, arXiv 2606.00374。[论文](https://arxiv.org/abs/2606.00374)、[项目仓库](https://github.com/talos-robotics-ai/constrainedmimic)。研究重点是 WBC 安全约束；可借鉴按任务建立约束，不是人体观测的 IK branch 解法。
- **Whole-Body Geometric Retargeting for Humanoid Robots**, arXiv 1909.10080。[论文](https://arxiv.org/abs/1909.10080)。可借鉴几何目标到全身 IK 的职责划分；检索结果未证实其显式分支连续机制。
- Mink differential IK。[文档](https://kevinzakka.github.io/mink/)、[代码](https://github.com/kevinzakka/mink)。每次求局部 QP 关节速度，不是全局多解分支求解器。
- 部分朝向约束参考：[CoppeliaSim IK 文档](https://manual.coppeliarobotics.com/en/solvingIkAndFk.htm)、[Drake manipulation motion planning](https://manipulation.csail.mit.edu/trajectories.html)。

检索边界：搜索结果中有些来源只提供摘要或索引片段；只有上述直接论文/项目链接作为主来源列出，指标性细节若未从原文确认，未用作本地算法验收结论。

## 本项目实验证据

本轮实验使用 S3 `kalman_3x3`、官方 G1 `scene_43dof.xml`、SONIC v1.1、preview=0；共同初始锚点与评分时间不变。默认配置仍是 warm-start 单解和完整朝向代价；分支实验与弱俯仰实验仅通过局部配置覆盖启用。

### 多初值 branch search

候选为 warm-start 加左右肩滚转/肘屈曲的 5 种局部偏置，优化器为相同 Mink DAQP。代价为加权 IK 任务残差、posture 残差和相对 warm-start 的关节位移代价。

首轮五候选搜索只有较小 `ik_branch_switch_penalty=0.02`，没有最小收益迟滞。它在 Take_012 选择右臂 `right_plus` 105 帧、左臂候选 6 帧、共切换 19 次；物理手臂方向误差由 21.08° 恶化至 41.93°，路径终点误差 0.545 m 增至 0.766 m。五段等权物理手臂误差由 17.59° 增至 21.84°，IK p50 由 1.31 ms 增至 4.95 ms。

加 `ik_branch_switch_margin=0.05` 后，留出 Take_006/007/008 从不切分支，结果逐项等于 warm-start，p50 仍约 4.94 ms。Take_012 只切换 2 帧，物理手臂误差 21.14°、路径误差 0.404 m，但仍无跨留出集证据证明收益。结论：当前 seed 集合和残差评分没有解决分支选择问题，且计算成本约乘 3.8；该功能保持关闭。

### 骨盆俯仰轴降权

把 pelvis FrameTask 的局部 y 轴旋转权重乘以 0.1（局部坐标为 Z-up、Y-left），其他任务不变：Take_006 的物理腿方向误差 11.19° 降至 10.41°、最大倾角 18.08° 降至 13.62°，但路径误差 0.511 m 增至 0.659 m，手臂误差略差；Take_012 的腿误差 8.57° 降至 8.41°，但脚滑 0.0872 增至 0.0960 m/s，手臂误差略差。此处仍是群体层面的试验，不足以更改默认权重。

## 下一步技术方案

1. 从可见的肩、肘、腕三点计算**带置信度的有符号 swivel/肘平面角**：以肩-腕为轴，取肘到该轴的正交投影方向作冗余构型观测；投影长度趋近零（手臂接近伸直）时标记为不可观，而不是强行选符号。
2. 机器人臂用显式 elbow-swivel 标量任务/约束（或配置流形参数）求解；其他肩肘腕位置误差仍按观测协方差加权。观测置信度低时将 swivel 任务逐步退回到上一可信值/弱中立先验，并用符号迟滞避免来回切换。
3. 把分支身份作为状态保留，以 **人体肘平面误差 + 手腕/肘观测误差 + 关节限位 + 关节速度/加速度连续性** 联合排序。候选生成应沿解析/数值 self-motion tangent 改 swivel，而不是随意扰动 shoulder-roll 和 elbow 两个角。
4. 姿态任务按可观测自由度设置权重：肩轴和躯干轴可观时跟踪局部方向；绕肢段自转不约束；手掌完整姿态沿用未知；骨盆俯仰权重由几何可观测性与置信度确定。
5. 先做只读诊断：逐帧记录左右臂 swivel、任务残差、候选间差距、关节跳变和 branch switch；在近伸直手臂、跨胸、T-pose、转身、遮挡恢复分别评分。只有独立留出片段方向误差、物理方向误差和连续性同时改善，才考虑启用。

验证：`tests/test_s45c_keypoint_ik.py` **12 passed**；当前扩大回归 **46 passed, 2 xfailed**。仿真闭环不是实机验证。
