# 多相机无标记人体动作 → 宇树 G1 全身模仿

用多台同步相机无标记地观察人的行走、转身、躯干和四肢运动，重建完整人体运动状态，再按人与 G1 的尺寸及肢体比例重定向为 G1 EDU 的全身动作。左右肢体逐一对应，步态来自源的腿部与骨盆运动。项目约定见 [AGENTS.md](AGENTS.md)。

```text
同步 RGB → 人体检测与二维关键点 → 跨视角人物选择与观测校验 → 三维重建
→ 因果人体状态（体型、根部运动、关节、接触、不确定性）→ 全身重定向
→ 轨迹约束与跟踪控制 → 具名 G1 指令（目前只进入模拟接收器）
```

## 当前状态

| 环节 | 状态 | 依据 |
|---|---|---|
| 二维前端 | RTMPose 身体点 + RTMW 手足点，共 33 点；PMPose、RTMPose-X 为未晋升候选 | [视觉模块](docs/vision_pipeline.md) |
| 多视角重建与观测恢复 | 跨视角人物选择已启用；所有视角一致的左右互换是主要残留错误，轨迹确认 N=3 未通过门槛 | [恢复事件验收](reports/recovery-events-2026-10-18.json) |
| 人体状态 | 因果状态估计、可修订尺寸约束、最多 100 ms 预测及过期处理；COCO17 无法观测脚掌/手掌完整朝向 | [联合状态](docs/vision_joint_state.md) |
| 真实数据评测 | MPI-INF-3DHP 开发集、留出集及冻结测试集（S3/S5/S7/S8，6,000 帧） | [公开数据](docs/public_real_datasets_2026-10-17.md) |
| 在线性能 | 文件回放约 11.3 Hz，未达到 24 Hz；尚无物理曝光延迟测量 | [联合状态报告](reports/vision-joint-state-2026-10-18.json) |
| G1 控制 | 29 自由度模型 + TWIST2 跟踪，仅仿真；已知 10 段完整模仿 0/10，骨盆路径跟踪未达标 | [G1 模仿](docs/source_robot_to_g1.md) |
| 实机 | 未联调：DDS/CRC、反馈与里程计、机型核对、独立停止机制均未接入 | [架构与硬件迁移](docs/architecture.md) |

下一步是双假设左右身份恢复：先在开发集通过同一套门槛，再对冻结测试集只评一次。TotalCapture / BioCV 获批后，为其编写独立导入器，用于检验触地相位和脚滑。

## 目录

| 路径 | 内容 |
|---|---|
| `pose_sim/` | 视觉感知：检测推理、几何与三角化、人物选择、观测校验、人体状态、公开数据导入与评测 |
| `motion_control/` | 重定向、参考轨迹、跟踪与平衡、停止/退出过渡、G1 指令与设备适配 |
| `config/` | 相机、视觉、数据集切分，以及 G1 型号、限位和控制参数 |
| `scripts/` | 环境安装、各阶段入口、基准与审计脚本；`clean_outputs.py` 清理生成产物 |
| `tests/` | `unittest` 回归测试 |
| `reports/` | 每轮简短验收报告（JSON） |
| `docs/` | 研究日志、模块说明、调研记录 |
| `assets/` | 模型校验和、机器人与人体资产及署名 |
| `outputs/` | 公开数据、运行结果、视频和日志（Git 忽略） |

## 常用命令

```powershell
python scripts/bootstrap.py --device cuda      # 视觉环境，写入 .runtime.json
python scripts/bootstrap_robot.py              # 机器人环境，写入 .robot-runtime.json
.\run.ps1 -Stage test                          # 视觉环境回归
.\run.ps1 -Stage robot-test                    # 机器人环境回归
.\run.ps1 -Stage robot-demo -Session outputs/my_human_g1 -Device cuda
.\run.ps1 -Stage vision-demo -InputSession outputs/human_walk_g1 -Session outputs/my_vision_run
.\run.ps1 -Stage stream-vision -InputSession outputs/human_walk_g1 -Session outputs/my_stream -DetectorWorkers 4 -PoseWorkers 4
.\run.ps1 -Stage refine-motion -Session outputs/my_motion
```

输出目录必须不存在；完整指标不通过时命令返回非零。安装细节、各阶段单独运行、结果文件格式和真实相机导入见 [安装、复现与历史结果](docs/setup_and_reproduction.md)。

## 文档

- [研究日志](docs/research_log.md)：对话约束、每轮决策与结果
- [视觉模块与复现](docs/vision_pipeline.md)、[联合状态](docs/vision_joint_state.md)
- 调研：[观测恢复](docs/observation_recovery_research_2026-10-18.md)、[前沿方法](docs/vision_frontier_2026-10-18.md)、[首轮视觉调研](docs/vision_research_2026-10-17.md)、[PMPose 对照](docs/pmpose_experiment.md)、[公开真实数据](docs/public_real_datasets_2026-10-17.md)
- [人体到 G1 的模块、科学问题与硬件边界](docs/source_robot_to_g1.md)、[架构与硬件迁移](docs/architecture.md)
- [安装、复现与历史结果](docs/setup_and_reproduction.md)

## 验收报告

`reports/` 按日期命名。视觉部分以 `recovery-events`、`vision-joint-state`、`pmpose-observation`、`vision-person-trajectory` 为当前基线；G1 控制以 `contact-observation`、`landing-momentum`、`stop-transition`、`recovery-rollout` 为准。`panda-g1-2026-10-17` 与 `robot-control-2026-10-17` 来自早期双臂示例，不作为全身模仿验收。

## 来源

- Blender: https://www.blender.org/ （示例固定 4.5.3，便于复现）
- RTMLib / RTMPose: https://github.com/Tau-J/rtmlib
- RTMPose 官方模型: https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose
- CesiumMan: https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/CesiumMan
- 资产署名与授权说明：[assets/ATTRIBUTION.md](assets/ATTRIBUTION.md)

项目未修改或替代外部软件、模型和资产各自的授权条款。
