# 时间对齐与低延迟时序处理实验：2026-11-08

## 结论

原 S2/S3 架构现在可以接收真实时间间隔，并新增了相机相对时差估计器、因果 alpha-beta 滤波、使用完整 $3\times 3$ 观测协方差矩阵的各向异性卡尔曼滤波（`AnisotropicKalmanFilter3D`），以及球面骨段方向/角速度卡尔曼滤波（`SphericalSkeletonKalmanFilter3D`）。

- **各向异性 $3\times 3$ 卡尔曼滤波（`kalman_3x3`）**：直接利用 S2 的 $3\times 3$ 协方差矩阵与按特征轴分解的新息机动自适应，在三段留出集（`Take_006/007/008`）上实现平均 MPJPE **56.43 mm**（优于 `raw` 的 56.48 mm、`alpha_beta` 的 63.56 mm 和 `one_euro` 的 99.48 mm），平均肢段方向误差 **6.10°**（优于 `raw` 的 6.12°），同时将轨迹误差抖动 RMS 从 **19.11 mm/slot** 降至 **13.50 mm/slot**（降噪 29.4%），单帧耗时中位数 **0.86 ms**。
- **球面骨段方向/角速度卡尔曼滤波（`spherical_kalman`）**：将肢段相对向量分解为沿骨轴的径向方差（1D 骨长卡尔曼滤波）和球面切空间 $\mathbb{S}^2$ 上的 $3\times 3$ 方向协方差（大圆罗德里格斯角速度外推），将视觉骨长波动 CV 从 `raw` 的 **6.32%** 大幅压低至 **2.02%**（降低 68%），肢段方向误差保持 **6.15°**；但由于 COCO 2D 关键点存在视角相关的表观骨长偏移，沿运动学链串联重建使笛卡尔端点 MPJPE 略增至 **58.02 mm**。

这次修复了确定性的软件问题：首帧关键点缺测不再被初始化为世界原点；S3 可由单调时间戳计算真实 `dt`，长间隔会重置旧滤波状态；S2 可按真实 `dt` 使用速度门限，并在长间隔后丢弃旧帧的位移历史。

## 算法接口

| 模块 | 输入 | 输出 | 单位与时基 | 失败行为 |
|---|---|---|---|---|
| `estimate_camera_offsets` | 每相机 2D 点、流内 PTS、投影矩阵 | 每相机相对时差及成对残差 | 像素、秒；参考相机偏移固定为 0 | 极线目标平坦、最优值触及搜索边界、或成对时差循环不一致时标记不可靠；不得用于重采样 |
| `resample_keypoints_to_common_times` | 每相机 2D 点、PTS、已接受的相对时差、共同时间 | `(T,C,J,3)` 同步 2D 点 | 像素、秒 | 不外推；无有效左右括帧或检测置信度不足时输出置信度 0。线性插值会用到未来帧，延迟必须记账 |
| `MultiViewTriangulator` 可选速度模式 | 同步多视角 2D 点、投影矩阵、`dt_s` | 27 个 3D 点和协方差 | 米、秒 | `dt_s` 缺失或无效时报错；超过 150 ms 的间隔不使用旧点做位移门限 |
| `TemporalFilter` 时间戳模式 | `(27,3)` 3D 点、严格递增 `timestamp_s` | 3D 低通点和速度 | 米、秒、米/秒 | 乱序或重复时间报错；超过 150 ms 的间隔重置 One-Euro 状态；未初始化的缺测点保持 NaN |
| `CausalAlphaBetaFilter3D` 实验候选 | `(27,3)` 3D 点、PTS、可选协方差 | 点、速度、新鲜度掩码和观测年龄 | 米、秒、米/秒 | 最多保持旧点 80 ms，超时后返回 NaN；不对缺测点外推新姿态 |
| `AnisotropicKalmanFilter3D` (`kalman_3x3`) | `(27,3)` 3D 点、`timestamp_s`、`(27,3,3)` 协方差 $\Sigma_t$ | 点、速度、新鲜度掩码和观测年龄 | 米、秒、米/秒、$\text{m}^2$ | 特征值钳位保证 $\Sigma_t$ 对称正定；按 $\Sigma_t$ 特征轴做新息机动膨胀；超过 80 ms 缺测返回 NaN |
| `SphericalSkeletonKalmanFilter3D` (`spherical_kalman`) | `(27,3)` 3D 点、`timestamp_s`、`(27,3,3)` 协方差 $\Sigma_t$ | 球面链式重建 3D 点、速度、新鲜度掩码和观测年龄 | 米、秒、弧度/秒、$\text{m}^2$ | 骨段方向在 $\mathbb{S}^2$ 上用罗德里格斯旋转预测，切空间 $J_u (\Sigma_a+\Sigma_b) J_u^\top$ 更新，径向方差 $u^\top \Sigma_d u$ 驱动 1D 骨长卡尔曼滤波 |

相机时差估计通过校准投影矩阵构造基础矩阵，按候选相对偏移比较跨相机 Sampson 极线误差，再对相机对形成的偏移图做鲁棒一致性检查。它只估计相机间相对时差，不确定绝对时间原点。当前验证用 2 秒启动段估计偏移；这是离线验证设定，不代表在线首帧即可立即工作。

## 留出片段结果

输入为 AdvancedTake 的四路既有 2D 关键点和 `cameras_rgbonly.json`。时差只从 2D 极线一致性估计。共有时间槽仍沿用当前参考相机的既有末帧/ BVH 槽映射；固定 `+6` 个 BVH 采样的评分偏移对所有分支相同。用第一段 2 秒的旧链路估计一个固定刚体世界变换，后续帧用于评分；不对每种滤波器分别拟合空间或时间偏移。相对时差插值的前视等待另行记录。

| Take | 偏移一致性 | 相对偏移重采样的前视等待 p95 | 原对齐中位重投影误差 | 新对齐中位重投影误差 | 原始点 MPJPE | 新对齐 MPJPE |
|---|---|---:|---:|---:|---:|---:|
| 006 | 拒绝：循环残差 258.1 ms | 不应用 | 3.46 px | 不应用 | 50.25 mm | 不应用 |
| 007 | 接受：循环残差 4.41 ms | 28.85 ms | 3.38 px | 3.37 px | 66.16 mm | 65.71 mm |
| 008 | 接受：循环残差 11.75 ms | 10.45 ms | 3.39 px | 3.41 px | 53.04 mm | 52.39 mm |

007/008 的估计偏移与末帧对齐假设接近；在该批留出数据中，3D 误差改善不到 1 mm，重投影变化也极小。007 的前视等待 p95 约 29 ms，已经高于一个 30 Hz 帧周期附近的低延迟预算。006 的成对偏移互相矛盾，因此被拒绝；没有挑选单个相机对的结果强行应用。

固定旧对齐下，五种因果时序处理方法在同一评分窗口上的留出集表现为：

| 方法 | Take_006 MPJPE | Take_007 MPJPE | Take_008 MPJPE | 三段平均 MPJPE | 平均肢段方向误差 | 平均骨长波动 CV | 误差抖动 RMS | 单帧中位耗时 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 原始 S2 点 (`raw`) | 50.25 mm | 66.16 mm | 53.04 mm | 56.48 mm | 6.12° | 6.32% | 19.11 mm | — |
| `one_euro` | 73.44 mm | 139.50 mm | 85.49 mm | 99.48 mm | 9.56° | 4.51% | 15.63 mm | 0.08 ms |
| `alpha_beta` | 56.17 mm | 76.98 mm | 57.53 mm | 63.56 mm | 6.84° | 5.83% | 11.64 mm | 0.21 ms |
| `kalman_3x3` | **49.75 mm** | 67.01 mm | **52.52 mm** | **56.43 mm** | **6.10°** | 5.30% | 13.50 mm | 0.86 ms |
| `spherical_kalman` | 52.05 mm | 68.97 mm | 53.05 mm | 58.02 mm | 6.15° | **2.02%** | 13.21 mm | 3.58 ms |

- `kalman_3x3` 在 `Take_006` 和 `Take_008` 上同时降低了 MPJPE、肢段方向误差和轨迹抖动，解决了标量增益滤波器（`one_euro`、`alpha_beta`）把横向运动和视线深度噪声混为一谈的问题。
- `spherical_kalman` 在保持肢段方向精度（6.15° vs 原始 6.12°）的同时，将骨长波动 CV 从 6.32% 压到 2.02%，消除了笛卡尔直线外推的切向过冲与视线拉伸。
- 可视化见 [comparison.png](../outputs/temporal_alignment_experiment_20261108/comparison.png)，逐项结果和诊断见 [report.json](../outputs/temporal_alignment_experiment_20261108/report.json)。

## 验证

使用仓库隔离实验环境运行：

```powershell
& .\outputs\retarget_ablation_20261107\runtime\Scripts\python.exe -m pytest tests/test_temporal_alignment.py tests/test_s3_temporal_filter.py tests/test_s2_triangulation.py tests/test_adv_eval.py -q
& .\outputs\retarget_ablation_20261107\runtime\Scripts\python.exe scripts/temporal_alignment_eval.py --takes Take_006 Take_007 Take_008
```

## 下游闭环验证（`s45c_keypoint_ik` + 官方 G1 `SONIC v1.1`）

固定 `KeypointIKRetargeter`、`scene_43dof.xml`、`SONIC v1.1`（`preview_s = 0.0`）以及与基线完全相同的初始平面原点/偏航锚点，仅替换 S3 时序滤波器，在全部五段真人数据（`Take_002/006/007/008/012`，共 25 次闭环）上的等权平均结果如下：

| S3 滤波器 | 存活 | 参考手臂误差 | 物理手臂误差 | 参考腿误差 | 物理腿误差 | 支撑脚滑移 | 终点偏差 | 平均最大倾角 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `one_euro`（基线） | 5/5 | 14.61° | 20.46° | 9.69° | 13.83° | **0.0612 m/s** | **0.307 m** | **14.65°** |
| `raw`（无滤波） | 5/5 | 11.89° | 18.03° | 8.05° | 12.27° | 0.1365 m/s | 0.482 m | 19.67° |
| `alpha_beta` | 5/5 | 12.79° | 18.60° | 8.42° | 12.64° | 0.0956 m/s | 0.511 m | 16.58° |
| `kalman_3x3` | 5/5 | **11.87°** | **17.70°** | 8.04° | 12.39° | 0.1103 m/s | 0.589 m | 19.08° |
| `spherical_kalman` | 5/5 | **11.87°** | **17.70°** | **8.01°** | **12.26°** | 0.1019 m/s | 0.459 m | 17.15° |

闭环实验揭示了两个关键事实：

1. **动作相似度（肢段方向误差）显著提升**：消除 One-Euro 的 67 ms 滞后并引入 $3\times 3$ 协方差后，`kalman_3x3` 与 `spherical_kalman` 将物理机器人的手臂方向误差从 **20.46° 降至 17.70°**（改善 **2.76° / 13.5%**），物理腿部方向误差从 **13.83° 降至 12.26°–12.39°**（改善 **1.44°–1.57° / 11.4%**）。在快速拍手+转身片段 `Take_012` 上，物理手臂误差从 `one_euro` 的 `24.98°`（`raw` 为 `22.06°`）进一步降到 `kalman_3x3` 的 `21.14°` 和 `spherical_kalman` 的 **`20.82°`**。
2. **在下游闭环中，`spherical_kalman` 综合优于 `kalman_3x3`**：虽然 `kalman_3x3` 的三维点笛卡尔 MPJPE 最低（`56.43 mm`），但 `s45c_keypoint_ik` 依赖的是单位骨段方向与骨盆/足底高度。`spherical_kalman` 通过抑制骨长拉伸（CV 降至 `2.02%`）并在球面上滤波角速度，在保持完全相同的手臂精度（`17.70°`）和略优的腿部精度（`12.26°`）的同时，比 `kalman_3x3` 降低了支撑脚滑移（`0.1103 → 0.1019 m/s`）、降低了机身最大倾角（`19.08° → 17.15°`），并减小了路径终点偏差（`0.589 → 0.459 m`）。
3. **低延迟与控制器打滑的权衡（未隐藏的负面指标）**：所有低延迟滤波器（`raw`、`kalman_3x3`、`spherical_kalman`）的支撑脚滑移（`0.10–0.14 m/s`）和终点偏差（`0.46–0.59 m`）仍高于强低通的 `one_euro`（`0.061 m/s`、`0.307 m`）。原因是 `one_euro` 大幅削平了 30 Hz 输入的关节速度差分 `dq` 和高度抖动，相当于把动作放慢变缓；而恢复真实动作相位与幅度后，`SONIC v1.1`（零预览模式）在跟踪高频步伐与根部高度变化时产生了更多轻载脚滑。

闭环汇总图见 [closed_loop_summary.png](../outputs/temporal_alignment_experiment_20261108/closed_loop_summary.png)，完整数据见 [closed_loop_report.json](../outputs/temporal_alignment_experiment_20261108/closed_loop_report.json)。

## 验证

使用仓库隔离实验环境运行：

```powershell
& .\outputs\retarget_ablation_20261107\runtime\Scripts\python.exe -m pytest tests/test_temporal_alignment.py tests/test_s3_temporal_filter.py tests/test_s2_triangulation.py tests/test_adv_eval.py -q
& .\outputs\retarget_ablation_20261107\runtime\Scripts\python.exe scripts/temporal_alignment_eval.py --takes Take_006 Take_007 Take_008 --closed-loop
```

结果为 **23 passed, 2 xfailed**。两个 xfail 是已有 S2/S3 规范参数反例测试，不是本次新失败。单元测试覆盖了 $3\times 3$ 协方差各向异性降噪、球面骨长径向方差抑制、长间隔状态过期以及四相机极线时差恢复。没有真实工业相机同步或机器人测试。

## 当前决定

- 按用户决策，主线 `TemporalFilter`（`g1_imitation/core/s3_temporal_filter.py`）与 `g1_imitation/config/pipeline_config.json` 已直接切换为 `mode = "kalman_3x3"`（`AnisotropicKalmanFilter3D`），同时支持配置切换 `"spherical_kalman"` 与 `"one_euro"`。
- 主线脚本（`scripts/advanced_take_s2s3_eval.py`、`scripts/advanced_take_s1b_regression.py`、`scripts/mamma_multiperson_eval.py`、`scripts/render_showcase_videos.py`）已统一将 S2 输出的 $3\times 3$ 协方差矩阵 `point_covariances_m2` 与真实时间戳 `timestamp_s` 传入 `TemporalFilter.process_frame(...)`。
- 升级后，主线 `selfcal_lag6` 留出集（`Take_006/007/008`）滤波后 MPJPE 从原 One-Euro 的 **61.2 mm** 降至 **35.78 mm**（`35.74 / 39.37 / 32.24 mm`，与未滤波 `raw` 的 `35.56 mm` 持平，且 `Take_006`、`Take_008` 优于 `raw`），平均肢段方向误差从 **6.8°** 降至 **5.97°**，轨迹误差抖动 RMS 从 `17.34 mm/slot` 降至 **`12.19 mm/slot`**；S3 3 Hz 动态正弦测试相位滞后 **< 15.0 ms**（`test_s3_default_kalman_dynamic_3hz_phase_lag` 通过）。
