# 本机 CPU 基线与真实视频离线复现

适用于 Windows x64、Python 3.11.9。CPU 环境与资产放在 `outputs/local_cpu/`，原始视频放在 `data/mamma_cpu_raw/`，均已被 Git 忽略。原 `pipeline_config.json` 和 CUDA/OptiX 配置保持不变。

当前已恢复 **本机 CPU 主线环境**、**G1 因果关键点 IK + SONIC 物理闭环**、**MAMMA 离线视频复现**，以及 **海康 4 相机纯 PoE 同步采集、棋盘格标定、3D 骨架重建与 G1 实时跟随窗口**。实测报告见 [`reports/hik-poe-sync-2026-11-09.md`](../reports/hik-poe-sync-2026-11-09.md)。

## 当前机器直接运行

在仓库根目录执行，无需激活虚拟环境或改变 PowerShell 执行策略：

```powershell
$py = '.\outputs\local_cpu\runtime\Scripts\python.exe'
& $py scripts/run_local_cpu.py test
& $py scripts/run_local_cpu.py baseline
& $py scripts/run_local_cpu.py offline
& $py scripts/run_local_cpu.py render
```

## 海康四相机硬件联调与实时跟随脚本

```powershell
$py = '.\outputs\local_cpu\runtime\Scripts\python.exe'
& $py scripts/hik_camera_check.py                                                     # 检查 4 台海康相机状态、曝光与清晰度
& $py scripts/hik_sync_capture_test.py --mode action --binning 2 --fps 30 --seconds 10 # 纯 PoE PTP + Action1 30Hz 同步测试
& $py scripts/hik_calib_capture.py --phase extrinsics --pattern 11x8 --session calib01 # 棋盘格同步采集
& $py scripts/calibrate_hik_cameras.py --session calib_full_1624 --square-mm 45 --board-thickness-mm 3 # 解算标定与地面 Z=0
& $py scripts/hik_live_pose_demo.py                                                   # 抓拍单帧 -> 3D 骨架 -> G1 重定向全景图
& $py scripts/hik_record_and_imitate.py --duration-s 8 --fps 15                       # 录制 8s 真人动作 -> SONIC 物理闭环 MP4
& $py scripts/hik_realtime_stream.py                                                  # 启动在线实时流式跟随窗口（按 Q 退出）
```

- `test`：G1、几何、时序与部署接口相关回归；不包含缺资产的 Roban、GMR 离线参照、MPI 视频集或 CUDA 延迟测试。
- `baseline`：10 组 × 10 秒合成关键点动作；保存参考、物理轨迹和力矩。逐动作记录存活、RMSE、脚滑、限位及耗时；任一合成验收失败则非零退出。**合成输入不验证视觉感知，原地跟踪不证明前进行走成功。**
- `offline`：完整 MAMMA 示例的 426 帧、4 路视频。指定 A001 中 `(440,440)` 像素处的人（按 1280×720 图像）为目标。保存每帧感知、滤波、参考与物理状态。执行结束不代表质量通过，以 `all_quality_criteria_pass` 为准。
- `render`：从保存轨迹生成 `preview.mp4`；只是呈现原仿真结果，不重新运行或修改物理。

默认输出分别是 `outputs/local_cpu/baseline/` 和 `outputs/local_cpu/offline_mamma_full/`。再次运行会覆盖对应生成结果；要保留多轮证据，可使用底层脚本的 `--out` 参数指定新目录。

## 在新目录恢复环境和资产

先安装 uv，并确保能获取 CPython 3.11.9。入口脚本只使用标准库，可以用已有 Python 启动：

```powershell
uv run --python 3.11.9 --no-project scripts/run_local_cpu.py setup
```

若已有项目部署包，可复用其中经过校验的模型网格，避免重新下载：

```powershell
uv run --python 3.11.9 --no-project scripts/run_local_cpu.py setup --model-zip 'C:\Users\twmmu\Desktop\g1_deploy_advtake_Take_007.zip'
```

`setup` 使用 `requirements-cpu-lock.txt` 精确同步独立环境，默认使用清华 PyPI 镜像；G1 模型来自 NVIDIA 固定提交，SONIC 权重来自 NVIDIA Hugging Face 发布并匹配原主配置 SHA256。视觉权重与示例视频使用 OpenMMLab / MPI 官方源，并与 `g1_imitation/config/local_cpu_assets.json` 中本次基线哈希比对。下载内容变化会拒绝接受，不自动更新基线。

模型和视频仅用于相应许可允许的本地研究与评测；不随本仓库再分发。MAMMA 视频和标定来源：

- https://github.com/cuevhv/mamma/tree/588492f18876e2ed6888b2d26929047cb6b575e7
- https://download.is.tue.mpg.de/mamma/assets/download/dataset_examples/pushing_and_lifting_from_ground/videos_light/
- https://huggingface.co/nvidia/GEAR-SONIC
- https://github.com/NVlabs/GR00T-WholeBodyControl/tree/45c03facbf848e0c743091b7d9c17952096ff908

## 配置、输入与失败行为

`load_config()` 优先使用显式路径，其次读取环境变量 `G1_IMITATION_CONFIG`，最后使用原主配置。入口脚本只对其子进程设置本机配置和单线程 BLAS，不修改全局环境。

离线输入为 30 Hz 视频与标定。标定平移以米计，四元数 wxyz 表示相机到世界旋转；入口求逆得到世界到相机外参，并按图像尺寸同比缩放内参。当前样例畸变系数为零；非零畸变会直接拒绝，不能静默套用针孔投影。检查各路解码 PTS 与名义帧时刻一致；**这不证明相机曝光硬件同步**。世界 Z-up、地面 z=0 沿用样例标定假设。

S3 使用主线 `kalman_3x3`；S2 缺测点即使被滤波器暂存，也在 IK 输入处还原为 NaN。IK 沿用原缺失肢体/躯干保持行为，不外推新动作；未锁定目标或未完成尺度校准时不会产生伪造参考。无可观测躯干、解码失败、时间异常、权重指纹错误、不可用资产会报错退出。仿真中的保持行为不等于实机安全停止机制。

AdvancedTake 的原始多视角视频及其标定未恢复。桌面部署包只有已生成的 Take_007 CSV 与展示视频，不能拿展示视频代替原始多视角输入。如需复现历史五段 AdvancedTake 结果，仍需原 `AdvancedTake.zip` 或视频目录和对应标定/时间信息。
