# 多视角 RGB → 全身三维关节点仿真

本项目将 **Blender 虚拟相机**与 **真实图像可复用的姿态估计管线**分开。
目标是先在本机跑通单人、固定多相机环境，再用真实图像及标定文件替换数据源。

本次交付的正式结果在 **`outputs/session01`**。`outputs/demo` 与 `outputs/verified`
是被后续修正替代的调试输出，不能用于验收；各目录已放置 `DEBUG_ONLY.txt`。

验收结果（2026-10-16）：4 视角 × 12 帧 = 48 张图像；实际检测后重建
201/204 个 COCO17 点（98.53%）。12 个身体骨骼代理点的有效覆盖率 99.31%，
平均代理误差 54.77 mm、中位数 38.94 mm、P95 142.61 mm；平均内点重投影误差
2.95 px。这不是人体真实标注精度，见下方代理真值限制。
相机导出与 Blender 投影最大差异 0.000054 px；6 项测试通过，包含无真值文件的
真实输入格式、空场景拒绝、畸变、异常视角、低视差与同步门限。
本机 CPU 串行推理平均 0.634 秒/图像，当前是离线原型，未达到多路实时采集处理。

```text
Blender：人体动画 + 4 台同步虚拟相机
       ↓ RGB + 相机标定 + 时间戳
YOLOX 人体检测 → RTMPose COCO17 二维关键点
       ↓
置信度筛选 → 多视角 RANSAC / 加权 DLT → 鲁棒重投影优化
       ↓
世界坐标中的 17 个关节点 + 有效性 + 质量指标
```

**真实图像迁移时保留 YOLOX、RTMPose、三角化和结果格式。**
Blender 仅负责生成测试输入。模型真值只由独立评测程序读取，绝不充当二维检测结果。

## 当前范围

- 单人，四台固定相机，640×640，默认 12 fps / 12 帧的短动作片段。
- 输出 COCO17；置信度不足、视角不足或几何退化的关节写为 `null`，不填入真值或伪造坐标。
- 实际检测渲染出来的 RGB，ONNX Runtime CPU 后端，无 CUDA 要求。
- Blender Cycles CPU 渲染，运行时与系统 Python 隔离。
- 真实图像与仿真图像共享 `cameras.json`、`frames.jsonl` 数据接口。
- 示例 CesiumMan 是风格化的公开人体动画，适合功能验收，不是写实人体精度基准。
- 示例模型仅有 12 个身体骨骼关节的评测代理值。面部 5 个点没有真值；骨骼中心与 COCO 标注定义也不完全相同。`body12_proxy_mpjpe_mm` 必须按此解释。
- 默认理想针孔、全局快门、精确同步。重建代码支持真实 Brown-Conrady 畸变；本例不模拟卷帘快门、真实曝光、网络延迟或相机触发电路。
- 本版不含多人跨视角身份关联、SMPL 拟合、机器人重定向或控制。

## 安装与启动（Windows PowerShell）

```powershell
python scripts/bootstrap.py
.\run.ps1 -Stage test
.\run.ps1 -Stage demo -Session outputs/my_new_session -Frames 12
```

安装脚本下载校验过 SHA256 的 Blender 4.5.3 便携版、开源人体示例，并创建隔离 Python 环境。
运行时位于 `%LOCALAPPDATA%/multiview-pose-sim`，不修改系统软件关联或基础 Python。
首次姿态检测还会下载官方模型权重。依赖与资产来源见文件末尾。

已存在的仿真 session 不会被渲染器覆盖。重复实验请换输出目录。
也可以对已有图像单独运行：

```powershell
.\run.ps1 -Stage detect -Session outputs/session01
.\run.ps1 -Stage reconstruct -Session outputs/session01
.\run.ps1 -Stage evaluate -Session outputs/session01
.\run.ps1 -Stage preview -Session outputs/session01
.\run.ps1 -Stage open-scene -Session outputs/session01
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
| `ground_truth.jsonl` | 只用于仿真评测的骨骼代理真值及其投影 |
| `metrics.json` | 实际检测重建的误差、有效率；以及独立几何自检结果 |
| `render_validation.json` | 导出相机与 Blender 投影是否一致 |
| `inference_timing.json` | CPU 串行图像推理耗时，不等同系统实时延迟 |
| `preview.png / .gif / .mp4` | 四视角检测结果与三维重建对照 |

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
.\run.ps1 -Stage detect -Session outputs/real01
.\run.ps1 -Stage reconstruct -Session outputs/real01
.\run.ps1 -Stage preview -Session outputs/real01
```

导入器以第一台相机的曝光时间为锚点匹配最近帧，不重复使用同一图像。
它不会建立硬件同步；运动越快，对时间误差要求越高。没有真实真值时跳过 evaluate。
设备采集 SDK、触发接线和标定结果仍需在具体硬件确定后接入，当前没有宣称已验证真实设备。

## 后续实验

建议先替换更接近真人的、具有明确关节定义的人体资产，再扩展动作、体型、服装、背景和光照。
每次扩展需保留独立真值导出及相机投影自检。可比较 2/4/6/8 相机、遮挡、失步与标定误差。
不要仅追求重投影误差小：错误的跨视角对应也可能给出看似一致的三维结果。

## 来源

- Blender: https://www.blender.org/ （示例固定 4.5.3，便于复现）
- RTMLib / RTMPose: https://github.com/Tau-J/rtmlib
- RTMPose 官方模型: https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose
- CesiumMan: https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/CesiumMan
- 资产署名与授权说明：[assets/ATTRIBUTION.md](assets/ATTRIBUTION.md)

项目未修改或替代外部软件、模型和资产各自的授权条款。
