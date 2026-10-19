# 真实多视角视频与动作参考：检索、下载和首轮测试

检索开始日期：2026-10-17。范围是原始 RGB / RGB-D、同步动作参考、相机标定和可核实的发布入口；不是只有宣传视频、预计算二维点或动作类别的资源。访问状态是本次实测，不能保证未来一直可用。

## 如何理解“真值”

可以用这些数据检验当前算法，但必须区分参考来源：

- **独立光学动捕**：Vicon、Qualisys、OptiTrack 等测量的标记轨迹，再经过骨架拟合得到关节和角度。比视觉重建更独立，但关节中心、软组织、拟合和同步仍有误差。
- **多视角视觉 / RGB-D 重建**：真实视频是真的，三维参考是另一套算法重建的。可用于回归和误差诊断，不能当作独立、无误差的物理测量。
- **SMPL / SMPL-X 参数**：是模型拟合或重定向结果；不能仅因文件里有旋转矩阵，就声称它测量到了完整关节轴向扭转或接触力。

我们的感知仍只输入 RGB、标定和时间。独立动捕标记仅用于数据提供方建立评分参考，不进入检测器或机器人控制。带动捕服/可见标记的数据，还需要另用普通服装视频检验外观迁移。

## 核实的候选

| 数据集 / 官方入口 | 视频和参考来源 | 对本项目的价值 | 本次获取情况与限制 |
|---|---|---|---|
| [MPI-INF-3DHP](https://vcai.mpi-inf.mpg.de/3dhp-dataset/) | 真实 14 相机录像、标定、逐帧 2D/3D；多相机无标记动捕参考 | 最快建立完整 RGB→3D 实测基线，姿态、服装和视角有变化 | 已选 S1/Seq1、S2/Seq2，各 4 相机；官方 HTTPS 直取，研究非商业用途，禁止再分发。使用官方训练分区，不称官方测试集 |
| [TotalCapture](https://cvssp.org/data/totalcapture/) | 8 台 HD、60 Hz，5 人；同步 Vicon、IMU、视频 | 独立位置/旋转、行走与快速动作，适合第二条验收线 | 官方数据目录实测 401；需登记并确认研究使用协议，无账号不能直接下载 |
| [Human3.6M](https://vision.imar.ro/human3.6m/description.php) / [条款](https://vision.imar.ro/human3.6m/eula.php) | 多视角真实录像和光学动捕 | 人物隔离的标准姿态比较 | 需注册并接受数据条款；未获取账号、未借用第三方镜像 |
| [Fit3D](https://fit3d.imar.ro/fit3d) | 4 视角、900×900、50 fps，Vicon、25 关节、GHUM/SMPL-X | 蹲起、弓步、体型及关节角度 | 需申请访问；官方 test 只有单个随机视角，多视角实验须另用训练部分按人物划分，不能冒充官方 test |
| [MoVi](https://www.biomotionlab.ca/movi/) / [访问教程](https://www.biomotionlab.ca/Data/Tutorials/DataverseTutorialBMLmoviV2.pdf) | 光学动捕、Visual3D/MoSh++、IMU、真实录像 | 人物、动作多样性；姿态与体型；角度参考 | 需机构邮箱和 Dataverse 文件授权；固定 PG 相机和动捕同步，手机相机并非同等硬件同步，不可直接等帧号配对 |
| [OpenCap 实验室验证](https://simtk.org/frs/?group_id=2385) / [官方复现代码](https://github.com/opencap-org/opencap-core/blob/main/ReproducePaperResults/labValidationVideosToKinematics.py) | 10 人、5 手机视角，独立动捕、测力台、肌电和 OpenSim 结果 | 与目标高度相关：走路、蹲起、坐站、跳跃；可验证支撑与动力学相关量 | 列表能读取；LabValidation_withVideos 下载确认页跳转登录。FieldStudy 不含 RGB，不能替代。未登录、未下载正文 |
| [BioCV](https://researchdata.bath.ac.uk/1258/) | 15 人、9 路 200 fps、Qualisys、1000 Hz 测力台，标定与 C3D | 快速动作、同步、接触/支撑验证特别有价值 | 官方明确要求申请并接受伦理使用条款；每人约 15 GB。未申请身份、未下载受限文件 |
| [CMU Panoptic](https://github.com/CMU-Perceptual-Computing-Lab/panoptic-toolbox) | 密集同步多视角；人体/手/脸的视觉三维重建 | 遮挡、左右混淆、多人和手部 | 官方数据站本次访问失败；官方工具列出 CMU 与 SNU 入口。参考不是独立 Vicon，不能混称 |
| [AIST++](https://google.github.io/aistplusplus_dataset/download.html) / [AIST 条款](https://aistdancedb.ongaaccel.jp/terms_of_use/) | 9 视角舞蹈，2D→三角化→SMPL 拟合 | 转身、交叉步、快速动作、时序 | 原始 AIST 视频要求先填申请表；未代填。预计算关键点不能替代 RGB 测试；需按官方 60 fps 对齐与忽略清单处理 |
| [RICH](https://rich.is.tue.mpg.de/) | 多视角真实视频、人体/场景拟合及接触标注 | 地面/环境接触，室内外变化 | 下载需登录和许可；接触标签不是测力台，不能当接触力真值 |
| [HuMMan-Recon](https://caizhongang.com/projects/HuMMan/recon.html) / [作者链接的镜像](https://huggingface.co/datasets/caizhongang/HuMMan) | 153 人、339 序列，真实图像、相机、SMPL 与 RGB-D 重建 | 体型、外观、运动迁移 | 存在作者授权镜像；大规模数据，本次未整库下载。自动拟合参考，不是独立光学动捕 |
| [HUMBI](https://humbi-data.net/) / [原论文](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yu_HUMBI_A_Large_Multiview_Dataset_of_Human_Body_Expressions_CVPR_2020_paper.pdf) | 多视角人体/手/脸、网格重建 | 外观、体型、手部跨视角 | 官网连接未成功；不是本次连续运动基线。未下载 |
| [HumanEva](https://www.cs.ubc.ca/~lsigal/download.html) | 同步多相机和 Vicon；I/II 的同步方案不同 | 小规模独立动捕基线 | 作者说明需注册；需尊重坏帧/同步修正，不可用别人预处理的 2D 点代替原视频 |
| [HUMAN4D](https://tofis.github.io/human4d_dataset/) / [Zenodo 示例](https://zenodo.org/records/4484110) | 4 RGB-D、Vicon、逐序列时间偏移 | 遮挡、多模态和双人 | 文件受限，公开页面不代表文件无条件可下载；需申请 |
| [HUM4D，2026](https://parkyeeun23.github.io/HUM4D/) / [论文](https://arxiv.org/abs/2604.12765) | 6 RGB-D、Vicon→SMPL；遮挡、身份交换、多人交互 | 最新的困难动作与遮挡候选 | 官方公开 Drive 可下载，CC BY 4.0。已取得并审计一个视频、标定和对应名称的 NPZ；存在下述契约缺口，暂不进行定量评分 |
| [首都体院双系统数据，2026 v2](https://zenodo.org/records/19720803) | 页面称双手机+120 Hz OptiTrack，21 人、7 类运动 | 若补齐 RGB/标定，可用于走跑蹲和独立参考 | API 显示 CC BY 4.0；实际检查 03_markerless_data.zip 中央目录，只有 294 个 CSV 和目录，未发现视频。不能据网页描述宣称已经获得 RGB 测试集 |

优先顺序：先完成 MPI-INF-3DHP 可复现基线；申请 TotalCapture / OpenCap / BioCV 建立独立测量与接触验证；补齐 HUM4D 的标定/时间对应后再纳入困难集。AMASS 等单独动作参数库不能替代真实 RGB 端到端验证。

2026-10-18 后续更新：S4/Seq1、S6/Seq2 的四路完整视频也已下载并完成冻结人物留出评测。当前共 4 人、992.52 秒原始时间轴，已评分 72 秒；来源及结果见 [简短研究日志](research_log.md) 和 [观测校验验收](../reports/vision-consistency-2026-10-18.json)。上表获取状态描述首轮调查，不能理解为目前只下载了两人。

## 已下载 HUM4D 样本的具体问题

本地：`outputs/public_datasets/hum4d_probe/`，逐文件 URL、SHA256 和检查结果见 `audit.json`。

选取公开目录 `Group_Break_Formation_MocapSuite_take001_Human01_cam_242422303238`：

- RGB 视频 1280×720、15 fps、89 帧；同名 Human01 的 NPZ 为 155 帧、15 fps。
- 标定 TXT 实为 RealSense XML，只提供传感器内部双目/RGB 参数，缺少该 RGB 相机到动捕世界的外参。
- 标定 RGB 尺寸为 1280×800，视频是 1280×720，缺少裁切/缩放映射。
- NPZ 的图像路径文字指向 `mocap_suit_with_dress`，与样本名不同，需要验证对应关系。

只安全读取 NPZ 数值数组，未执行 pickle 对象或上游代码。没有猜测外参、截断两者后硬配对、或通过对齐参考来掩盖这些问题。这个结论只针对已下载样本，不等同于宣布整库不可用。

## MPI-INF-3DHP 固定协议与模块接口

配置：`config/public_mpi3dhp.json`。S1/Seq1、S2/Seq2；相机 0/2/5/7；每序列在 20、80、200 秒处取 6 秒，25 Hz，共 900 个同步时刻、3600 张输入图像。片段和算法参数在评分前固定；活动名称仅取自作者约一分钟的活动区块说明，不是人工逐帧分类。本次含出画和躺卧困难样本，仍计入分母。

源视频 2048×2048，推理缩至 1024×1024；内参使用与 OpenCV resize 一致的像素中心映射。标定平移与 annot3 从 mm 转 m，世界轴 `X=X_provider, Y=-Z_provider, Z=Y_provider`，保持右手系。数据集相对时间为 `frame_index / 25`，整型纳秒；不假装知道真实曝光延迟。

- `pose_sim/public_download.py`：官方 HTTPS、有限并发分段，检查 206/Content-Range、大小、ETag；按显式文件名提取 ZIP 成员并验 CRC。失败不产生正式文件，不执行下载的脚本。
- `pose_sim/mpi3dhp.py`：标定与参考转换。仅采用 `annot3`，不用统一骨长的 `univ_annot3`；检查零起始帧号和跨相机世界参考一致性。坐标、帧号、相机模型不合约则失败。
- `scripts/import_mpi3dhp.py`：保留原视频，生成 `inputs/` 和独立 `references/`，记录源文件哈希；不复制大量逐帧 JPEG。
- `scripts/evaluate_mpi3dhp.py`：推理阶段只读取 `inputs/` 中视频、标定和时钟，CUDA 检测/2D 模型、CPU 几何/状态；完成后评分才读 `references/`。同一 2D 观测比较原始几何、关闭上一轮细节修正的状态、当前状态。

只对应 12 个肩/肘/腕/髋/膝/踝关节。世界位置误差不做刚性、尺度、时间或根部对齐，另列髋中心相对误差；同时统计覆盖率、预测点、所有目标中误差≤100 mm 的比例、共同输出集合误差。报告躯干地标坐标系和肘膝屈曲误差；不称为完整解剖旋转真值。速度由参考位置差分，仅作诊断。相位延迟分辨率 40 ms；无接触/遮挡事件真值，不编造脚滑或遮挡恢复评分。

这是官方训练部分上的首次本地探索性测试，预训练模型可能见过相关数据，不能称为完全独立测试或与论文排行榜横比。用于调参后，应把这批永久转为开发集，另外固定未参与调参的人物/序列，并补上跨数据集独立动捕测试。

## 复现

```powershell
$rt = Get-Content .runtime.json | ConvertFrom-Json
& $rt.python -X utf8 scripts/import_mpi3dhp.py --download --output outputs/public_datasets/mpi_benchmark
& $rt.python -X utf8 scripts/evaluate_mpi3dhp.py --dataset outputs/public_datasets/mpi_benchmark --output outputs/public_datasets/mpi_results --preview
& $rt.python -X utf8 -m unittest tests.test_mpi3dhp -v
```

已存在的评测目录会拒绝覆盖。更换配置应使用新实验目录；保留当前基线后按项目清理约定处理被替代产物。视频、参考、预览与运行日志均留在忽略的 `outputs/` 内，仓库只保留脚本、配置、测试和简短报告。

使用 MPI 数据发表结果时须引用 Mehta 等人的 *Monocular 3D Human Pose Estimation In The Wild Using Improved CNN Supervision*（3DV 2017）与 *VNect: Real-time 3D Human Pose Estimation With A Single RGB Camera*（ACM TOG / SIGGRAPH 2017）。本轮仅作本地非商业评测；未训练或发布模型，未分发数据。

## 2026-10-18 完成结果

源文件共 623,152,044 字节，8 路完整视频及两套动作/标定文件。实际评分 900 个同步时刻、3600 张图像、10800 个身体关节目标。推理配置与上一轮保持一致，未用本次参考调参。

| 指标 | 原始几何重建 | 当前时序状态 |
|---|---:|---:|
| 各自可用输出的世界位置均值 | 66.69 mm | 66.72 mm |
| 各自可用输出的 P95 | 132.18 mm | 132.53 mm |
| 输出覆盖率（含短期预测） | 97.97% | 99.00% |
| 全部目标中误差≤100 mm | 84.06% | 85.33% |
| 相同 10563 个目标上的位置均值 | 66.55 mm | 65.45 mm |

因此不能单凭不同覆盖率的均值宣称提升；当前状态在共同目标上稍好，并补出部分短期预测，但仍有严重失败。当前最大误差 1206.15 mm，31 个关节点次超过 500 mm，其中 11 个仍被当作当前观测输出；另有 6 个点次原始重建低于 150 mm 而状态超过 500 mm。

六片段当前均值分别为 67.69、46.92、85.85、70.46、69.04、61.14 mm。片段级躯干地标旋转均值约 13.11–24.64°，肘膝屈曲约 6.42–12.39°。这些是参考地标导出的误差，不是独立测量的完整解剖旋转。

具体失败证据在 `outputs/public_datasets/mpi_results/failure_5097_2d.jpg`：camera 5 把椅子识别成人；camera 0/5 的真人出画或截断。第 5097 帧右腕采用 camera 0/7，重投影误差只有 0.326 px，三维误差却 1173.57 mm。第 5099 帧原始右腕恢复至 125.79 mm 误差，时序门限仍以 `temporal innovation outlier` 拒绝，输出误差 1206.15 mm 的预测。

下一轮应以这些反例驱动：跨视角人物身份与出画检测；错误对应条件下的不确定性；可信新观测对错误历史状态的恢复机制。上一轮手脚细节修正开关对本次 12 个身体关节指标完全相同，本数据不能证明手掌/脚掌朝向改进。

三套 ONNX 模型启用 CUDA，计算平均 112.60 ms/时刻（检测与二维 81.06 ms，几何与状态 31.54 ms）；额外视频解码平均 41.22 ms。这里只测完整帧离线吞吐，未证明 25 Hz 在线运行或真实曝光延迟。启动出现一次 cuDNN execution-plan 日志，模型继续完成全部推理；未改环境，运行时验证 CUDA provider 为首选并禁用会话级静默回退。

回归：120 项执行，114 通过、6 项因需机器人环境跳过、0 失败；新增 6 项数据/评分契约测试通过。H.264 对照视频实解码 900 帧、25 fps、36 秒、1024×1088。绿色为参考、橙色为估计。完整数字、哈希与限制见 `reports/public-real-data-2026-10-18.json`；完整逐片段指标在 `outputs/public_datasets/mpi_results/report.json`。
