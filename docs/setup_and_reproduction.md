# 鐜瀹夎銆佹暟鎹牸寮忎笌澶嶇幇鎸囧崡

鏈枃妗ｆ眹鎬绘湰椤圭洰鐨勫浐瀹氳繍琛岀幆澧冮厤缃€佸父鐢ㄨ繍琛屽叆鍙ｃ€佸悇闃舵缁撴灉鏂囦欢瀛楁鏍煎紡鍙婄湡瀹炵浉鏈烘暟鎹鍏ヨ鑼冦€傚悇妯″潡鐨勭畻娉曞師鐞嗚 [瑙嗚濮挎€佹ā鍧梋(vision_pipeline.md) 涓?[G1 鍏ㄨ韩閲嶅畾鍚戜笌鎺у埗](source_robot_to_g1.md)锛屽巻娆″疄楠岃褰曡 [鐮旂┒鏃ュ織](research_log.md)銆?
---

## 涓€銆?鍥哄畾杩愯鐜瀹夎锛圵indows PowerShell锛?
椤圭洰灏?*瑙嗚鎰熺煡鐜锛圕UDA + ONNX Runtime锛?*涓?*鏈哄櫒浜烘帶鍒剁幆澧冿紙MuJoCo + Pinocchio/SciPy + Ruckig + ONNX Runtime锛?*闅旂鍦?`%LOCALAPPDATA%/multiview-pose-sim` 涓嬶紝涓嶆薄鏌撶郴缁?Python锛?
```powershell
# 1. 鍒濆鍖栬瑙夌幆澧冿紙鍐欏叆 .runtime.json锛?python scripts/bootstrap.py --device cuda

# 2. 鍒濆鍖栨満鍣ㄤ汉鎺у埗鐜锛堝啓鍏?.robot-runtime.json锛?python scripts/bootstrap_robot.py

# 3. 杩愯鍏ㄩ噺鍥炲綊娴嬭瘯锛?52 椤癸級
.\run.ps1 -Stage robot-test
```

- **鐗堟湰涓庢牎楠?*锛氫娇鐢ㄥ浐瀹?`uv 0.9.7` 鍚屾鐙珛 `Python 3.11.9` 鐜锛屾墍鏈変緷璧栧寘銆丷TMPose/RTMW 妯″瀷浠ュ強 TWIST2/SONIC 鎺у埗鍣ㄦ潈閲嶅潎鏍￠獙 SHA256锛堣 `requirements-*-lock.txt` 涓?`assets/*/manifest.json`锛夈€?- **CUDA 绾︽潫**锛欸PU 鐜鍥哄畾 `onnxruntime-gpu 1.22.0` 涓?`cudnn 9.10.2.21`锛岃繍琛屾椂鑷姩妫€鏌ユā鍨嬩細璇濆苟鎷掔粷闈欓粯 CPU 鍥為€€銆傛棤 NVIDIA GPU 鐨勬満鍣ㄥ彲浣跨敤 `--device cpu` 鍒濆鍖?CPU 鐜銆?
---

## 浜屻€?鏍稿績杩愯鍏ュ彛

娉ㄦ剰锛氫负闃叉闈欓粯瑕嗙洊鍘嗗彶鏁版嵁锛屾墍鏈夎剼鏈殑鐩爣杈撳嚭鐩綍锛坄-Session` 鎴?`--output`锛?*蹇呴』涓嶅瓨鍦?*銆?
### 1. 鏍囧噯娴佹按绾垮懡浠わ紙`run.ps1`锛?
```powershell
# 瀵瑰凡鏈夊浘鍍忎細璇濊繍琛?33 鐐硅瑙夋娴嬨€佷笁缁撮噸寤轰笌鍥犳灉浜轰綋鐘舵€佷及璁?.\run.ps1 -Stage vision-demo -InputSession outputs/public_mamma_session -Session outputs/my_vision_run

# 瀹氭椂娴佸紡瑙嗚鍥炴斁锛? 璺娴嬪苟鍙?+ 4 璺Э鎬佹帹鐞嗗苟鍙戯級
.\run.ps1 -Stage stream-vision -InputSession outputs/public_mamma_session -Session outputs/my_stream -DetectorWorkers 4 -PoseWorkers 4

# 瀵瑰凡缂撳瓨鐨勪簩缁存娴嬭繍琛屽瑙嗚鍏宠仈銆佽娴嬩竴鑷存€т笌浜轰綋鐘舵€佷及璁?.\run.ps1 -Stage refine-motion -InputSession outputs/public_mamma_session -Detections outputs/public_mamma_session/detections.jsonl -Session outputs/my_motion

# 杩愯 BRM-C2 + 50 Hz C虏 杩炵画鍙傝€冧笌 SONIC v1.1 / TWIST2 浜旀鑷劧鍔ㄤ綔闂幆瀵规瘮
.\run.ps1 -Stage continuous-v3 -Session outputs/my_continuous_v3

# 鎴峰澶氫汉鏁版嵁闆?MAMMA 璺ㄨ瑙掓瀬绾胯仛绫婚€変汉涓庡洓闃舵闂幆婕旂ず
.\run.ps1 -Stage mamma-teleop
```

### 2. 鐪熷疄鍏紑鏁版嵁闆嗕笌杩炵画鎺у埗璇勬祴鑴氭湰

```powershell
$visionPython = (Get-Content .runtime.json | ConvertFrom-Json).python
$robotPython  = (Get-Content .robot-runtime.json | ConvertFrom-Json).python

# MPI-INF-3DHP 鐪熷疄澶氳瑙掓暟鎹鍏ヤ笌璇勬祴
& $visionPython scripts/import_mpi3dhp.py --config config/public_mpi3dhp.json --output outputs/my_mpi_inputs
& $visionPython scripts/evaluate_mpi3dhp.py --dataset outputs/my_mpi_inputs --output outputs/my_mpi_eval

# 鐙珛鍦烘櫙鍦伴潰骞抽潰鎷熷悎
& $visionPython scripts/estimate_scene_ground.py --source outputs/public_datasets/mpi_benchmark/inputs/s1_walk --cache outputs/mainline_validation_20261019/inference/s1_walk --output outputs/my_ground

# 12 甯у洜鏋?IRLS 杩炵画鐘舵€?+ BRM-C2 杩炵画閲嶅畾鍚?+ TWIST2 / SONIC 闂幆瀵规瘮涓庢覆鏌?& $robotPython scripts/benchmark_continuous_twist_sonic.py --output outputs/my_twist_vs_sonic
& $robotPython scripts/benchmark_continuous_v3.py --output outputs/my_continuous_v3
& $robotPython scripts/visualize_continuous_v3.py --input outputs/my_continuous_v3

# 鎴峰澶氫汉鏁版嵁闆?MAMMA 璺ㄨ瑙掓瀬绾胯仛绫婚€変汉涓庡洓闃舵闂幆婕旂ず
& $robotPython scripts/run_mamma_teleop_showcase.py
& $robotPython scripts/render_end_to_end_showcase.py
```

---

## 涓夈€?缁撴灉鏂囦欢瀛楀吀

| 鏂囦欢鍚?| 浜х敓闃舵 | 瀛楁鍚箟涓庡绾?|
|---|---|---|
| `cameras.json` | 瀵煎叆 | 鐩告満鍐呭弬鐭╅樀 `K`銆佺暩鍙樼郴鏁?`distortion`銆佷笘鐣屽埌鐩告満鏃嬭浆 `R_world_to_camera` 涓庡钩绉?`t_world_to_camera_m`锛堢背鍒讹紝涓栫晫 Z 杞村悜涓婏級 |
| `frames.jsonl` | 瀵煎叆 | 姣忎竴甯у悓姝ュ浘鍍忓寘锛歚frame_index`銆佸悕涔夋椂闂存埑涓庡悇瑙嗚鍥惧儚鐩稿璺緞鍙婃洕鍏夋椂闂?`timestamp_ns` |
| `detections.jsonl` | 2D 妫€娴?| 鍚勮瑙掑叿鍚嶄簩缁村叧閿偣鍍忕礌鍧愭爣 `keypoints_px`銆佺疆淇″害 `scores`銆佷汉鐗╂涓庢ā鍨?SHA256 绛惧悕 |
| `poses3d.jsonl` | 3D 涓夎鍖?| 閫愬抚鐙珛涓夎鍖栫殑涓夌淮涓栫晫鍧愭爣 `joints_world_m`銆佸唴鐐硅瑙掋€侀噸鎶曞奖璇樊銆佷笁瑙掑寲澶硅鍙婂眬閮ㄧ嚎鎬у寲鍗忔柟宸?`position_covariance_m2` |
| `states3d.jsonl` | 浜轰綋鐘舵€佷及璁?| 鍥犳灉浜轰綋鐘舵€侊紙`human_state_v1` / `human_state_v2`锛夛細婊ゆ尝/棰勬祴浣嶇疆銆侀€熷害銆佹瘡鐐圭姸鎬侊紙`filtered` / `predicted` / `missing`锛夈€侀闀夸及璁°€佽函骞?瓒虫帉灞€閮ㄥ潗鏍囩郴鍙婃帴瑙﹀€欓€?|
| `humanoid_reference.jsonl` | 鍏ㄨ韩閲嶅畾鍚?| 25 Hz 婧愬抚鐜囦笅鐨?G1 35 缁村叏韬弬鑰冿紙鏍归儴灞€閮ㄩ€熷害銆佷笘鐣岄珮搴︺€佹í婊?淇话銆佸亸鑸閫熷害銆?9 鍏宠妭鐩爣瑙掞級鍙婂噯鍏ョ姸鎬?鎷掔粷鍘熷洜 |
| `*_brm_c2_50hz_stream.npz` | 杩炵画鍙傝€冪敓鎴?| 缁忓寮傚鍨掑崟璋冩煍鍖栭噸瀹氬悜涓?Ruckig $C^2$ 杞ㄨ抗鐢熸垚鍚庣殑 50 Hz 杩炵画鍙傝€冩祦锛堜綅缃€侀€熷害銆佸姞閫熷害鍧囪繛缁湁鐣岋級 |
| `g1_commands.jsonl` | 闂幆鎺у埗鍥炴斁 | 鍙戦€佺粰妯℃嫙鎺ユ敹鍣ㄧ殑 50 Hz 鍏峰悕 29 鍏宠妭鎸囦护锛堝惈搴忓彿銆佹湁鏁堟湡 `valid_until_ns`銆佹ā鍨嬪搱甯屻€丳D 鍙傛暟涓庡姏鐭╅檺骞咃級 |

---

## 鍥涖€?鐪熷疄澶氱浉鏈烘暟鎹鍏ヨ鑼?
鎺ュ叆瀹為獙瀹ょ湡瀹炵浉鏈烘椂锛屼繚鎸佹瘡鍙扮浉鏈虹劍璺濄€佸垎杈ㄧ巼鍥哄畾锛屽畬鎴愬唴澶栧弬鏍囧畾锛屽苟鍑嗗濂藉寘鍚‖浠舵洕鍏夋椂闂存埑鐨?CSV 娓呭崟锛坄camera_id,timestamp_ns,image`锛夛細

1. **鐩告満鏍囧畾鏂囦欢鏍煎紡锛坄cameras.json`锛?*锛?
```json
{
  "schema_version": 1,
  "units": "metre",
  "world_axes": "right-handed, Z-up",
  "camera_axes": "OpenCV: X-right Y-down Z-forward",
  "cameras": {
    "cam00": {
      "K": [[1200.0, 0.0, 960.0], [0.0, 1200.0, 540.0], [0.0, 0.0, 1.0]],
      "R_world_to_camera": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
      "t_world_to_camera_m": [0.0, 0.0, 3.0],
      "distortion": [0.0, 0.0, 0.0, 0.0, 0.0],
      "width": 1920,
      "height": 1080
    }
  }
}
```

2. **鎵ц瀵煎叆涓庡悓姝ラ厤瀵?*锛?
```powershell
$visionPython = (Get-Content .runtime.json | ConvertFrom-Json).python
& $visionPython scripts/import_real.py --cameras path/to/cameras.json --manifest path/to/frames.csv --output outputs/real_session01 --tolerance-ms 5.0
```

瀵煎叆瀹屾垚鍚庯紝`outputs/real_session01` 涓庡叕寮€鏁版嵁闆嗕細璇濆叿鏈夊畬鍏ㄧ浉鍚岀殑鎺ュ彛锛屽彲鐩存帴杩愯 `vision-demo`銆乣stream-vision` 鎴栦笅娓?G1 閲嶅畾鍚戜笌鎺у埗鑴氭湰銆?