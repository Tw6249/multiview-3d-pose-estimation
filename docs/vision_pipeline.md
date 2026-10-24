# 澶氳瑙掕瑙夋劅鐭ユā鍧椾笌澶嶇幇璇存槑

鏈枃妗ｈ鏄?`pose_sim/` 涓嬪瑙嗚鏃犳爣璁颁汉浣撹瑙夋劅鐭ラ摼璺殑鍚勪釜鐙珛妯″潡銆佽緭鍏ヨ緭鍑哄绾︺€佸け璐ュ鐞嗘満鍒跺強澶嶇幇鏂瑰紡銆?
---

## 涓€銆?瑙嗚鎰熺煡閾捐矾鎬昏

```text
鍚屾澶氳瑙?RGB (frames.jsonl + cameras.json)
  鈫?1. 2D 妫€娴嬩笌鍏抽敭鐐瑰墠绔?(runtime.py / inference.py)
YOLOX-M 浜轰綋妗?+ RTMPose-M 韬綋 17 鐐?+ RTMW-L 鎵嬭冻 16 鐐?(human33)
  鈫?2. 璺ㄨ瑙掗€変汉涓庡叧鑱旀牎楠?(person_selection.py / observation_consistency.py / auxiliary.py)
鍗曠洰鏍囬攣瀹?+ 鐣欎竴瑙嗚閲嶆姇褰变竴鑷存€ф牎楠?+ 鍏宠妭瀵?鎵嬭冻缁嗚妭鍏宠仈
  鈫?3. 澶氳瑙掍笁缁撮噸寤?(geometry.py / multiview_temporal.py / uncertainty.py)
鍔犳潈 DLT + 椴佹閲嶆姇褰变紭鍖?+ 灞€閮ㄧ嚎鎬у寲 3脳3 鍗忔柟宸?(poses3d.jsonl)
  鈫?4. 鍥犳灉浜轰綋鐘舵€佷笌鎺ヨЕ浼拌 (human_state.py / human_motion.py / continuous_state.py / foot_contact.py)
楠ㄩ暱绾︽潫 + 鍥犳灉婊ゆ尝 / 12 甯?Huber IRLS 杩炵画鐘舵€?+ 鍦伴潰瓒抽儴鎺ヨЕ鍊欓€?(states3d.jsonl)
```

---

## 浜屻€?鏍稿績瀛愭ā鍧楀绾?
### 1. 2D 妫€娴嬩笌 33 鍏抽敭鐐瑰墠绔紙`pose_sim/runtime.py`銆乣inference.py`銆乣landmarks.py`锛?
- **妯″瀷缁勫悎**锛氶粯璁や娇鐢?`YOLOX-M` 妫€娴嬩汉浣撹竟鐣屾锛宍RTMPose-M` 鎻愬彇 COCO 韬綋 17 鐐癸紝`RTMW-L` 琛ュ厖宸﹀彸鍚?3 涓冻閮ㄧ偣锛坄big_toe`銆乣small_toe`銆乣heel`锛変笌宸﹀彸鍚?5 涓帉閮ㄧ偣锛坄wrist`銆乣thumb_mcp`銆乣index_mcp`銆乣middle_mcp`銆乣pinky_mcp`锛夛紝鏋勬垚 `human33` 鎷撴墤銆?- **閫夋嫨渚濇嵁**锛氫繚鐣欏弻妯″瀷娣峰悎锛圚ybrid锛夋槸鍥犱负鍗曠敤 RTMW 鏇挎崲韬綋鐐规椂楂嬮儴杩炵画鎬т笌鎺у埗鍣ㄥ噯鍏ョ巼涓嬮檷锛涜€岃韩浣撶綉缁滀繚鎸?`batch=1` 閫愬浘鎺ㄧ悊銆佷粎瀵?RTMW 鎵嬭冻缃戠粶鍚堟壒锛屽彲閬垮厤鎵瑰鐞嗗湪姝т箟鐐瑰垏鎹?SimCC 宄板€笺€?- **澶辫触琛屼负**锛氭棤鏈夋晥浜虹墿妗嗘椂涓嶆帹鐞嗗叧閿偣锛涘浘鍍忚竟鐣屽鐨勯娴嬬偣缃俊搴︾疆闆讹紝涓嶅厖褰撳彲瑙佽娴嬶紱CUDA 妯″紡寮哄埗楠岃瘉 ONNX Runtime GPU 鎵ц鎻愪緵鍣紝鎷掔粷闈欓粯鍥為€€ CPU銆?
### 2. 璺ㄨ瑙掍汉鐗╅€夋嫨涓庤娴嬩竴鑷存€ф牎楠?
| 妯″潡 | 杈撳叆涓庡崟浣?| 杈撳嚭 | 澶辫触琛屼负涓庨獙璇佽竟鐣?|
|---|---|---|---|
| `person_selection.py` | 鍚勮瑙?2D 鍏抽敭鐐瑰€欓€夋銆佺浉鏈烘爣瀹氥€佹洕鍏夋椂闂?`ns` | 鍚屾椂闂存埑涓嬫瘡涓瑙掔殑鍞竴閫変腑浜轰綋鍊欓€?| 鍒╃敤鍗曚汉瑙嗚涓夎鍖栬偐/楂?鑶濋敋鐐瑰苟鍥炴姇鍒板浜烘涔夎瑙掞紙瑕佹眰 $\ge 4$ 涓敋鐐癸紝涓綅娈嬪樊 $\le 20\text{ px}$銆丳80 $\le 35\text{ px}$锛屼笌娆′紭鍊欓€夊樊璺?$\ge 15\text{ px}$锛夈€傝瘉鎹笉瓒虫垨鍏ㄨ瑙掓涔夋椂淇濇寔缂哄け锛堣 [閫愮幆鑺備富绾块獙璇乚(../reports/mainline-validation-2026-10-19.json)锛夈€?|
| `observation_consistency.py` | 宸插叧鑱旂殑璺ㄨ瑙?2D 鍍忕礌鐐逛笌缃俊搴︺€佺浉鏈烘爣瀹?| 鍓旈櫎鐭涚浘瑙傛祴鍚庣殑 2D 璁板綍鍙婃嫆缁濆師鍥?| 閲囩敤鐣欎竴瑙嗚鐙珛閲嶅缓骞跺洖鎶曟楠岋紝鎾ゅ洖閲嶅宸﹀彸鑲綋鎴栧急瀹炰緥缃俊搴︼紱鍙挙鍥炲垎鏁帮紝涓嶄吉閫犲潗鏍囷紙瑙?[瑙傛祴鎭㈠涓庢寔缁及璁(observation_recovery_research_2026-10-18.md)锛夈€?|
| `multiview_temporal.py` | 璺ㄨ瑙?2D 鐐广€佹爣瀹氥€佷笂涓€甯т笁缁寸姸鎬?| 褰撳墠涓夌淮瑙傛祴涓庡乏鍙虫爣绛惧垎閰嶇粨鏋?| 鎸夊叧鑺傚锛堣偐銆佽倶銆佽厱銆侀珛銆佽啙銆佽笣锛夌嫭绔嬭瘎浼版瀬绾胯窛绂讳笌鏈夌晫鍘嗗彶鎶曞奖锛岄槻姝㈠崟鐐归敊璇繛甯︾炕杞暣鏉¤偄浣擄紱鍏ㄨ瑙掑叡鍚岃涔変簰鎹㈡爣涓烘涔夈€?|
| `auxiliary.py` | 韬綋 17 鐐逛笌鎵嬭冻 16 鐐癸紙2D 鍍忕礌涓?3D 涓栫晫鍧愭爣锛?| 鎵?鑴氭暣缁勭疆鎹㈡牎姝ｅ強涓夌淮闄勭潃涓€鑷存€ц瘖鏂?| 妫€鏌ユ墜鑵曗€旀帉鏍广€佽剼韪濃€旇剼璺?鑴氳毒鐨勭墿鐞嗛檮鐫€璺濈涓婄晫锛屾嫆缁濊劚鑺傜殑杩滅鍣偣锛堝叚鍦烘櫙鎵嬭厱鈥旀帉鏍归棿璺?P95 鐢?`154.46 mm` 闄嶈嚦 `90.82 mm`锛夈€?|

### 3. 澶氳瑙掍笁缁翠笁瑙掑寲涓庡嚑浣曚笉纭畾鎬э紙`pose_sim/geometry.py`銆乣uncertainty.py`锛?
- **姹傝В娴佺▼**锛氱浉鏈哄 RANSAC 鍊欓€夌瓫閫?$\rightarrow$ 缃俊搴﹀姞鏉?DLT 鍒濆€?$\rightarrow$ 鍗曡皟绾挎悳绱?soft-L1 椴佹閲嶆姇褰变紭鍖栵紙鏈敹鏁涙椂鍥為€€ SciPy锛夈€?- **鍑犱綍绾︽潫**锛氫弗鏍兼鏌ョ浉鏈哄墠鏂规娣卞害锛?Z_{\text{cam}} > 0$锛変笌鏈€灏忚宸皠绾垮す瑙掞紙榛樿 $\ge 5^\circ$锛夈€?- **涓嶇‘瀹氭€ч噺鍖?*锛氬湪鍋囪鍐呯偣鍍忕礌鐙珛鍚勫悜鍚屾€у櫔澹帮紙$\sigma = 2\text{ px}$锛変笅锛岀敱灞€閮?Jacobian 璁＄畻 $\Sigma = \sigma^2 (J^\top J)^{-1}$ 杈撳嚭 $3\times 3$ 浣嶇疆鍗忔柟宸煩闃点€傝鍊煎弽鏄犺瑙掑嚑浣曟潯浠讹紝涓嶇瓑鍚屼簬鍚爣瀹氳宸殑鏍″噯缃俊鍖洪棿銆?
### 4. 鍥犳灉浜轰綋鐘舵€佷及璁′笌杩炵画绐楀彛鎷熷悎

| 妯″潡 | 鐘舵€佺増鏈笌鏍稿績鏂规硶 | 鍏抽敭鐗规€т笌杈圭晫 |
|---|---|---|
| `human_state.py`<br>`human_motion.py`<br>`body_shape.py` | `human_state_v1` / `human_state_v2`<br>鍥犳灉 Kalman + 婊氬姩绋冲仴楠ㄩ暱绾︽潫 | 姣忎釜鍏宠妭鐐规槑纭爣璁?`filtered`锛堝綋鍓嶅抚鏈夎娴嬶級銆乣predicted`锛堜粎鍘嗗彶澶栨帹锛屾渶闀?`100 ms`锛夋垨 `missing`锛堣繃鏈?鏃犲巻鍙诧紝鍧愭爣涓?`null`锛夈€傝緭鍑鸿函骞层€佽冻鎺屻€佹墜鎺屽眬閮ㄦ爣蹇楃偣鍧愭爣绯伙紝杞村悜绾壄杞繚鎸佹湭鐭ャ€?|
| `continuous_state.py` | `A2` 杩炵画鍥犳灉绐楀彛鐘舵€?br>12 甯у洜鏋?Huber IRLS + 绔偣楠ㄩ暱鎶曞奖 | 鍦ㄦ嫆缁濇垨鐭椂缂哄け鏈熼棿缁撳悎 12 甯ф粦鍔ㄧ獥鍙ｅ仛绋冲仴澶氶」寮忔嫙鍚堜笌閫愮骇鐖跺瓙楠ㄩ暱鎶曞奖锛屾秷闄よ娴嬫仮澶嶆椂鐨勪綅缃樁璺冿紝渚涗笅娓歌繛缁噸瀹氬悜妯″潡鐢熸垚骞虫粦鍙傝€冿紙瑙?[reports/continuous-prediction-2026-10-20.json](../reports/continuous-prediction-2026-10-20.json)锛夈€?|
| `ground.py`<br>`foot_contact.py` | 鐙珛鑷劧鍦伴潰骞抽潰 + 5 甯у洜鏋滀綅缃獥鍙ｈ冻閮ㄦ帴瑙﹀熀绾?| 鍩轰簬鍦烘櫙鑷劧绾圭悊澶氳瑙掍笁瑙掑寲鎷熷悎鍦伴潰骞抽潰 $n\cdot X + d = 0$锛岀粨鍚堣剼璺?澶у皬鑴氳毒鐨勮繎鍦伴珮搴︺€佺獥鍙ｉ€熷害涓庢嫙鍚堟畫宸緭鍑?`contact / swing / unknown`锛堣 瓒抽儴鎺ヨЕ鍩虹嚎（历史归档 `foot_contact_baseline_2026-10-19.md`）锛夈€?|

### 5. 鍦ㄧ嚎娴佸紡鍚屾涓庡苟鍙戞帹鐞嗭紙`pose_sim/online.py`銆乣streaming_vision.py`锛?
- `ExposureSynchronizer` 涓?`LatestMailbox` 灏嗗悇璺浉鏈烘洕鍏夊抚鎸夊叡鍚屽崟璋冩椂閽熼厤瀵癸紝闃熷垪瀹归噺闄愬埗涓?1锛堝彧澶勭悊鏈€鏂板悓姝ョ粍锛屼富鍔ㄤ涪寮冪Н鍘嬫棫甯э級銆?- 鏀寔 4 璺?YOLOX 妫€娴嬪苟鍙戜笌 4 璺?RTMPose `batch=1` 浼氳瘽骞跺彂锛堜笌鍗曠嚎绋嬮€愮偣杈撳嚭瀹屽叏涓€鑷达級锛屽崟缁勫洓瑙嗚澶勭悊鑰楁椂绾?`85鈥?8 ms`锛堢害 `11.3鈥?1.5 Hz`锛夈€?
---

## 涓夈€?鍊欓€夊疄楠岀粨璁烘眹鎬伙紙鏈浛鎹㈤粯璁ょ殑鎺㈢储锛?
浠ヤ笅鍊欓€夋柟鍚戝凡瀹屾垚鍚屽熀鍑嗗鐓ц瘎娴嬶紝鍥犳湭鍏ㄩ潰瓒呰秺榛樿鍩虹嚎鑰屾湭骞跺叆瀹氬瀷涓荤嚎锛堣瑙?[鐮旂┒鏃ュ織](research_log.md)锛夛細

1. **PMPose-S 姒傜巼濮挎€佸ご瀵圭収**锛氬湪 MPI-INF-3DHP 2,500 甯т笂鍏卞悓鏈夋晥鐐瑰钩鍧囪宸敱 `49.51 mm` 闄嶈嚦 `46.96 mm`锛屼絾瑕嗙洊鐜囩暐闄嶏紝涓旀爣绉?90% 姒傜巼鍖哄煙瀹為檯瑕嗙洊鐜囦粎 `56%鈥?5%`銆?2. **鍒嗗竷铻嶅悎涓庡厜娴佽韩浠藉亣璁?*锛氬厜娴佸浘鍍忚窡韪兘缂撹В鍗曠被宸﹀彸浜掓崲锛屼絾鍦ㄦ甯稿姩浣滀笂鎹熷け瑕嗙洊鐜囷紝涓旂己鍏夋祦璇佹嵁鏃朵細寤剁画閿欒韬唤銆?3. **RTMPose-X 澶фā鍨嬪ご涓庝簲甯ц仈鍚堥噸鎶曞奖绐楀彛**锛氭湭閫氳繃鑰楁椂涓庡叏鐩爣绮惧害鑱斿悎闂ㄦ銆?
---

## 鍥涖€?甯哥敤澶嶇幇鍛戒护

```powershell
$rt = (Get-Content .runtime.json | ConvertFrom-Json).python

# 1. 瀵瑰瑙嗚鍥惧儚浼氳瘽杩愯瀹屾暣妫€娴嬨€佷笁瑙掑寲涓庝汉浣撶姸鎬佷及璁?& $rt scripts/vision_demo.py outputs/public_mamma_session --output outputs/my_vision_run

# 2. 瀵?MPI-INF-3DHP 鐪熷疄澶氳瑙掓暟鎹泦杩愯璇勬祴
& $rt scripts/evaluate_mpi3dhp.py --dataset outputs/public_datasets/mpi_test --output outputs/my_mpi_eval

# 3. 杩愯 12 甯у洜鏋?Huber IRLS 杩炵画鐘舵€佷及璁?& $rt scripts/estimate_continuous_motion.py --input outputs/mainline_validation_20261019/inference/s1_walk/fused.jsonl --output outputs/my_continuous_state
```
