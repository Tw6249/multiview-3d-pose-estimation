# 浜轰綋鍔ㄤ綔鍒板畤鏍?G1 鍏ㄨ韩閲嶅畾鍚戜笌闂幆鎺у埗

鏈枃妗ｈ鏄?`motion_control/` 涓嬩粠涓夌淮浜轰綋杩愬姩鐘舵€佸埌瀹囨爲 G1 EDU锛?9 鑷敱搴︼級鍏ㄨ韩鍙傝€冭建杩圭敓鎴愩€佽繛缁寲绾︽潫鍙婅嚜鐢卞熀搴х墿鐞嗛棴鐜窡韪帶鍒剁殑妯″潡鏋舵瀯銆佹紨杩涚粨璁轰笌澶嶇幇鍛戒护銆?
---

## 涓€銆?鎺у埗涓荤嚎鏋舵瀯锛堝綋鍓嶅畾鍨嬮摼璺級

```text
涓夌淮浜轰綋鐘舵€佹祦 (states3d.jsonl / 12甯у洜鏋?Huber IRLS 杩炵画鐘舵€?A2)
  鈫?1. 楠ㄧ泦璺緞姣斾緥鏄犲皠涓庢牴閮ㄨ窡韪?(root_path.py / root_tracking.py)
鎸変汉涓?G1 鑵块暱姣斾緥鏄犲皠姘村钩浣嶇Щ锛岃浆鎹㈣嚦鏈哄櫒浜哄綋鍓嶈埅鍚戝眬閮ㄥ潗鏍囩郴
  鈫?2. 濂囧紓澹佸瀿鍗曡皟鏌斿寲杩炵画閲嶅畾鍚?(continuous_pose_reference.py / humanoid.py)
杞鍨掓秷闄よ偐/鑲?鑶濆寮傚尯 IK 鍒嗘敮璺冲彉 + 閫熷害鏈夌晫杩炵画姹傝В (25 Hz, 29 鍏宠妭)
  鈫?3. 50 Hz C虏 杩炵画鍙傝€冭建杩圭敓鎴?(continuous_pose_reference.py / reference_trajectory.py)
Ruckig 鍦ㄧ嚎涓夐樁璺冨害鍙楅檺鎻掑€?(浣嶇疆銆侀€熷害銆佸姞閫熷害涓ユ牸杩炵画锛屾渶澶ц穬搴?鈮?1500 rad/s鲁)
  鈫?4. 鍏ㄨ韩鐗╃悊闂幆鎺у埗绛栫暐 (comparison_policies.py / twist.py / humanoid_replay.py)
50 Hz 绁炵粡缃戠粶绛栫暐 (SONIC v1.1 鍘熺敓鍗佸抚绐楀彛 / TWIST2) 鈫?1000 Hz MuJoCo PD 鍔涚煩闂幆
  鈫?5. 鍏峰悕鎸囦护灏佽涓庡畨鍏ㄩ棬鎺?(g1_command.py / stop_transition.py)
甯︽ā鍨?SHA256銆佸簭鍙枫€乀TL 涓庡姏鐭╀笂闄愮殑 29 鍏宠妭鎸囦护 (褰撳墠浠呰繘鍏ユā鎷熸帴鏀跺櫒)
```

---

## 浜屻€?鏍稿績妯″潡涓庢帴鍙ｅ绾?
| 妯″潡 | 杈撳叆涓庡崟浣?| 杈撳嚭 | 鍏抽敭鏈哄埗涓庡け璐ヨ竟鐣?|
|---|---|---|---|
| `motion_control/humanoid.py` | 涓夌淮浜轰綋鍏宠妭鐐癸紙`m`锛変笌鏈夋晥鎬ф帺鐮?| 29 鍏宠妭鍏ㄨ韩鍙傝€冭锛坄rad`锛変笌 35 缁?TWIST/SONIC 鍙傝€冨抚 | 鎻愬彇鍙屼晶鑲┿€佽倶銆佽厱銆侀珛銆佽啙銆佽笣鍏?12 鐐硅偄娈垫柟鍚戯紝鎸夋満鍣ㄤ汉鑷韩 FK 鍖归厤鏂瑰悜锛涘乏鍙宠偄浣撲弗鏍间竴涓€瀵瑰簲锛屾鎬佸畬鍏ㄦ潵鑷簮鑵块儴涓庨鐩嗚繍鍔ㄣ€?|
| `motion_control/continuous_pose_reference.py` | 12 甯у洜鏋?IRLS 杩炵画涓夌淮鐘舵€佹祦锛坄A2`锛?| 25 Hz 杩炵画閲嶅畾鍚戝簭鍒椾笌 50 Hz `BRM-C2` 杩炵画鍙傝€冩祦 | **濂囧紓澹佸瀿鍗曡皟鏌斿寲杩炵画閲嶅畾鍚戯紙BRM-C2锛?*锛氬湪鑲╄倶鎺ヨ繎鍏辩嚎绛夊寮傚尯寮曞叆C虏鍗曡皟杞鍨掍笌涓婁竴甯цВ鐑惎鍔紝娑堥櫎閫愬抚 IK 鍒嗘敮缈昏浆锛堣璧拌浆韬抚闂存渶澶у叧鑺傝烦鍙樼敱 `98.1掳` 闄嶈嚦 `13.8掳`锛屽噯鍏ョ巼杈?`100%`锛夈€?|
| `motion_control/root_path.py`<br>`motion_control/root_tracking.py` | 婧愰鐩嗕笘鐣屼綅绉伙紙`m`锛夈€佷汉鏈鸿吙闀挎瘮銆佹満鍣ㄤ汉涓栫晫閲岀▼璁?| 鐩爣楠ㄧ泦姘村钩璺緞锛坄m`锛変笌鏈哄櫒浜哄眬閮ㄨ埅鍚戝潗鏍囩郴涓嬬殑閫熷害/鍋忚埅绾犲亸鎸囦护 | 浠ラ涓彲闈犻鐩嗕负鍘熺偣鎸夎吙闀挎瘮渚嬬缉鏀惧閲忎綅绉伙紱鍏堝湪瀵归綈鐨勪笘鐣岀郴涓嬪悎鎴愬墠棣堜笌鍙嶉閫熷害锛屽啀鏃嬭浆鑷虫満鍣ㄤ汉褰撳墠鑸悜绯汇€?|
| `motion_control/twist.py`<br>`motion_control/comparison_policies.py` | 50 Hz 鍙傝€冪獥鍙ｃ€佹満鍣ㄤ汉鏈綋 IMU 涓?29 鍏宠妭浣嶇疆/閫熷害鍙嶉 | 29 鍏宠妭 PD 鐩爣浣嶇疆锛坄rad`锛夛紝缁?1000 Hz PD 寰嬭浆鎹负鍏宠妭鍔涚煩锛坄N路m`锛?| 灏佽鍥哄畾鏉冮噸鐨?**TWIST2** 涓?**SONIC v1.1**锛堝惈鍘熺敓闂撮殧鍗佸抚鏈潵/褰撳墠绐楀彛涓庣洿鎺ュ綋鍓嶅弬鑰冩ā寮忥級鍙?**BeyondMimic** 瀵圭収鎺ュ彛銆?|
| `motion_control/visual_contact.py`<br>`motion_control/contact_reference.py` | 瑙嗚瓒抽儴鎺ヨЕ鍖咃紙`contact/swing/unknown`锛変笌鍏ㄨ韩鍙傝€?| 瓒冲簳闈炵┛閫忎笌鏀拺閿氬畾淇鍚庣殑鑵块儴鍏宠妭瑙掍笌鍩哄骇楂樺害 | 瀹為獙鎬ф帴瑙︽姇褰辨ā鍧楋細鏈煡鐘舵€佺粷涓嶉攣鑴氾紱鍦?MPI 琛岃蛋娈典笂鍙皢楠ㄧ泦璺緞璇樊 P95 鐢?`3.196 m` 闄嶈嚦 `2.396 m`锛堣 瓒抽儴鎺ヨЕ鍩虹嚎（历史归档 `foot_contact_baseline_2026-10-19.md`）锛夈€?|
| `motion_control/stop_transition.py`<br>`motion_control/recovery_rollout.py` | 褰撳墠宸插簲鐢ㄥ弬鑰冦€佹満鍣ㄤ汉鏀拺鍙嶉涓庡け姝?缁撴潫淇″彿 | 骞虫粦鎾ゅ洖绔欏Э鐨勮繃娓″弬鑰冨簭鍒?| 澶勭悊瑙傛祴涓柇鎴栫墖娈电粨鏉熸椂鐨勫钩绋冲仠鏈猴紱閬垮厤楂橀€熻繍鍔ㄤ腑鐬椂褰掗浂閫熷害瀵艰嚧鍓嶅€捐穼鍊掋€?|
| `motion_control/g1_command.py` | 29 鍏宠妭鐩爣銆佹ā鍨嬮厤缃笌鍗曡皟鏃堕挓 `ns` | `JointCommand`锛堝啓鍏?`g1_commands.jsonl`锛?| 鏍￠獙妯″瀷 SHA256銆佸叧鑺傚悕绉伴『搴忋€侀檺浣嶃€佹湁闄愭暟鍊间笌鏈夋晥鏈?TTL锛沗hardware_enabled=false`銆?|

---

## 涓夈€?鍏抽敭绉戝闂涓庡疄娴嬫紨杩涚粨璁?
### 1. 涓轰粈涔堟棭鏈熼€愬抚 IK + 闂ㄦ帶鍦ㄧ湡瀹炶棰戜笂棰戠箒涓柇锛?鍦?鍙傝€冨姩浣滀笉杩炵画鍒嗗眰璇婃柇（历史归档 `reference-continuity-2026-10-20.json`） 涓紝鎴戜滑灏嗗弬鑰冭建杩规媶鍒嗕负鈥滄簮閫愬抚 IK鈥濄€佲€滃噯鍏ラ棬鎺ф挙鍥炩€濆拰鈥滀笅娓搁檺閫熷櫒鈥濅笁灞傦紝瀹氫綅鍑轰笁澶ф牴婧愶細
1. **瑙傛祴缂烘祴涓庢仮澶嶉樁璺?*锛氬崟鐐圭疆淇″害娉㈠姩瑙﹀彂闂ㄦ帶鎾ゅ洖锛堜緥濡?`s1_walk` 150 甯т粎鎺ュ彈 107 甯э紝浜х敓 9 娆＄墖娈靛唴鎾ゅ洖鍥炵珯濮夸簨浠讹級锛?2. **閫愬抚鐙珛 IK 鍒嗘敮璺冲彉**锛氭墜鑷傛憜杩囪韩浣撴渚ф柟鎴栨繁韫叉椂锛屽眬閮ㄩ潪鍑?IK 鍦ㄧ浉閭讳袱甯ц烦鍒颁笉鍚屽眬閮ㄦ瀬灏忓€硷紙鐩搁偦 25 Hz 甯ф渶澶ц烦鍙樿揪 `98.1掳鈥?45.4掳`锛夛紱
3. **涓€闃堕檺閫熷櫒寮曡捣鐨勯珮棰戝姞閫熷害/璺冨害鍐插嚮**锛氫笅娓哥畝鍗曢檺閫熷櫒鍦ㄦ姌绾挎嫄瑙掑浜х敓楂樿揪 `600 rad/s虏` 鐨勫姞閫熷害璺冲彉涓?`45,000 rad/s鲁` 鐨勫啿鍑伙紙Jerk锛夛紝鐩存帴鐮村潖寮哄寲瀛︿範鎺у埗鍣ㄧ殑骞宠　瑁曞害銆?
### 2. BRM-C2 杩炵画閲嶅畾鍚?+ SONIC / TWIST2 闂幆瀵规瘮缁撴灉
閫氳繃寮曞叆 **12 甯у洜鏋?Huber IRLS 鐘舵€佷及璁?+ 濂囧紓澹佸瀿鍗曡皟鏌斿寲杩炵画閲嶅畾鍚戯紙BRM-C2锛? 50 Hz Ruckig $C^2$ 杞ㄨ抗娴?*锛堣 [continuous-reference-twist-vs-sonic-2026-10-21.json](../reports/continuous-reference-twist-vs-sonic-2026-10-21.json) 鍙?`outputs/continuous_v3_20261023`锛夛細
- **鍙傝€冭繛缁€?*锛氫簲绫荤湡瀹炰汉浣撹嚜鐒跺姩浣滐紙`s1_walk` 琛岃蛋杞韩銆乣s1_crouch` 蹇€熻捣韫层€乣s1_exercise` 娣辫共浣撴搷銆乣s2_misc` 鏃ュ父杞綋鎸ヨ噦銆乣s2_sports` 澶ц法搴︿綋鑲插姩浣滐級甯у噯鍏ョ巼鍏ㄩ儴杈惧埌 **100%锛?50/150 甯э紝0 娆＄墖娈靛唴涓柇锛?*锛涘弬鑰冭建杩规渶澶ц穬搴﹀彈鎺у湪 **$\le 1500\text{ rad/s}^3$**锛屾繁韫查鐩嗘渶浣庨珮搴﹁揪 `0.488 m`锛堢洰鏍?`0.485 m`锛夈€?- **闂幆鍔ㄥ姏瀛﹀钩琛★紙120 ms 鍥犳灉寤惰繜锛孧uJoCo 1000 Hz锛?*锛?  - 浣跨敤鍘熺敓 **SONIC v1.1** 鎺у埗绛栫暐锛堟棤浠讳綍鐗╃悊鍙傛暟绡℃敼鎴栬緭鍑洪挸浣嶏級锛?*浜旂被鑷劧鍔ㄤ綔搴忓垪鍏ㄩ儴 100% 淇濇寔鐩寸珛骞宠　骞跺钩绋冲仠绋?*锛?  - 浜旂被搴忓垪骞冲潎鍏ㄨ韩鍏宠妭璺熻釜 RMSE 涓?**19.32掳**锛屼笅鑲㈠叧鑺?RMSE 涓?**13.37掳**锛屾按骞宠矾寰勫亸宸?P95 绾?**0.61 m**锛?  - 鐩告瘮涔嬩笅锛?*TWIST2** 鍦ㄥ叧鑺傚眬閮ㄨ窡韪宸笂鐣ヤ綆锛屼絾鍦ㄥ揩閫熻浆韬垨鍋滄闃舵鏇村鏄撳洜缂轰箯鏈潵绐楀彛鑰屽€惧€掞紙璇﹁ 鎺у埗鍣ㄥ鐓ф姤鍛奭(../reports/controller-comparison-2026-10-20.json) 涓?[TWIST2 骞宠　璇婃柇（历史归档 `twist-balance-diagnosis-2026-10-20.json`）锛夈€?
### 3. 褰撳墠灏氭湭瑙ｅ喅鐨勯仐鐣欓棶棰?1. **闀胯窛绂绘按骞充綅缃疮绉紓绉?*锛氱敱浜庤嚜鐢卞熀搴т豢鐪熶腑鏀拺鑴氬瓨鍦ㄥ井绫宠嚦姣背绾у垏鍚戝井婊戠Щ锛岄暱璺濈琛岃蛋杞韩锛坄s1_walk`锛夋按骞宠矾寰?P95 鍋忓樊绾?`0.83 m`锛涗笅涓€姝ラ渶缁撳悎瑙嗚鑴氳窡/鑴氳毒鏀拺鏈熻瘑鍒繘琛屾敮鎾戣剼浣嶇Щ閿氬畾銆?2. **蹇€熻捣韫茬浉浣嶆粸鍚?*锛歚s1_crouch` 鍦?120 ms 鍥犳灉寤惰繜涓庡浐瀹氳穬搴︿笂闄愪笅鍏ㄨ韩鍏宠妭璇樊涓?`21.1掳`锛涗笅涓€姝ユ嫙楠岃瘉鎸夎窡韪畫宸嚜閫傚簲璋冨害鐨勮穬搴︿笂闄愩€?3. **鍓嶈噦鑷棆涓庢墜鑵曡嚜鐢卞害涓嶅彲瑙傛祴**锛氱幇鏈?23/33 鐐圭█鐤忓叧閿偣鏃犳硶鍞竴纭畾鍓嶈噦杞村悜鑷棆锛? 涓厱閮ㄨ嚜鐢卞害鐩墠淇濇寔涓€у厛楠屻€?4. **灏氭湭鍦ㄧ墿鐞?G1 瀹炴満楠岃瘉**锛氭墍鏈夊姩鍔涘缁撹鍧囬檺浜?MuJoCo 浠跨湡锛屽疄鏈鸿仈璋冨墠缃姹傝 [绯荤粺鏋舵瀯涓庡疄鏈鸿縼绉昏竟鐣宂(architecture.md#涓?瀹炴満纭欢杩佺Щ鍓嶇疆娓呭崟)銆?
---

## 鍥涖€?澶嶇幇鍛戒护

```powershell
$robotPython = (Get-Content .robot-runtime.json | ConvertFrom-Json).python

# 1. 杩愯 TWIST2 鏈綋鎺у埗鍩哄噯锛堢珯绔嬨€佹姉鎺ㄦ壈銆佷笂鑲€佸畼鏂瑰弬鑰冦€佽瑙夊弬鑰冿級
& $robotPython scripts/benchmark_twist_policy.py --output outputs/my_twist_eval

# 2. 杩愯 TWIST2 / SONIC / BeyondMimic 澶氭帶鍒跺櫒鍚屽熀鍑嗗鐓?& $robotPython scripts/benchmark_controllers.py --output outputs/my_controller_compare

# 3. 杩愯杩炵画鍙傝€冿紙A0 / A2_gated / A2_continuous锛夊湪 TWIST2 涓?SONIC 涓嬬殑闂幆璇勬祴
& $robotPython scripts/benchmark_continuous_twist_sonic.py --output outputs/my_continuous_twist_sonic

# 4. 杩愯鏈€鏂?BRM-C2 杩炵画閲嶅畾鍚戜笌浜旂被搴忓垪闂幆楠岃瘉鍙婅棰戠敓鎴?& $robotPython scripts/benchmark_continuous_v3.py --output outputs/my_continuous_v3
& $robotPython scripts/visualize_continuous_v3.py --input outputs/my_continuous_v3
```

---

## 浜斻€?澶栭儴鎺у埗鍣ㄨ祫浜т笌寮€婧愯鍙?
- **TWIST2**锛歨ttps://github.com/amazon-far/TWIST2 锛圡IT License锛岃 `assets/humanoid/TWIST2-LICENSE.txt`锛?- **SONIC**锛歨ttps://github.com/NVlabs/SONIC 锛堣 `assets/humanoid/controller_comparison/SONIC-LICENSE.txt`锛?- **RoboJuDo / BeyondMimic**锛氳 `assets/humanoid/controller_comparison/RoboJuDo-LICENSE.txt` 涓?`assets/humanoid/controller_comparison_manifest.json`
- **Unitree G1 URDF**锛氳 `assets/unitree/LICENSE` 涓?`assets/unitree/manifest.json`
