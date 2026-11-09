# -*- coding: utf-8 -*-
"""Generate high-resolution physical equipment & wiring visual guide plates for the deployment manual."""

import os
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

RAW_DIR = "data/deployment_bundle/docs/equipment_photos"
OUT_DIR = "docs/figures"
os.makedirs(OUT_DIR, exist_ok=True)


def load_img_contain(path: str, canvas_w: int, canvas_h: int, bg=(255, 255, 255)) -> np.ndarray:
    """Load an image and fit it cleanly onto a white RGB canvas of (canvas_w, canvas_h)."""
    im = Image.open(path)
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        im = im.convert("RGBA")
        white_bg = Image.new("RGBA", im.size, (*bg, 255))
        im = Image.alpha_composite(white_bg, im).convert("RGB")
    else:
        im = im.convert("RGB")

    arr = np.asarray(im)
    mask = np.any(arr < 248, axis=-1)
    if np.any(mask):
        ys, xs = np.where(mask)
        pad = 8
        y0, y1 = max(0, ys.min() - pad), min(im.height, ys.max() + pad)
        x0, x1 = max(0, xs.min() - pad), min(im.width, xs.max() + pad)
        im = im.crop((x0, y0, x1, y1))

    im.thumbnail((int(canvas_w * 0.92), int(canvas_h * 0.92)), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (canvas_w, canvas_h), bg)
    ox = (canvas_w - im.width) // 2
    oy = (canvas_h - im.height) // 2
    canvas.paste(im, (ox, oy))
    return np.asarray(canvas)


def draw_equip1_bom_gallery():
    """Plate 1: 6-Card Physical Hardware BOM Gallery with Real Product Photos & Key Specs."""
    fig = plt.figure(figsize=(18, 11.2), dpi=220, facecolor="white")
    ax_main = fig.add_axes([0, 0, 1, 1])
    ax_main.set_xlim(0, 180)
    ax_main.set_ylim(0, 112)
    ax_main.axis("off")

    ax_main.text(
        4.0, 107.2,
        "自建多相机硬件方案：核心设备实物图鉴与选型清单（BOM Visual Guide）",
        fontsize=17, fontweight="bold", color="#0F172A"
    )
    ax_main.text(
        4.0, 103.8,
        "下图展示本方案采购与部署的 6 类核心物理设备实物外观、型号规格及在系统中的工程角色",
        fontsize=10.5, color="#475569"
    )
    ax_main.plot([4.0, 176.0], [102.0, 102.0], color="#CBD5E1", lw=1.2)

    cards = [
        {
            "badge": "设备 01 | 视觉采集前端 × 4~6 台",
            "badge_color": "#1D4ED8",
            "title": "海康全局快门工业相机 MV-CS020-10GC",
            "img": os.path.join(RAW_DIR, "hik_cs020_photo.png"),
            "specs": [
                "• 传感器：Sony IMX430（1/1.7\" 全局快门 CMOS，无果冻效应）",
                "• 分辨率 / 帧率：1624 × 1240 @ 50 Hz（最高 60 fps BayerRG8）",
                "• 物理接口：背部千兆 RJ45（支持 802.3af PoE 供电）+ 6 芯 P7 航插",
                "• 机械体积：29 × 29 × 42 mm（约 100 g），四面带 M3 安装螺孔",
            ],
            "note": "关键设定：PixelFormat=BayerRG8、ExposureTime=1800 us",
            "note_bg": "#EFF6FF", "note_fg": "#1E40AF"
        },
        {
            "badge": "设备 02 | 光学成像镜头 × 4~6 支",
            "badge_color": "#0F766E",
            "title": "海康 6 mm C口工业定焦镜头 MVL-KF0618M-12MPE",
            "img": os.path.join(RAW_DIR, "lens_kf0618m.jpg"),
            "specs": [
                "• 焦距 / 靶面：f = 6 mm 广角定焦，1/1.7\" 大靶面、1200 万像素解析度",
                "• 视场角（FOV）：水平 62.7° × 垂直 49.9°（距中心 3.0 m 覆盖高 2.8 m）",
                "• 光圈范围：F1.8 ~ F16 手动调节（实机建议设于 F2.0 ~ F2.8）",
                "• 锁紧机构：镜筒带对焦环手拧锁紧螺丝 + 光圈环手拧锁紧螺丝",
            ],
            "note": "关键设定：标定前必须拧紧两颗机械锁紧螺丝，启用 5 参数去畸变",
            "note_bg": "#F0FDFA", "note_fg": "#115E59"
        },
        {
            "badge": "设备 03 | 硬件同步触发源 × 1 套 + 6 芯航插线 × 4~6 根",
            "badge_color": "#B45309",
            "title": "50 Hz 脉冲发生器 (XY-LPWM/STM32) + HR10A-7P-6S 触发线",
            "img_pair": (
                os.path.join(RAW_DIR, "xy_lpwm_module2.jpg"),
                os.path.join(RAW_DIR, "hirose_6pin_io.jpg"),
            ),
            "specs": [
                "• 触发发生器：输出 50.00 Hz（周期 20.000 ms）、10% 占空比方波",
                "• 驱动电平：5 V ~ 12 V 推挽/MOSFET 并联驱动各相机光耦输入 Line0",
                "• 6 芯航插线：一端 HR10A-7P-6S 航空插头插相机，一端 6 色散线接端子",
                "• 核心接线：黄线 (Pin 2, Line0+) 接脉冲+，绿线 (Pin 5, Line0-) 接 GND",
            ],
            "note": "关键设定：严禁用 3.3V 单片机无驱动直连多台光耦；橙线(电源)悬空绝缘",
            "note_bg": "#FFFBEB", "note_fg": "#92400E"
        },
        {
            "badge": "设备 04 | 供电与万兆汇聚核心 × 1 台",
            "badge_color": "#0369A1",
            "title": "H3C 24口千兆 PoE+ 万兆上联交换机 (S5120V3 / S5136S)",
            "img": os.path.join(RAW_DIR, "h3c_poe_switch2.jpg"),
            "specs": [
                "• 下联端口：24 × 1000BASE-T PoE+ 电口（单口最高 30W，相机仅需 3W）",
                "• 上联端口：4 × 10G SFP+ 万兆口（配万兆电口模块或光纤连主机网卡 1）",
                "• 带宽余量：6 台 50 Hz BayerRG8 合计 4.83 Gbps，万兆上联余量 > 2 倍",
                "• 交换特性：全线速无阻塞转发，支持 9 KB 巨型帧（Jumbo Frame）",
            ],
            "note": "关键设定：一根 Cat6A 网线同时解决单台相机的 PoE 供电与图像传输",
            "note_bg": "#F0F9FF", "note_fg": "#075985"
        },
        {
            "badge": "设备 05 | 工作站双网物理隔离接口 × 1 块",
            "badge_color": "#4338CA",
            "title": "Intel X550-T1 PCIe 万兆网卡（配合主板千兆口组成双网）",
            "img": os.path.join(RAW_DIR, "intel_x550.jpg"),
            "specs": [
                "• 网卡 1（Intel X550 万兆口）：静态 IP 192.168.10.1/24，专收相机图像",
                "• 网卡 1 必配：开启 9014 Bytes 巨帧、RX Buffer=4096、挂载海康巨帧驱动",
                "• 网卡 2（主板自带千兆口）：静态 IP 192.168.123.99/24，独立网线直连 G1",
                "• 隔离目的：彻底杜绝 4.83 Gbps 视频流微突发冲垮 500 Hz 机器人控制包",
            ],
            "note": "关键设定：视觉网与机器人控制网在物理网卡层彻底分开，严禁共网混接",
            "note_bg": "#EEF2FF", "note_fg": "#3730A3"
        },
        {
            "badge": "设备 06 | 动作执行终端与几何基准 × 1 套",
            "badge_color": "#BE123C",
            "title": "宇树 Unitree G1 EDU (29-DoF) + A1 哑光铝塑 ChArUco 标定板",
            "img_pair": (
                os.path.join(RAW_DIR, "unitree_g1_official.png"),
                "data/deployment_bundle/calibration/charuco_a1_12x9_60mm_300dpi.png",
            ),
            "specs": [
                "• 机器人本体：宇树 G1 EDU（29 自由度全身受控：双腿 12 + 腰 3 + 双臂 14）",
                "• 通信协议：unitree_sdk2（CycloneDDS unitree_hg::LowCmd/LowState @ 500 Hz）",
                "• 空间标定板：841 × 594 mm（A1 横版），12×9 格（方格 60 mm / ArUco 45 mm）",
                "• 标定板材质：5 mm 加厚铝塑复合板 + 哑光裱膜（防补光灯镜面反光）",
            ],
            "note": "关键设定：标定板平铺地面强制对齐世界坐标系 Z=0；实机调试配龙门防坠吊架",
            "note_bg": "#FFF1F2", "note_fg": "#9F1239"
        },
    ]

    col_xs = [4.0, 62.0, 120.0]
    row_ys = [52.5, 3.0]
    card_w, card_h = 56.0, 47.0

    for idx, c in enumerate(cards):
        col = idx % 3
        row = idx // 3
        x0 = col_xs[col]
        y0 = row_ys[row]

        box = FancyBboxPatch(
            (x0, y0), card_w, card_h,
            boxstyle="round,pad=0.3,rounding_size=1.2",
            facecolor="#FFFFFF", edgecolor="#CBD5E1", linewidth=1.3
        )
        ax_main.add_patch(box)

        badge_box = FancyBboxPatch(
            (x0 + 1.2, y0 + card_h - 4.3), card_w - 2.4, 3.1,
            boxstyle="round,pad=0.1,rounding_size=0.6",
            facecolor=c["badge_color"], edgecolor="none"
        )
        ax_main.add_patch(badge_box)
        ax_main.text(
            x0 + 2.4, y0 + card_h - 2.75, c["badge"],
            fontsize=9.2, fontweight="bold", color="white", va="center"
        )

        ax_main.text(
            x0 + 1.5, y0 + card_h - 6.8, c["title"],
            fontsize=10.5, fontweight="bold", color="#0F172A", va="center"
        )

        img_box = Rectangle(
            (x0 + 1.5, y0 + 19.8), card_w - 3.0, 18.8,
            facecolor="#F8FAFC", edgecolor="#E2E8F0", linewidth=1.0
        )
        ax_main.add_patch(img_box)

        fig_x = (x0 + 1.8) / 180.0
        fig_y = (y0 + 20.1) / 112.0
        fig_w = (card_w - 3.6) / 180.0
        fig_h = 18.2 / 112.0

        if "img" in c:
            ax_img = fig.add_axes([fig_x, fig_y, fig_w, fig_h])
            arr = load_img_contain(c["img"], 760, 380)
            ax_img.imshow(arr)
            ax_img.axis("off")
        else:
            ax_img1 = fig.add_axes([fig_x, fig_y, fig_w * 0.48, fig_h])
            arr1 = load_img_contain(c["img_pair"][0], 370, 380)
            ax_img1.imshow(arr1)
            ax_img1.axis("off")

            ax_img2 = fig.add_axes([fig_x + fig_w * 0.51, fig_y, fig_w * 0.49, fig_h])
            arr2 = load_img_contain(c["img_pair"][1], 370, 380)
            ax_img2.imshow(arr2)
            ax_img2.axis("off")

        sy = y0 + 17.5
        for line in c["specs"]:
            ax_main.text(
                x0 + 1.6, sy, line,
                fontsize=8.4, color="#334155", va="center"
            )
            sy -= 3.25

        note_box = FancyBboxPatch(
            (x0 + 1.4, y0 + 1.1), card_w - 2.8, 3.2,
            boxstyle="round,pad=0.1,rounding_size=0.5",
            facecolor=c["note_bg"], edgecolor="none"
        )
        ax_main.add_patch(note_box)
        ax_main.text(
            x0 + 2.2, y0 + 2.7, c["note"],
            fontsize=8.1, fontweight="bold", color=c["note_fg"], va="center"
        )

    out_path = os.path.join(OUT_DIR, "equip1_hardware_bom_gallery.png")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    print("Saved:", out_path)


def clean_cad_image(path: str) -> Image.Image:
    """Remove leftover 'and standard. / mension' text in top-left of cropped datasheet CAD image."""
    im = Image.open(path).convert("RGB")
    arr = np.asarray(im).copy()
    h, w, _ = arr.shape
    arr[0:int(h * 0.28), 0:int(w * 0.32)] = 255
    return Image.fromarray(arr)


def draw_equip2_camera_lens_hirose_pinout():
    """Plate 2: Camera Body Ports, Lens Locking Screws, and 6-Pin Hirose Trigger Cable Wiring."""
    fig = plt.figure(figsize=(18, 9.8), dpi=220, facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 180)
    ax.set_ylim(0, 98)
    ax.axis("off")

    ax.text(
        4.0, 93.5,
        "视觉前端实物接口详解：海康相机本体、6mm 镜头锁紧机构与 6 芯航插硬触发接线图",
        fontsize=16.5, fontweight="bold", color="#0F172A"
    )
    ax.text(
        4.0, 90.2,
        "对应手册阶段一（4.1.3 节）与阶段三（4.3.1 节）—— 实操接线前请严格对照背部接口、镜头锁紧螺丝位置与 6 芯线色定义",
        fontsize=10.2, color="#475569"
    )
    ax.plot([4.0, 176.0], [88.5, 88.5], color="#CBD5E1", lw=1.2)

    p1 = FancyBboxPatch(
        (4.0, 3.5), 55.0, 82.8,
        boxstyle="round,pad=0.3,rounding_size=1.2",
        facecolor="#FFFFFF", edgecolor="#CBD5E1", lw=1.3
    )
    ax.add_patch(p1)
    ax.text(6.0, 83.2, "（a）海康 MV-CS020-10GC 正背面接口与机械尺寸", fontsize=11.5, fontweight="bold", color="#1E3A8A")

    ax_cam = fig.add_axes([5.5 / 180, 50.5 / 98, 52.0 / 180, 29.0 / 98])
    ax_cam.imshow(load_img_contain(os.path.join(RAW_DIR, "hik_cs020_photo.png"), 900, 500))
    ax_cam.axis("off")

    callouts_a = [
        ("① 前部 C-Mount 接口", "内嵌 1/1.7\" IMX430 全局快门靶面，直接旋入 6mm 镜头"),
        ("② 背部上方 LAN/POE 口", "千兆 RJ45 网口（带锁扣孔），一根 Cat6A 网线同时供电+传图"),
        ("③ 背部左下 PWR/IO 航插", "6 芯 P7 (HR10A-7R-6P) 公座，接入外部 50 Hz 光耦硬触发信号"),
        ("④ 四面 M3×3 安装螺孔", "支持常规横装（62.7°×49.9°）或旋转 90° 竖装（适配紧凑场地）"),
    ]
    cy = 48.0
    for k_title, k_desc in callouts_a:
        ax.text(6.0, cy, f"{k_title}：{k_desc}", fontsize=8.3, color="#1E293B")
        cy -= 3.2

    ax_cad = fig.add_axes([5.5 / 180, 5.2 / 98, 52.0 / 180, 28.5 / 98])
    cad_clean_path = os.path.join(RAW_DIR, "hik_cs020_cad_clean.png")
    clean_cad_image(os.path.join(RAW_DIR, "hik_cs020_cad.png")).save(cad_clean_path)
    ax_cad.imshow(load_img_contain(cad_clean_path, 900, 490))
    ax_cad.axis("off")

    p2 = FancyBboxPatch(
        (61.5, 3.5), 45.0, 82.8,
        boxstyle="round,pad=0.3,rounding_size=1.2",
        facecolor="#FFFFFF", edgecolor="#CBD5E1", lw=1.3
    )
    ax.add_patch(p2)
    ax.text(63.5, 83.2, "（b）MVL-KF0618M-12MPE 镜头与锁死规程", fontsize=11.5, fontweight="bold", color="#0F766E")

    ax_lens = fig.add_axes([62.8 / 180, 31.5 / 98, 42.4 / 180, 48.0 / 98])
    lens_canvas = load_img_contain(os.path.join(RAW_DIR, "lens_kf0618m.jpg"), 840, 780)
    ax_lens.imshow(lens_canvas)
    ax_lens.set_xlim(0, 840)
    ax_lens.set_ylim(780, 0)
    ax_lens.axis("off")

    ax_lens.annotate(
        "前组广角镜片\n(f=6mm, FOV 62.7°)",
        xy=(340, 120), xytext=(15, 75),
        arrowprops=dict(arrowstyle="->", color="#0F766E", lw=1.8),
        fontsize=8.2, fontweight="bold", color="#0F766E",
        bbox=dict(boxstyle="round,pad=0.3", fc="#F0FDFA", ec="#0F766E", lw=1.1)
    )
    ax_lens.annotate(
        "对焦环 + 锁紧螺丝\n(合焦后拧紧锁死)",
        xy=(495, 340), xytext=(580, 230),
        arrowprops=dict(arrowstyle="->", color="#BE123C", lw=1.8),
        fontsize=8.2, fontweight="bold", color="#BE123C",
        bbox=dict(boxstyle="round,pad=0.3", fc="#FFF1F2", ec="#BE123C", lw=1.1)
    )
    ax_lens.annotate(
        "光圈环 + 锁紧螺丝\n(设 F2.0~2.8 拧死)",
        xy=(485, 470), xytext=(580, 515),
        arrowprops=dict(arrowstyle="->", color="#B45309", lw=1.8),
        fontsize=8.2, fontweight="bold", color="#B45309",
        bbox=dict(boxstyle="round,pad=0.3", fc="#FFFBEB", ec="#B45309", lw=1.1)
    )
    ax_lens.annotate(
        "C-Mount 标准螺纹\n(旋入相机前口)",
        xy=(360, 585), xytext=(15, 595),
        arrowprops=dict(arrowstyle="->", color="#1E3A8A", lw=1.8),
        fontsize=8.2, fontweight="bold", color="#1E3A8A",
        bbox=dict(boxstyle="round,pad=0.3", fc="#EFF6FF", ec="#1E3A8A", lw=1.1)
    )

    lens_rules = [
        "【标定前镜头机械锁死规程】",
        "1. 旋入相机 C 口，开启 MVS 预览（曝光设为 1800 us）；",
        "2. 调光圈环至 F2.0~F2.8（兼顾 1.5~4m 景深），拧紧锁紧螺丝；",
        "3. 对准场地中心（~3.0 m）标定板调焦至锐利，拧紧锁紧螺丝；",
        "4. 在镜筒接缝贴防拆标记胶带，标定后严禁触碰镜头环！",
    ]
    ly = 27.5
    for idx_l, rule in enumerate(lens_rules):
        fw = "bold" if idx_l == 0 else "normal"
        col_c = "#0F766E" if idx_l == 0 else "#1E293B"
        ax.text(63.5, ly, rule, fontsize=8.2, fontweight=fw, color=col_c)
        ly -= 4.2

    p3 = FancyBboxPatch(
        (109.0, 3.5), 67.0, 82.8,
        boxstyle="round,pad=0.3,rounding_size=1.2",
        facecolor="#FFFFFF", edgecolor="#CBD5E1", lw=1.3
    )
    ax.add_patch(p3)
    ax.text(111.0, 83.2, "（c）6 芯航插触发线实物、引脚线色表与 50 Hz 触发器接线", fontsize=11.5, fontweight="bold", color="#9A3412")

    ax_hir = fig.add_axes([110.5 / 180, 56.5 / 98, 31.0 / 180, 23.5 / 98])
    ax_hir.imshow(load_img_contain(os.path.join(RAW_DIR, "hirose_6pin_io.jpg"), 520, 390))
    ax_hir.set_title("6 芯 HR10A-7P-6S 触发线实物", fontsize=8.8, fontweight="bold", color="#334155")
    ax_hir.axis("off")

    ax_pwm = fig.add_axes([143.0 / 180, 56.5 / 98, 31.0 / 180, 23.5 / 98])
    ax_pwm.imshow(load_img_contain(os.path.join(RAW_DIR, "xy_lpwm_module2.jpg"), 520, 390))
    ax_pwm.set_title("50 Hz PWM 硬件脉冲发生器实物", fontsize=8.8, fontweight="bold", color="#334155")
    ax_pwm.axis("off")

    ax.text(111.0, 53.2, "海康 6 芯 P7 航插（MV-CS020-10GC）官方引脚与线色对照表：", fontsize=9.2, fontweight="bold", color="#0F172A")

    headers = ["引脚", "官方信号名", "标准线色", "电气方向", "本方案实操接法（PoE 供电 + Line0 硬触发）"]
    col_x = [111.0, 117.5, 131.0, 140.5, 149.5]
    ax.add_patch(Rectangle((110.5, 48.8), 64.0, 3.2, facecolor="#1E293B", edgecolor="none"))
    for cx, h_text in zip(col_x, headers):
        ax.text(cx, 50.4, h_text, fontsize=8.0, fontweight="bold", color="white", va="center")

    pin_rows = [
        ("Pin 1", "DC_PWR (9~24V)", "橙色 (Orange)", "电源输入", "悬空并用热缩管绝缘（已由网线 PoE 供电）", "#F8FAFC", "#475569", False),
        ("Pin 2", "OPTO_IN (Line0+)", "黄色 (Yellow)", "光耦输入+", "★ 必接：接 5V~12V 50Hz 脉冲源 PWM+ 输出", "#FEF3C7", "#92400E", True),
        ("Pin 3", "GPIO (Line2)", "紫色 (Purple)", "双向非隔离", "悬空不用", "#F8FAFC", "#475569", False),
        ("Pin 4", "OPTO_OUT (Line1+)", "蓝色 (Blue)", "光耦输出+", "选接：接示波器测 Strobe 曝光有效脉冲宽度", "#F8FAFC", "#475569", False),
        ("Pin 5", "OPTO_GND (Line0-)", "绿色 (Green)", "光耦信号地", "★ 必接：接 50Hz 脉冲源 / 光耦驱动板 GND", "#DCFCE7", "#166534", True),
        ("Pin 6", "GND (Power/Line2-)", "灰色 (Gray)", "电源/GPIO地", "悬空不用", "#F8FAFC", "#475569", False),
    ]
    ry = 45.4
    for pin_no, sig, color_str, dir_str, usage, row_bg, row_fg, is_bold in pin_rows:
        ax.add_patch(Rectangle((110.5, ry), 64.0, 3.3, facecolor=row_bg, edgecolor="#E2E8F0", lw=0.6))
        fw = "bold" if is_bold else "normal"
        for cx, val in zip(col_x, [pin_no, sig, color_str, dir_str, usage]):
            ax.text(cx, ry + 1.65, val, fontsize=7.7, fontweight=fw, color=row_fg, va="center")
        ry -= 3.35

    schem_box = FancyBboxPatch(
        (111.0, 5.2), 63.0, 21.8,
        boxstyle="round,pad=0.2,rounding_size=0.8",
        facecolor="#FFFBEB", edgecolor="#F59E0B", lw=1.2
    )
    ax.add_patch(schem_box)
    ax.text(112.5, 24.5, "一分六硬触发并联接线回路（保证 4~6 台相机曝光时差 < 1 us）：", fontsize=9.0, fontweight="bold", color="#92400E")

    ax.add_patch(FancyBboxPatch((112.5, 8.0), 21.0, 14.0, boxstyle="round,pad=0.2", fc="white", ec="#B45309", lw=1.2))
    ax.text(123.0, 19.2, "50 Hz 硬件脉冲源", fontsize=8.8, fontweight="bold", color="#92400E", ha="center")
    ax.text(123.0, 15.8, "XY-LPWM / STM32+MOSFET", fontsize=7.8, color="#334155", ha="center")
    ax.text(123.0, 12.6, "FREQ = 50.0 Hz (T=20ms)", fontsize=8.0, fontweight="bold", color="#1D4ED8", ha="center")
    ax.text(123.0, 9.8, "DUTY = 10% (2ms, 5V~12V)", fontsize=7.8, color="#0F766E", ha="center")

    ax.plot([133.5, 148.0], [18.0, 18.0], color="#D97706", lw=2.5)
    ax.text(140.7, 19.2, "黄色线并联 (Pin 2: Line0+)", fontsize=7.9, fontweight="bold", color="#B45309", ha="center")

    ax.plot([133.5, 148.0], [11.0, 11.0], color="#16A34A", lw=2.5)
    ax.text(140.7, 8.5, "绿色线共地 (Pin 5: Line0-)", fontsize=7.9, fontweight="bold", color="#15803D", ha="center")

    ax.add_patch(FancyBboxPatch((148.0, 7.5), 24.5, 15.0, boxstyle="round,pad=0.2", fc="white", ec="#1E3A8A", lw=1.2))
    ax.text(160.2, 20.0, "相机 #1 ~ #6 内部光耦", fontsize=8.6, fontweight="bold", color="#1E3A8A", ha="center")
    ax.text(160.2, 16.5, "内置限流电阻 + 光电二极管", fontsize=7.8, color="#334155", ha="center")
    ax.text(160.2, 13.2, "导通压降 ~1.3V, 电流 5~10mA", fontsize=7.8, color="#334155", ha="center")
    ax.text(160.2, 9.6, "上升沿触发 1800 us 同步曝光", fontsize=8.0, fontweight="bold", color="#BE123C", ha="center")

    out_path = os.path.join(OUT_DIR, "equip2_camera_lens_hirose_pinout.png")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    print("Saved:", out_path)


def draw_equip3_network_switch_nic_and_g1_rig():
    """Plate 3: Dual-NIC Physical Network Wiring & Unitree G1 3-Stage Safety Rig."""
    fig = plt.figure(figsize=(18, 9.8), dpi=220, facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 180)
    ax.set_ylim(0, 98)
    ax.axis("off")

    ax.text(
        4.0, 93.5,
        "双网物理隔离连线实物图与宇树 G1 三级安全联调台架（Phase 1 / Phase 3 / Phase 5）",
        fontsize=16.5, fontweight="bold", color="#0F172A"
    )
    ax.text(
        4.0, 90.2,
        "上半幅展示从相机、PoE+ 交换机、万兆网卡到机器人的物理端口连线；下半幅展示 A1 标定板平铺 Z=0 对齐与真机三级安全联调规程",
        fontsize=10.2, color="#475569"
    )
    ax.plot([4.0, 176.0], [88.5, 88.5], color="#CBD5E1", lw=1.2)

    top_box = FancyBboxPatch(
        (4.0, 46.5), 172.0, 40.0,
        boxstyle="round,pad=0.3,rounding_size=1.2",
        facecolor="#F8FAFC", edgecolor="#CBD5E1", lw=1.3
    )
    ax.add_patch(top_box)
    ax.text(6.0, 83.8, "（a）双网卡物理隔离端口级实物连线图（视觉万兆网 192.168.10.0/24  vs.  机器人实时控制网 192.168.123.0/24）",
            fontsize=11.5, fontweight="bold", color="#0F172A")

    nodes = [
        (5.5, 49.0, 33.0, 32.5, "① 4~6 台海康相机\nMV-CS020-10GC", os.path.join(RAW_DIR, "hik_cs020_photo.png"),
         "IP: 192.168.10.11 ~ .16\nBayerRG8 | 巨帧 8192B | GevSCPD 错峰", "#EFF6FF", "#1D4ED8"),
        (50.0, 49.0, 36.5, 32.5, "② H3C PoE+ 万兆上联交换机\nS5120V3 / S5136S-24P", os.path.join(RAW_DIR, "h3c_poe_switch2.jpg"),
         "电口 1~6: 802.3af PoE 供电+收图\n万兆口 25: 10GbE 汇聚上联 (4.83 Gbps)", "#F0F9FF", "#0369A1"),
        (98.0, 49.0, 33.5, 32.5, "③ 工作站双网卡接口\n网卡1: Intel X550 万兆 + 网卡2: 千兆", os.path.join(RAW_DIR, "intel_x550.jpg"),
         "网卡 1 (10G): 192.168.10.1 (Jumbo 9014B)\n网卡 2 (1G): 192.168.123.99 (DDS 专网)", "#EEF2FF", "#4338CA"),
        (144.5, 49.0, 29.5, 32.5, "④ 宇树 G1 EDU 真机\n(29-DoF 全身力控)", os.path.join(RAW_DIR, "unitree_g1_official.png"),
         "机体 IP: 192.168.123.161\nunitree_hg LowCmd/LowState @ 500 Hz", "#FFF1F2", "#BE123C"),
    ]

    for nx, ny, nw, nh, ntitle, nimg, nsub, nbg, ncol in nodes:
        ax.add_patch(FancyBboxPatch((nx, ny), nw, nh, boxstyle="round,pad=0.2,rounding_size=0.8", fc="white", ec=ncol, lw=1.4))
        ax.text(nx + nw / 2, ny + nh - 2.6, ntitle, fontsize=9.0, fontweight="bold", color=ncol, ha="center", va="center")
        ax_in = fig.add_axes([(nx + 1.2) / 180, (ny + 7.5) / 98, (nw - 2.4) / 180, (nh - 12.5) / 98])
        ax_in.imshow(load_img_contain(nimg, 500, 320))
        ax_in.axis("off")
        ax.add_patch(Rectangle((nx + 0.8, ny + 0.8), nw - 1.6, 5.8, fc=nbg, ec="none"))
        ax.text(nx + nw / 2, ny + 3.7, nsub, fontsize=7.7, fontweight="bold", color="#1E293B", ha="center", va="center")

    ax.annotate("", xy=(49.8, 65.5), xytext=(38.7, 65.5), arrowprops=dict(arrowstyle="->", lw=2.8, color="#0284C7"))
    ax.text(44.25, 69.5, "4~6 根 Cat6A\nPoE 供电 + GVSP", fontsize=7.9, fontweight="bold", color="#0369A1", ha="center")

    ax.annotate("", xy=(97.8, 65.5), xytext=(86.7, 65.5), arrowprops=dict(arrowstyle="->", lw=3.2, color="#4338CA"))
    ax.text(92.25, 69.5, "10GbE 万兆网线\n接网卡 1 (X550)", fontsize=7.9, fontweight="bold", color="#4338CA", ha="center")

    ax.annotate("", xy=(144.3, 65.5), xytext=(131.7, 65.5), arrowprops=dict(arrowstyle="<->", lw=2.8, color="#BE123C"))
    ax.text(138.0, 69.5, "独立 Cat6A 直连\n接网卡 2 (500Hz)", fontsize=7.9, fontweight="bold", color="#BE123C", ha="center")

    b_left = FancyBboxPatch(
        (4.0, 3.5), 82.0, 40.5,
        boxstyle="round,pad=0.3,rounding_size=1.2",
        facecolor="#FFFFFF", edgecolor="#CBD5E1", lw=1.3
    )
    ax.add_patch(b_left)
    ax.text(6.0, 41.2, "（b）A1 哑光铝塑 ChArUco 标定板实物图纸与地面 Z=0 标定（阶段三）",
            fontsize=11.2, fontweight="bold", color="#0F766E")

    ax_board = fig.add_axes([5.5 / 180, 6.0 / 98, 30.5 / 180, 32.5 / 98])
    ax_board.imshow(load_img_contain("data/deployment_bundle/calibration/charuco_a1_12x9_60mm_300dpi.png", 640, 520))
    ax_board.axis("off")

    board_steps = [
        "【标定板实物规格】",
        "• 图纸路径：data/deployment_bundle/calibration/ (300 DPI 原大图纸)",
        "• 几何规格：A1 横版 (841×594 mm)，12×9 格，方格 60 mm，ArUco 45 mm",
        "• 制作工艺：高精度写真裱于 5 mm 铝塑复合板，表面覆哑光膜防反光",
        "",
        "【三步标定与物理地板 Z=0 强制对齐】",
        "1. 单机内参畸变：在视野四角与中心采集 40~60 帧，解算 K 与 5 参数畸变；",
        "2. 多机联合 BA：在场地中央旋转采集 150~300 组同步帧，全局优化外参；",
        "3. 地面 Z=0 对齐：标定板平铺中央地板 (+X 朝前)，扣除 5 mm 板厚后锁死",
        "   世界系 Z=0；场地四角独立复测要求 |Z| < 3.0 mm，距离误差 < 2.0 mm。",
    ]
    by = 36.8
    for line in board_steps:
        if line == "":
            by -= 1.2
            continue
        is_hdr = line.startswith("【")
        ax.text(37.2, by, line, fontsize=7.9, fontweight="bold" if is_hdr else "normal",
                color="#0F766E" if is_hdr else "#1E293B")
        by -= 3.25

    b_right = FancyBboxPatch(
        (88.5, 3.5), 87.5, 40.5,
        boxstyle="round,pad=0.3,rounding_size=1.2",
        facecolor="#FFFFFF", edgecolor="#CBD5E1", lw=1.3
    )
    ax.add_patch(b_right)
    ax.text(90.5, 41.2, "（c）宇树 G1 真机上电三级安全联调台架与龙门保护吊架示意（阶段五）",
            fontsize=11.2, fontweight="bold", color="#BE123C")

    stages = [
        ("第一级：零增益通信压测", "#EFF6FF", "#1D4ED8", 91.0,
         ["• 状态：相机+GPU 满载运行", "• 电机：Kp = 0, Kd = 0 (不出力)", "• 动作：连续收发 200s LowState", "• 红线：周期偏差 < 0.2 ms", "  P99.9 抖动 < 2.0 ms"]),
        ("第二级：龙门架悬空 10cm 测试", "#FEF3C7", "#B45309", 119.5,
         ["• 状态：吊带挂龙门架，双脚离地 10cm", "• 必测1：2.0s 软离合平滑啮合无抽搐", "• 必测2：单抬左右手脚核对 1:1 非镜像", "• 必测3：拔掉相机万兆网线，验证", "  ≤ 35ms 看门狗熔断回安全站姿"]),
        ("第三级：微松保护绳地面闭环", "#DCFCE7", "#15803D", 148.0,
         ["• 状态：双脚触地，保护绳微松兜底", "  (倾角 > 30° 立刻拉住机身)", "• 顺序：站立啮合 → 上肢挥手转身", "  → 原地深蹲 → 小步幅前后行走", "• 全程专人手握物理急停与阻尼键"]),
    ]

    for st_title, st_bg, st_col, sx, st_lines in stages:
        ax.add_patch(FancyBboxPatch((sx, 5.5), 26.2, 33.0, boxstyle="round,pad=0.2,rounding_size=0.8", fc=st_bg, ec=st_col, lw=1.3))
        ax.text(sx + 13.1, 36.2, st_title, fontsize=8.6, fontweight="bold", color=st_col, ha="center")

        gx = sx + 13.1
        ax.plot([gx - 8, gx - 8, gx + 8, gx + 8], [21.5, 33.8, 33.8, 21.5], color="#475569", lw=2.0)
        ax.plot([gx - 10, gx + 10], [21.5, 21.5], color="#0F172A", lw=1.8)
        if "悬空" in st_title:
            ry_bot = 23.5
            ax.plot([gx, gx], [33.8, 30.8], color="#DC2626", lw=2.0)
            ax.text(gx + 1.2, 22.3, "离地 10cm", fontsize=7.2, fontweight="bold", color="#DC2626")
        elif "地面" in st_title:
            ry_bot = 21.5
            ax.plot([gx, gx - 1.2, gx], [33.8, 31.2, 29.0], color="#16A34A", lw=1.6, ls="--")
            ax.text(gx + 1.5, 31.5, "微松保护绳", fontsize=7.2, fontweight="bold", color="#15803D")
        else:
            ry_bot = 21.5
            ax.plot([gx, gx], [33.8, 29.0], color="#64748B", lw=1.5, ls=":")
            ax.text(gx + 1.5, 31.5, "Kp=Kd=0", fontsize=7.2, fontweight="bold", color="#1D4ED8")

        ax.add_patch(Circle((gx, ry_bot + 6.3), 0.9, fc=st_col, ec="none"))
        ax.add_patch(Rectangle((gx - 1.4, ry_bot + 2.8), 2.8, 2.5, fc=st_col, ec="none"))
        ax.plot([gx - 0.8, gx - 0.9], [ry_bot + 2.8, ry_bot], color=st_col, lw=2.2)
        ax.plot([gx + 0.8, gx + 0.9], [ry_bot + 2.8, ry_bot], color=st_col, lw=2.2)

        ty = 18.5
        for line in st_lines:
            ax.text(sx + 1.0, ty, line, fontsize=7.5, color="#1E293B")
            ty -= 2.85

    out_path = os.path.join(OUT_DIR, "equip3_network_switch_nic_and_g1_rig.png")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    print("Saved:", out_path)


if __name__ == "__main__":
    draw_equip1_bom_gallery()
    draw_equip2_camera_lens_hirose_pinout()
    draw_equip3_network_switch_nic_and_g1_rig()
