"""Summary figure for the AdvancedTake evaluation (reads metrics.json only; no re-computation).

Inputs : outputs/advanced_take/s2s3/{selfcal,selfcal_lag6}/metrics.json
Output : outputs/advanced_take/figures/s2s3_vs_bvh_summary.png
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "advanced_take"
HELD = ["Take_006", "Take_007", "Take_008"]
C = ["#2a78d6", "#eb6834", "#1baf7a"]          # categorical slots 1-3 (validated all-pairs set)
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"

m0 = json.loads((OUT / "s2s3" / "selfcal" / "metrics.json").read_text())
m6 = json.loads((OUT / "s2s3" / "selfcal_lag6" / "metrics.json").read_text())

plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "text.color": INK, "axes.facecolor": SURF, "figure.facecolor": SURF})
fig, ax = plt.subplots(2, 2, figsize=(11, 7.6), constrained_layout=True)


def style(a, title):
    a.set_title(title, loc="left", fontsize=10, color=INK)
    a.grid(axis="y", color=GRID, lw=0.8)
    a.set_axisbelow(True)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)


# (a) time-lag sweep, held-out takes, raw S2 output
a = ax[0, 0]
for t, c in zip(HELD, C):
    lag = {int(k): v for k, v in m0["takes"][t]["lag_raw"].items()}
    ks = sorted(lag)
    a.plot([k * 8.333 for k in ks], [lag[k] for k in ks], color=c, lw=1.6, label=t.replace("Take_", "Take "))
a.axvline(0, color=GRID, lw=1)
a.axvline(50, color=INK2, lw=1, ls=(0, (3, 3)))
a.text(52, a.get_ylim()[1] * 0.97, "estimated offset\n+50 ms (calibration takes)", fontsize=8, color=INK2, va="top")
a.set_xlabel("BVH read-time shift vs legacy end-alignment (ms)")
a.set_ylabel("world MPJPE (mm)")
a.legend(frameon=False, loc="upper left")
style(a, "A  Time offset alone moves MPJPE by 6–18 mm (held-out, raw S2)")

# (b) per-joint MPJPE after offset compensation, mean of held-out
a = ax[0, 1]
names = list(m6["takes"][HELD[0]]["raw"]["per_joint_mpjpe_mm"])
vals = np.mean([[m6["takes"][t]["raw"]["per_joint_mpjpe_mm"][n] for n in names] for t in HELD], axis=0)
order = np.argsort(vals)
a.barh(np.arange(len(names)), vals[order], color=C[0], height=0.62)
a.set_yticks(np.arange(len(names)), [names[i].replace("_", " ") for i in order])
for i, v in enumerate(vals[order]):
    a.text(v + 0.8, i, f"{v:.0f}", va="center", fontsize=8, color=INK2)
a.set_xlabel("MPJPE (mm), mean of 3 held-out takes, +50 ms offset")
a.grid(axis="x", color=GRID, lw=0.8)
a.grid(axis="y", visible=False)
for s in ("top", "right"):
    a.spines[s].set_visible(False)
a.set_title("B  Error sits in shoulders/hips (landmark ≠ joint centre)", loc="left", fontsize=10)

# (c) joint-angle MAE vs bias, lag-compensated
a = ax[1, 0]
ang = ["l_knee_flex", "r_knee_flex", "l_hip_flex", "r_hip_flex", "l_shoulder_flex", "r_shoulder_flex",
       "l_elbow_flex", "r_elbow_flex"]
mae = np.mean([[m6["takes"][t]["raw"]["joint_angle_mae_deg"][n] for n in ang] for t in HELD], axis=0)
bias = np.mean([[m6["takes"][t]["raw"]["joint_angle_bias_deg"][n] for n in ang] for t in HELD], axis=0)
x = np.arange(len(ang))
a.bar(x - 0.2, mae, 0.38, color=C[0], label="mean abs error")
a.bar(x + 0.2, np.abs(bias), 0.38, color=C[1], label="|systematic bias|")
a.set_xticks(x, [n.replace("_flex", "").replace("_", " ") for n in ang], rotation=30, ha="right")
a.set_ylabel("degrees")
a.legend(frameon=False)
style(a, "C  Joint-angle error (held-out): elbow is mostly a bias")

# (d) S3 filter effect
a = ax[1, 1]
w = 0.38
raw = [m0["takes"][t]["raw"]["world_mpjpe_mm"] for t in HELD]
flt = [m0["takes"][t]["filtered"]["world_mpjpe_mm"] for t in HELD]
x = np.arange(len(HELD))
a.bar(x - w / 2, raw, w, color=C[0], label="S2 raw")
a.bar(x + w / 2, flt, w, color=C[1], label="+ S3 One-Euro (spec params, 30 Hz)")
for i, (r, f) in enumerate(zip(raw, flt)):
    a.text(i - w / 2, r + 1, f"{r:.0f}", ha="center", fontsize=8, color=INK2)
    a.text(i + w / 2, f + 1, f"{f:.0f}", ha="center", fontsize=8, color=INK2)
a.set_xticks(x, [t.replace("Take_", "Take ") for t in HELD])
a.set_ylabel("world MPJPE (mm), offset not compensated")
a.legend(frameon=False, loc="upper left")
style(a, "D  S3 smoothing adds ~100 ms delay and raises error")

(OUT / "figures").mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "figures" / "s2s3_vs_bvh_summary.png", dpi=140)
print("saved", OUT / "figures" / "s2s3_vs_bvh_summary.png")
