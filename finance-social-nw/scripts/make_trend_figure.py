"""Single-column figure: narrative vs. price trends over matched 6-month windows.

Reads analysis/metrics/paper_stats.json (produced by compute_paper_stats.py).
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
S = json.loads((ROOT / "analysis/metrics/paper_stats.json").read_text())
OUT = ROOT / "analysis/figures/narrative_trends"

W = S["six_month_windows"]
G = S["gkg_full"]["windows_gpu_cloud_share"]
labels = ["Nov22", "May23", "Nov23", "May24", "Nov24", "May25", "Nov25", "May26*"]
x = np.arange(len(W))

plt.rcParams.update({"font.size": 7, "font.family": "serif", "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6})
fig, axes = plt.subplots(3, 1, figsize=(3.3, 4.4), sharex=True)
RED, BLUE, GREY, GREEN = "#c0392b", "#2c6fbb", "#555555", "#1e8449"

ax = axes[0]
ax.plot(x, [g["gpu_cloud_share"] * 100 for g in G], "-o", color=RED, ms=3, lw=1.2,
        label="All GKG co-mentions (29,690)")
ax.plot(x, [w["gpu_cloud_total"] / w["n"] * 100 for w in W], "--s", color=GREY, ms=3, lw=1,
        label="Sampled pair relations (2,193)")
ax.set_ylabel("GPU–Cloud share (%)")
ax.set_ylim(0, 50)
ax.legend(frameon=False, loc="lower right", fontsize=6)
ax.set_title("(a) GPU firms enter the Big-Tech narrative", loc="left", fontsize=7)

ax = axes[1]
c = np.array([w["gpu_cloud_causal"] for w in W])
s = np.array([w["gpu_cloud_sub"] for w in W])
ax.bar(x - 0.18, c, 0.36, color=RED, label="causal-driver")
ax.bar(x + 0.18, s, 0.36, color=BLUE, label="substitute")
ax.set_ylabel("GPU–Cloud relations")
ax.legend(frameon=False, loc="upper left", fontsize=6)
ax.set_title("(b) Cross-sector framing is complementary", loc="left", fontsize=7)

ax = axes[2]
ax.plot(x, [w["corr_gpu_cloud"] for w in W], "-o", color=GREEN, ms=3, lw=1.2, label="GPU–Cloud")
ax.plot(x, [w["corr_gpu"] for w in W], "--^", color=GREY, ms=3, lw=1, label="GPU–GPU")
ax.set_ylabel("Mean return corr.")
ax.set_ylim(0, 0.7)
ax.legend(frameon=False, loc="lower left", fontsize=6, ncol=2)
ax.set_title("(c) …but price co-movement does not follow", loc="left", fontsize=7)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=0, fontsize=6)
ax.set_xlabel("6-month window start (*partial)")

for a in axes:
    a.spines[["top", "right"]].set_visible(False)
    a.axvline(0.5, color="#999", lw=0.5, ls=":")
fig.tight_layout(h_pad=0.6)
fig.savefig(f"{OUT}.pdf", bbox_inches="tight")
fig.savefig(f"{OUT}.png", dpi=250, bbox_inches="tight")
print(f"wrote {OUT}.pdf/.png")
