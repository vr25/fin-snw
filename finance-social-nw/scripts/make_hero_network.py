"""Hero figure: narrative co-mention networks before / after ChatGPT.

Three panels on a fixed, sector-clustered layout so the same firm sits in the
same spot in every era:
  (a) Pre-ChatGPT      2022-11-01 .. 2022-11-29
  (b) Early ChatGPT era 2022-11-30 .. 2023-04-30
  (c) Mature AI era     2023-05-01 .. end of corpus

Usage: python scripts/make_hero_network.py
"""
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, Patch

ROOT = Path(__file__).resolve().parents[1]
REL_PATH = ROOT / "data/processed/pair_relations.jsonl"
PRICE_PATH = ROOT / "data/raw/prices.parquet"
OUT_PNG = ROOT / "analysis/figures/hero_network_eras.png"
OUT_PDF = ROOT / "analysis/figures/hero_network_eras.pdf"
OUT_JSON = ROOT / "analysis/metrics/hero_network_eras.json"

LAUNCH = "2022-11-30"
SPLIT = "2023-05-01"

SECTORS = {
    "GPU / Semis": ["NVDA", "AMD", "AVGO", "SMCI"],
    "Cloud / AI Platform": ["MSFT", "GOOGL", "META", "PLTR"],
    "Speculative AI": ["IONQ", "RGTI", "BBAI", "SOUN"],
    "Control": ["KO", "PG", "XOM"],
}
SECTOR_COLOR = {
    "GPU / Semis": "#E4572E",
    "Cloud / AI Platform": "#2E86AB",
    "Speculative AI": "#F2A541",
    "Control": "#7D8491",
}
SECTOR_CENTER = {
    "GPU / Semis": (0.19, 0.66),
    "Cloud / AI Platform": (0.81, 0.66),
    "Speculative AI": (0.22, 0.065),
    "Control": (0.79, 0.065),
}
SECTOR_RADIUS = {"GPU / Semis": 0.17, "Cloud / AI Platform": 0.17,
                 "Speculative AI": 0.12, "Control": 0.105}

REL_STYLE = {
    "causal-driver": dict(color="#C0392B", ls="-", label="Causal-driver"),
    "substitute-comparison": dict(color="#1F5FAD", ls=(0, (4, 2)), label="Substitute"),
    "co-hype": dict(color="#8E44AD", ls=(0, (1, 1.5)), label="Co-hype"),
}

WIN_THRESHOLD = 500.0   # total return (%) -> "AI winner"
LAG_THRESHOLD = 200.0   # total return (%) -> "AI laggard"


def firm_sector(t):
    for s, firms in SECTORS.items():
        if t in firms:
            return s
    return None


def layout():
    pos = {}
    for s, firms in SECTORS.items():
        cx, cy = SECTOR_CENTER[s]
        r = SECTOR_RADIUS[s] * 0.55
        n = len(firms)
        for i, f in enumerate(firms):
            ang = (math.pi / 4 if n == 4 else math.pi / 6) + 2 * math.pi * i / n
            pos[f] = (cx + r * math.cos(ang), cy + r * math.sin(ang))
    return pos


def load_relations():
    rels = [json.loads(l) for l in open(REL_PATH)]
    eras = {
        "pre": [r for r in rels if r["timestamp"][:10] < LAUNCH],
        "early": [r for r in rels if LAUNCH <= r["timestamp"][:10] < SPLIT],
        "late": [r for r in rels if r["timestamp"][:10] >= SPLIT],
    }
    return eras


def load_returns():
    p = pd.read_parquet(PRICE_PATH).sort_values("Date")
    g = p.groupby("ticker")["close"].agg(["first", "last"])
    return ((g["last"] / g["first"] - 1) * 100).to_dict(), p["Date"].min(), p["Date"].max()


def era_graph(rels):
    pair = defaultdict(Counter)
    mentions = Counter()
    for r in rels:
        a, b = sorted((r["ticker_a"], r["ticker_b"]))
        pair[(a, b)][r["relation"]] += 1
        mentions[a] += 1
        mentions[b] += 1
    return pair, mentions


def density(pair, firms):
    n = len(firms)
    e = sum(1 for (a, b) in pair if a in firms and b in firms)
    return e / (n * (n - 1) / 2)


def gpu_cloud_counts(pair):
    gpu, cloud = set(SECTORS["GPU / Semis"]), set(SECTORS["Cloud / AI Platform"])
    c = Counter()
    for (a, b), cnt in pair.items():
        if (a in gpu and b in cloud) or (a in cloud and b in gpu):
            c.update(cnt)
    return c


def draw_panel(ax, pair, mentions, pos, returns, title, subtitle, show_returns, note=None):
    ax.set_xlim(-0.06, 1.06)
    ax.set_ylim(-0.12, 0.97)
    ax.set_aspect("equal")
    ax.axis("off")

    for s, (cx, cy) in SECTOR_CENTER.items():
        ax.add_patch(Circle((cx, cy), SECTOR_RADIUS[s], facecolor=SECTOR_COLOR[s],
                            alpha=0.10, edgecolor=SECTOR_COLOR[s], lw=1.2, zorder=0))
        ax.text(cx, cy + SECTOR_RADIUS[s] + (0.05 if cy > 0.5 else 0.035), s, ha="center", va="bottom",
                fontsize=10, fontweight="bold", color=SECTOR_COLOR[s])

    # Edges: faint grey for pure co-mentions, typed colour for meaningful relations.
    for (a, b), cnt in sorted(pair.items(), key=lambda kv: sum(kv[1].values())):
        (x1, y1), (x2, y2) = pos[a], pos[b]
        meaningful = {k: cnt[k] for k in REL_STYLE if cnt[k] > 0}
        rad = 0.12 if firm_sector(a) == firm_sector(b) else 0.0
        if not meaningful:
            ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-",
                                         connectionstyle=f"arc3,rad={rad}",
                                         color="#9AA0A6", lw=0.7, alpha=0.55, zorder=1))
            continue
        m = sum(meaningful.values())
        dom = max(meaningful, key=meaningful.get)
        st = REL_STYLE[dom]
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-",
                                     connectionstyle=f"arc3,rad={rad}",
                                     color=st["color"], ls=st["ls"],
                                     lw=0.6 + 0.6 * math.log1p(m), alpha=0.85, zorder=2))

    for f, (x, y) in pos.items():
        s = firm_sector(f)
        k = mentions.get(f, 0)
        ret = returns.get(f)
        active = k > 0
        size = 34 + 30 * math.log1p(k) if active else 34
        if not active:
            ring, rw = "#9AA0A6", 1.0
        elif ret is not None and ret >= WIN_THRESHOLD:
            ring, rw = "#D4A017", 2.2
        elif ret is not None and ret < LAG_THRESHOLD and s != "Control":
            ring, rw = "#222222", 1.7
        else:
            ring, rw = "#555555", 0.8
        ax.scatter(x, y, s=size, facecolor=SECTOR_COLOR[s] if active else "white",
                   edgecolor=ring, linewidth=rw, alpha=1.0 if active else 0.9,
                   linestyle="-" if active else (0, (2, 2)), zorder=4)
        label = f
        if show_returns and ret is not None:
            label = f"{f}\n{ret:+.0f}%"
        cx, cy = SECTOR_CENTER[s]
        ang = math.atan2(y - cy, x - cx)
        off = 0.04 + 0.003 * math.log1p(k)
        lx, ly = x + off * math.cos(ang), y + off * math.sin(ang)
        ha = "center" if abs(math.cos(ang)) < 0.3 else ("left" if math.cos(ang) > 0 else "right")
        va = "center" if abs(math.sin(ang)) < 0.3 else ("bottom" if math.sin(ang) > 0 else "top")
        ax.text(lx, ly, label, ha=ha, va=va,
                fontsize=8.0 if show_returns else 8.3, fontweight="bold",
                color="#111111" if active else "#9AA0A6", zorder=5, linespacing=0.95)

    ax.set_title(title, fontsize=13, fontweight="bold", pad=30)
    ax.text(0.5, 0.995, subtitle, ha="center", va="bottom", fontsize=8.6, color="#333333",
            linespacing=1.2)
    if note:
        ax.text(0.5, 0.39, note, ha="center", va="center", fontsize=8.6, style="italic",
                color="#333333", bbox=dict(boxstyle="round,pad=0.35", fc="#FFFBE6",
                                           ec="#D4A017", lw=1), zorder=6)


def main():
    eras = load_relations()
    returns, pmin, pmax = load_returns()
    pos = layout()

    specs = [
        ("pre", "(a) Before ChatGPT", "Nov 1 - Nov 29, 2022"),
        ("early", "(b) Early ChatGPT era", "Nov 30, 2022 - Apr 2023"),
        ("late", "(c) Mature AI era", "May 2023 - Jun 2026"),
    ]

    stats = {}
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 4.05))
    for ax, (key, title, span) in zip(axes, specs):
        rels = eras[key]
        pair, mentions = era_graph(rels)
        types = Counter(r["relation"] for r in rels)
        gc = gpu_cloud_counts(pair)
        st = {
            "n_relations": len(rels),
            "active_firms": len(mentions),
            "edges": len(pair),
            "relation_types": dict(types),
            "density": {s: round(density(pair, f), 3) for s, f in SECTORS.items()},
            "gpu_cloud": {"causal": gc["causal-driver"], "substitute": gc["substitute-comparison"],
                          "co_hype": gc["co-hype"], "all": sum(gc.values())},
        }
        stats[key] = st
        causal_share = 100 * types["causal-driver"] / max(len(rels), 1)
        subtitle = (f"{span}\n{len(rels):,} relations $\\cdot$ {len(pair)} edges $\\cdot$ "
                    f"{len(mentions)}/15 firms $\\cdot$ {causal_share:.0f}% causal")
        note = None
        if key == "pre":
            note = "NVDA & all GPU firms absent:\nnarrative = Big-Tech only"
        elif key == "late":
            note = (f"GPU$\\leftrightarrow$Cloud: {gc['causal-driver']} causal vs "
                    f"{gc['substitute-comparison']} substitute\n"
                    f"GPU density {st['density']['GPU / Semis']:.2f}, "
                    f"Speculative AI density {st['density']['Speculative AI']:.2f}")
        draw_panel(ax, pair, mentions, pos, returns, title, subtitle,
                   show_returns=(key == "late"), note=note)

    for i in range(2):
        fig.add_artist(FancyArrowPatch((0.330 + i * 0.334, 0.55), (0.346 + i * 0.334, 0.55),
                                       transform=fig.transFigure, arrowstyle="-|>",
                                       mutation_scale=16, color="#444444", lw=2))

    handles = [Patch(fc=c, ec="#555555", label=s) for s, c in SECTOR_COLOR.items()]
    handles += [Line2D([0], [0], color=v["color"], ls=v["ls"], lw=2.4, label=v["label"])
                for v in REL_STYLE.values()]
    handles += [
        Line2D([0], [0], color="#9AA0A6", lw=1.2, label="Co-mention only"),
        Line2D([0], [0], marker="o", ls="", ms=11, mfc="white", mec="#D4A017", mew=3,
               label=f"AI winner (>{WIN_THRESHOLD:.0f}%)"),
        Line2D([0], [0], marker="o", ls="", ms=11, mfc="white", mec="#222222", mew=2.2,
               label=f"AI laggard (<{LAG_THRESHOLD:.0f}%)"),
        Line2D([0], [0], marker="o", ls="", ms=11, mfc="white", mec="#9AA0A6", mew=1.2,
               label="Absent"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=8.6, frameon=False,
               bbox_to_anchor=(0.5, -0.01), handlelength=2.2, columnspacing=1.1)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.93, bottom=0.12, wspace=0.10)

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT_PDF, bbox_inches="tight", facecolor="white")
    stats["returns_pct"] = {k: round(v, 1) for k, v in returns.items()}
    stats["price_window"] = [str(pmin.date()), str(pmax.date())]
    OUT_JSON.write_text(json.dumps(stats, indent=2))
    print(json.dumps({k: v for k, v in stats.items() if k in ("pre", "early", "late")}, indent=1))


if __name__ == "__main__":
    main()
