#!/usr/bin/env python3
"""
Multi-Model Judging Summary - Grouped Bar Chart (AI preference rates)

- Groups: grok4, claude4sonnet, gpt5
- Bars per group: % AI-preference under Well-written vs Human-written prompts
- CIs: 95% Wilson (articles as sampling unit, n=20), shown as color-matched "lollipop" error bars
- Styling: seaborn, Computer Modern, blue-inspired palette
- Output: executive_summary.png (in the script directory)
"""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.ticker import PercentFormatter
import numpy as np

ADD_CIS = True  # toggle CI overlay

def wilson_ci(k, n, alpha=0.05):
    """Wilson score interval (returns proportion lower/upper)."""
    if n == 0:
        return (0.0, 0.0)
    z = 1.959963984540054  # ~1.96 for 95%
    phat = k / n
    denom = 1.0 + (z**2) / n
    centre = (phat + (z**2) / (2 * n)) / denom
    halfwidth = (z * math.sqrt((phat * (1 - phat) + (z**2) / (4 * n)) / n)) / denom
    return (max(0.0, centre - halfwidth), min(1.0, centre + halfwidth))

def main():
    # ---------------------------
    # Hardcoded counts (n=20 per condition)
    # ---------------------------
    n_articles = 20
    data_counts = {
        "judge": ["grok4", "claude4sonnet", "gpt5"],
        "well_written_ai_k": [14, 15, 9],
        "human_written_ai_k": [3, 2, 0],
    }

    dfc = pd.DataFrame(data_counts)

    # Long-form with percents and Wilson CIs
    rows = []
    for _, r in dfc.iterrows():
        for key, label in [("well_written_ai_k", "Well-written"),
                           ("human_written_ai_k", "Human-written")]:
            k = int(r[key])
            p = 100.0 * k / n_articles
            lo, hi = wilson_ci(k, n_articles)
            rows.append({
                "judge": r["judge"],
                "prompt": label,
                "percent": p,
                "ci_lo_pct": 100.0 * lo,
                "ci_hi_pct": 100.0 * hi,
                "k": k,
                "n": n_articles
            })
    df = pd.DataFrame(rows)

    # ---------------------------
    # Styling
    # ---------------------------
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Computer Modern Roman", "CMU Serif", "DejaVu Serif", "Times New Roman", "Times"]
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["axes.titleweight"] = "semibold"

    sns.set_theme(style="whitegrid", context="talk")

    # === Stronger contrast blues ===
    # Bars: very dark vs light; Lines: slightly darker than bars
    blues = sns.color_palette("Blues", 9)  # indices 0..8
    bar_palette = {
        "Well-written": blues[5],   # darker blue for strong contrast
        "Human-written": blues[2],  # light blue
    }
    line_palette = {
        "Well-written": blues[8],   # darkest for the CI line
        "Human-written": blues[4],  # mid blue for CI line over light bar
    }

    # ---------------------------
    # Figure
    # ---------------------------
    fig, ax = plt.subplots(figsize=(12, 8), dpi=100)
    hue_order = ["Well-written", "Human-written"]
    x_order = ["grok4", "claude4sonnet", "gpt5"]

    # Bars with crisp white edges so lines read cleanly on top
    g = sns.barplot(
        data=df,
        x="judge",
        y="percent",
        hue="prompt",
        order=x_order,
        hue_order=hue_order,
        palette=bar_palette,
        width=0.7,
        ax=ax,
        edgecolor="white",
        linewidth=1.0,
        ci=None
    )

    # Y-axis formatting
    ax.yaxis.set_major_formatter(PercentFormatter(100))
    ax.set_ylim(0, 100)
    ax.set_ylabel("AI preference rate (%)")
    ax.set_xlabel("Judge")

    # Title & caption
    ax.set_title("Multi-Model Judging Summary", pad=14)
    fig.text(
        0.5, 0.93,
        "Generation model: deepseek/deepseek-chat-v3.1 • Articles evaluated: 20 • Max output tokens per call: 4096\n"
        "Bars: % AI preferred. CIs: 95% Wilson across articles (n=20).",
        ha="center", va="center", fontsize=12, color="dimgray"
    )

    # Legend (no title)
    leg = ax.legend(title=None, frameon=False, loc="upper right")
    for txt in leg.get_texts():
        txt.set_fontsize(12)

    # ---------------------------
    # Tasteful, color-matched "lollipop" CIs + labels offset to the right
    # ---------------------------
    if ADD_CIS:
        cat_idx = {cat: i for i, cat in enumerate(x_order)}
        hue_idx = {hue: i for i, hue in enumerate(hue_order)}
        n_hue = len(hue_order)
        bar_width = 0.7
        dodge = bar_width / n_hue

        xticks = ax.get_xticks()
        cap_half = dodge * 0.32
        line_width = 2.4
        alpha_line = 0.95

        for _, row in df.iterrows():
            i = cat_idx[row["judge"]]
            j = hue_idx[row["prompt"]]
            cat_center = xticks[i]
            leftmost = cat_center - (bar_width / 2.0) + (dodge / 2.0)
            x_c = leftmost + j * dodge

            y = row["percent"]
            lo = row["ci_lo_pct"]
            hi = row["ci_hi_pct"]

            col = line_palette[row["prompt"]]

            # CI line and caps
            ax.vlines(x=x_c, ymin=lo, ymax=hi, colors=col, linewidth=line_width, alpha=alpha_line)
            ax.hlines(y=lo, xmin=x_c - cap_half, xmax=x_c + cap_half, colors=col, linewidth=line_width, alpha=alpha_line)
            ax.hlines(y=hi, xmin=x_c - cap_half, xmax=x_c + cap_half, colors=col, linewidth=line_width, alpha=alpha_line)

            # Center marker
            ax.plot([x_c], [y], marker="o", markersize=5,
                    markerfacecolor="white", markeredgecolor=col,
                    markeredgewidth=1.6, zorder=6)

            # ===== Shift label slightly to the right of the lollipop =====
            ax.annotate(f"{y:.0f}%",
                        (x_c, y),
                        ha="left", va="bottom",
                        fontsize=12, color="black",
                        xytext=(6, 7),               # +x to move right; +y for breathing room
                        textcoords="offset points",
                        clip_on=False)

    # Clean up & save
    sns.despine(fig=fig, left=False, bottom=False)
    out_path = Path(__file__).with_name("executive_summary.png")
    plt.tight_layout(rect=[0, 0, 1, 0.88])
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved figure to: {out_path.resolve()}")

if __name__ == "__main__":
    main()
