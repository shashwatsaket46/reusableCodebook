#!/usr/bin/env python
"""Headline figure: MSE penalty vs R@1 loss for hierarchical TurboQuant.

Uses results from:
  results/codebooks/all_strategies_metrics.csv  (MSE per strategy per bit)
  results/multiseed_recall.csv                  (R@1 per strategy per dataset)

Falls back to single-seed two_tier_recall.csv if multiseed not available.
"""
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGS = RESULTS / "figures"
FIGS.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------
# Load MSE data — try all_strategies_metrics first, then hierarchical_metrics
# ----------------------------------------------------------------------

def load_mse_data():
    """Returns dict[(strategy, bits)] = mse."""
    mse = {}

    # Preferred: from all_strategies_metrics.csv (4 strategies)
    path1 = RESULTS / "codebooks" / "all_strategies_metrics.csv"
    if path1.exists():
        with open(path1) as f:
            for row in csv.DictReader(f):
                b = int(row["bits"])
                mse[("native", b)] = float(row["native"])
                mse[("bottom_up", b)] = float(row["bottom_up"])
                mse[("top_down", b)] = float(row["top_down"])
                mse[("middle_anchor", b)] = float(row["middle_anchor"])
        return mse

    # Fallback: hierarchical_metrics.csv (3 strategies, no middle_anchor)
    path2 = RESULTS / "codebooks" / "hierarchical_metrics.csv"
    if path2.exists():
        with open(path2) as f:
            for row in csv.DictReader(f):
                b = int(row["bits"])
                mse[("native", b)] = float(row["native_mse"])
                mse[("bottom_up", b)] = float(row["hier_bu_mse"])
                mse[("top_down", b)] = float(row["hier_td_mse"])
        return mse

    raise FileNotFoundError(
        f"Neither {path1} nor {path2} found. "
        "Run hierarchical_codebook.py and middle_anchor.py first."
    )


# ----------------------------------------------------------------------
# Load recall data — prefer multiseed (with mean across seeds)
# ----------------------------------------------------------------------

def load_recall_data():
    """Returns dict[(dataset, strategy, bits)] = mean R@1.

    For two-tier recall, the relevant b is BITS_EXPENSIVE (=4 in our pipeline).
    We use single_b4_R@1 (single-tier b=4 reference) as the recall measurement
    per strategy, since that's what reflects the codebook quality directly.

    Per-bit recalls aren't available — the two-tier pipeline only runs at one
    (bits_cheap, bits_expensive) pair. So we report bits=4 as the single point.
    """
    recall = {}

    # Try multiseed first (preferred — has mean across seeds)
    multi_path = RESULTS / "multiseed_recall.csv"
    if multi_path.exists():
        # Group by (dataset, strategy) and average single_b4 across seeds
        grouped = {}
        with open(multi_path) as f:
            for row in csv.DictReader(f):
                key = (row["dataset"], row["strategy"])
                grouped.setdefault(key, []).append(float(row["single_b4_R@1"]))
        for (ds, strat), vals in grouped.items():
            recall[(ds, strat, 4)] = float(np.mean(vals))
        return recall

    # Fallback: single-seed two-tier results
    tt_path = RESULTS / "two_tier_recall.csv"
    if tt_path.exists():
        with open(tt_path) as f:
            for row in csv.DictReader(f):
                # Take any T value — single-tier b=4 is the same across T
                key = (row["dataset"], row["strategy"], 4)
                recall[key] = float(row["single_b4_R@1"])
        return recall

    raise FileNotFoundError(
        f"Neither {multi_path} nor {tt_path} found. "
        "Run run_multiseed_recall.py or run_two_tier_recall.py first."
    )


# ----------------------------------------------------------------------
# Main plot: MSE penalty vs R@1 loss
# ----------------------------------------------------------------------

def main():
    mse_data = load_mse_data()
    recall_data = load_recall_data()

    datasets = sorted(set(k[0] for k in recall_data.keys()))
    strategies = ["bottom_up", "top_down", "middle_anchor"]

    colors = {
        "bottom_up": "#1f77b4",
        "top_down": "#ff7f0e",
        "middle_anchor": "#2ca02c",
    }
    markers = {
        "glove200_100k": "o",
        "openai1536": "s",
        "openai3072": "^",
    }

    fig, ax = plt.subplots(figsize=(10, 7))

    # For each (dataset, strategy), compute MSE penalty vs R@1 loss at b=4
    # (b=4 is the rerank tier — where the codebook quality matters)
    plotted_any = False
    for ds in datasets:
        for strat in strategies:
            if (strat, 4) not in mse_data:
                continue
            if (ds, "native", 4) not in recall_data:
                continue
            if (ds, strat, 4) not in recall_data:
                continue

            mse_native = mse_data[("native", 4)]
            mse_strat = mse_data[(strat, 4)]
            r1_native = recall_data[(ds, "native", 4)]
            r1_strat = recall_data[(ds, strat, 4)]

            mse_pct = 100 * (mse_strat - mse_native) / mse_native
            r1_loss = 100 * (r1_native - r1_strat)

            ax.scatter(mse_pct, r1_loss,
                       color=colors[strat],
                       marker=markers.get(ds, "o"),
                       s=140, alpha=0.75, zorder=3,
                       edgecolors="black", linewidths=0.5,
                       label=f"{strat} on {ds}")
            ax.annotate(strat,
                        (mse_pct, r1_loss),
                        fontsize=8, alpha=0.7,
                        xytext=(8, 6), textcoords="offset points")
            plotted_any = True

    if not plotted_any:
        print("No data to plot — check that MSE and recall CSVs are populated.")
        return

    # Reference lines
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5,
               label="zero recall loss (target)")
    ax.axvline(0, color="gray", linestyle=":", alpha=0.5)

    ax.set_xlabel("MSE penalty at b=4 vs unconstrained Lloyd-Max (%)", fontsize=12)
    ax.set_ylabel("R@1 loss at b=4 vs unconstrained native (percentage points)",
                  fontsize=12)
    ax.set_title("Hierarchical TurboQuant: MSE penalty vs recall loss is sublinear\n"
                 "(b=4 rerank tier; markers = datasets, colors = strategies)",
                 fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.95, ncol=2)

    fig.tight_layout()
    out = FIGS / "mse_vs_recall_summary.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

    # ------------------------------------------------------------------
    # Bonus: per-bit R@1 across strategies (bar chart)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, len(datasets),
                             figsize=(5 * len(datasets), 5),
                             sharey=False)
    if len(datasets) == 1:
        axes = [axes]

    strategy_labels = ["native", "top_down", "middle_anchor", "bottom_up"]
    for ax, ds in zip(axes, datasets):
        r1_values = []
        present_strats = []
        for s in strategy_labels:
            if (ds, s, 4) in recall_data:
                r1_values.append(recall_data[(ds, s, 4)])
                present_strats.append(s)
        if not r1_values:
            continue
        x = np.arange(len(present_strats))
        bar_colors = [colors.get(s, "#404040") for s in present_strats]
        # native is gray
        bar_colors = ["#404040" if s == "native" else c
                      for s, c in zip(present_strats, bar_colors)]
        ax.bar(x, r1_values, color=bar_colors, alpha=0.85,
               edgecolor="black", linewidth=0.5)
        for xi, v in zip(x, r1_values):
            ax.text(xi, v + 0.005, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(present_strats, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("Single-tier b=4 R@1", fontsize=10)
        d_str = ds.replace("glove200_100k", "200").replace("openai", "")
        ax.set_title(f"{ds} (d={d_str})", fontsize=11)
        ax.set_ylim(0, max(r1_values) * 1.1)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Single-tier b=4 R@1 by strategy", fontsize=13, y=1.02)
    fig.tight_layout()
    out2 = FIGS / "single_tier_b4_by_strategy.png"
    fig.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out2}")


if __name__ == "__main__":
    main()