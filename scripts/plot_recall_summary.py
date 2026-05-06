# scripts/plot_recall_summary.py
"""Headline figure: MSE penalty vs R@1 loss for hierarchical TurboQuant."""
import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent

# Load recall results
recall_csv = ROOT / "results" / "hierarchical_recall.csv"
recall_data = {}  # (dataset, strategy, bits) -> R@1
with open(recall_csv) as f:
    reader = csv.DictReader(f)
    for row in reader:
        key = (row["dataset"], row["strategy"], int(row["bits"]))
        recall_data[key] = float(row["R@1"])

# Load MSE results
mse_csv = ROOT / "results" / "codebooks" / "hierarchical_metrics.csv"
mse_data = {}  # (strategy, bits) -> mse
with open(mse_csv) as f:
    reader = csv.DictReader(f)
    for row in reader:
        bits = int(row["bits"])
        mse_data[("native", bits)] = float(row["native_mse"])
        mse_data[("bottom_up", bits)] = float(row["hier_bu_mse"])
        mse_data[("top_down", bits)] = float(row["hier_td_mse"])

# Compute deltas
datasets = sorted(set(k[0] for k in recall_data.keys()))
fig, ax = plt.subplots(figsize=(8, 6))

colors = {"bottom_up": "tab:blue", "top_down": "tab:orange"}
markers = {"glove200_100k": "o", "openai1536": "s", "openai3072": "^"}

for ds in datasets:
    for strat in ["bottom_up", "top_down"]:
        xs, ys = [], []
        for b in [2, 3, 4, 5]:
            if (ds, strat, b) not in recall_data:
                continue
            mse_native = mse_data[("native", b)]
            mse_strat = mse_data[(strat, b)]
            r1_native = recall_data[(ds, "native", b)]
            r1_strat = recall_data[(ds, strat, b)]
            mse_pct = 100 * (mse_strat - mse_native) / mse_native
            r1_loss = 100 * (r1_native - r1_strat)
            xs.append(mse_pct)
            ys.append(r1_loss)
            ax.annotate(f"b={b}", (mse_pct, r1_loss), fontsize=8, alpha=0.7,
                        xytext=(3, 3), textcoords="offset points")
        if xs:
            ax.scatter(xs, ys, color=colors[strat], marker=markers.get(ds, "o"),
                       s=80, alpha=0.7,
                       label=f"{strat} on {ds}")

ax.axhline(0, color="gray", linestyle=":", alpha=0.5)
ax.axvline(0, color="gray", linestyle=":", alpha=0.5)
ax.set_xlabel("MSE penalty vs unconstrained Lloyd-Max (%)", fontsize=12)
ax.set_ylabel("R@1 loss vs unconstrained Lloyd-Max (percentage points)", fontsize=12)
ax.set_title("Hierarchical TurboQuant: MSE penalty vs recall loss is sublinear",
             fontsize=13)
ax.grid(True, alpha=0.3)
ax.legend(loc="upper left", fontsize=9)
fig.tight_layout()

out = ROOT / "results" / "figures" / "mse_vs_recall_summary.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")