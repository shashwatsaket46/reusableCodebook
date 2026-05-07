#!/usr/bin/env python
"""Unified recall@k plot: all methods + all hierarchical strategies in one figure.

For each dataset, plots Recall@1@k from k=1 to k=32 with:
  - Original baselines: PQ, TurboQuant, ExRaBitQ at b=2 and b=4
  - Hierarchical strategies at b=4: top-down, bottom-up, middle-anchor

Pulls data from:
  - results/pq_pickles/pq_results_<ds>.pkl
  - results/tq_pickles/tq_results_<ds>.pkl
  - results/logs/exrabitq_<ds>_b<N>.log
  - results/two_tier_recall.csv (single-tier b=4 columns for each strategy)

Outputs:
  - results/figures/unified_recall_<dataset>.png  (per-dataset)
  - results/figures/unified_recall_all.png       (combined 3-panel)
"""
from __future__ import annotations

import csv
import pickle
import re
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "results" / "logs"
PQ_PKLS = ROOT / "results" / "pq_pickles"
TQ_PKLS = ROOT / "results" / "tq_pickles"
FIGS = ROOT / "results" / "figures"
RESULTS = ROOT / "results"
FIGS.mkdir(parents=True, exist_ok=True)

EVAL_PAT = re.compile(r"EVAL bits=(\d+) nprobe=(\d+) k=(\d+) recall=([\d.]+)")
KS = [1, 2, 4, 8, 16, 32]


# ============================================================================
# Data collection per dataset
# ============================================================================

def load_pq_results(name: str) -> dict[str, dict[int, float]]:
    """Returns {label: {k: recall}}."""
    out = {}
    path = PQ_PKLS / f"pq_results_{name}.pkl"
    if not path.exists():
        return out
    raw = pickle.load(open(path, "rb"))
    for k, v in raw.items():
        m = re.match(r"PQ-(\d+)bit", k)
        if m:
            out[f"PQ-{m.group(1)}bit"] = v
    return out


def load_tq_results(name: str) -> dict[str, dict[int, float]]:
    out = {}
    path = TQ_PKLS / f"tq_results_{name}.pkl"
    if not path.exists():
        return out
    raw = pickle.load(open(path, "rb"))
    for k, v in raw.items():
        m = re.match(r"TurboQuant(?:Prod)?-(\d+)bit", k)
        if m:
            out[f"TurboQuant-{m.group(1)}bit"] = v
    return out


def load_exrabitq_results(name: str) -> dict[str, dict[int, float]]:
    """Read exrabitq_*.log files. Returns {ExRaBitQ-Nbit: {k: recall}}."""
    out = {}
    pattern = re.compile(rf"exrabitq_{re.escape(name)}_b(\d+)\.log$")
    for log_path in LOGS.glob(f"exrabitq_{name}_b*.log"):
        m = pattern.search(log_path.name)
        if not m:
            continue
        b = int(m.group(1))
        by_np: dict[int, dict[int, float]] = {}
        for bb, np_, k_str, r_str in EVAL_PAT.findall(log_path.read_text()):
            by_np.setdefault(int(np_), {})[int(k_str)] = float(r_str)
        if by_np:
            best_np = max(by_np)
            out[f"ExRaBitQ-{b}bit"] = by_np[best_np]
    return out


def load_hierarchical_results(name: str) -> dict[str, dict[int, float]]:
    """Read two_tier_recall.csv to extract single-tier b=4 R@k per strategy.

    Each row in the CSV has single_b4_R@1, single_b4_R@4, etc. — pick one row
    per (strategy) and extract R@k for k in KS.
    """
    out = {}
    path = RESULTS / "two_tier_recall.csv"
    if not path.exists():
        return out

    # CSV columns: dataset, d, strategy, T, single_b2_R@1, single_b4_R@1, two_tier_R@1, ...
    # We use single_b4_R@k as the strategy's true b=4 ceiling.
    seen_strategies = {}  # strategy -> dict[k, recall]
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["dataset"] != name:
                continue
            strat = row["strategy"]
            if strat in seen_strategies:
                continue  # already captured (rows are duplicated across T values)
            recalls = {}
            for k in KS:
                col = f"single_b4_R@{k}"
                if col in row and row[col] not in ("", "n/a"):
                    recalls[k] = float(row[col])
            if recalls:
                seen_strategies[strat] = recalls

    # Map to display labels
    label_map = {
        "native": "TQ-native-4bit",
        "top_down": "TQ-top_down-4bit",
        "bottom_up": "TQ-bottom_up-4bit",
        "middle_anchor": "TQ-middle_anchor-4bit",
    }
    for strat, recalls in seen_strategies.items():
        if strat in label_map:
            out[label_map[strat]] = recalls
    return out


# ============================================================================
# Plotting
# ============================================================================

# Style: groups have similar colors, bit-widths differ in shade
STYLES = {
    # Original baselines — solid lines
    "PQ-2bit":            {"color": "#9bc6e8", "marker": "o", "linestyle": "-",  "linewidth": 2.0},
    "PQ-4bit":            {"color": "#1f77b4", "marker": "o", "linestyle": "-",  "linewidth": 2.0},
    "TurboQuant-2bit":    {"color": "#ffbb88", "marker": "^", "linestyle": "-",  "linewidth": 2.0},
    "TurboQuant-4bit":    {"color": "#ff7f0e", "marker": "^", "linestyle": "-",  "linewidth": 2.0},
    "ExRaBitQ-2bit":      {"color": "#a8d5a8", "marker": "s", "linestyle": "-",  "linewidth": 2.0},
    "ExRaBitQ-4bit":      {"color": "#2ca02c", "marker": "s", "linestyle": "-",  "linewidth": 2.0},
    # Hierarchical strategies — dashed lines at b=4 only
    "TQ-native-4bit":         {"color": "#404040", "marker": "*", "linestyle": "--", "linewidth": 2.5},
    "TQ-top_down-4bit":       {"color": "#cc6600", "marker": "D", "linestyle": "--", "linewidth": 2.0},
    "TQ-bottom_up-4bit":      {"color": "#3b6394", "marker": "v", "linestyle": "--", "linewidth": 2.0},
    "TQ-middle_anchor-4bit":  {"color": "#197419", "marker": "<", "linestyle": "--", "linewidth": 2.0},
}

# Display order
DISPLAY_ORDER = [
    "PQ-2bit", "PQ-4bit",
    "TurboQuant-2bit", "TurboQuant-4bit",
    "ExRaBitQ-2bit", "ExRaBitQ-4bit",
    "TQ-native-4bit",
    "TQ-top_down-4bit",
    "TQ-bottom_up-4bit",
    "TQ-middle_anchor-4bit",
]


def collect_all(name: str) -> dict[str, dict[int, float]]:
    out = {}
    out.update(load_pq_results(name))
    out.update(load_tq_results(name))
    out.update(load_exrabitq_results(name))
    out.update(load_hierarchical_results(name))
    return out


def plot_single(name: str, dataset_label: str = None):
    results = collect_all(name)
    if not results:
        print(f"  [{name}] no data found, skipping")
        return None

    fig, ax = plt.subplots(figsize=(11, 7))
    label_map = {
        "PQ-2bit": "PQ (2 bits)",
        "PQ-4bit": "PQ (4 bits)",
        "TurboQuant-2bit": "TurboQuant (2 bits)",
        "TurboQuant-4bit": "TurboQuant (4 bits, native)",
        "ExRaBitQ-2bit": "ExRaBitQ (2 bits)",
        "ExRaBitQ-4bit": "ExRaBitQ (4 bits)",
        "TQ-native-4bit": "TQ-native (4 bits) [shared codebook ref]",
        "TQ-top_down-4bit": "TQ-top_down (4 bits) [nested, ours]",
        "TQ-bottom_up-4bit": "TQ-bottom_up (4 bits) [nested]",
        "TQ-middle_anchor-4bit": "TQ-middle_anchor (4 bits) [nested]",
    }

    for label in DISPLAY_ORDER:
        if label not in results:
            continue
        ks_present = sorted(results[label].keys())
        ks_to_use = [k for k in KS if k in ks_present]
        if not ks_to_use:
            continue
        ys = [results[label][k] for k in ks_to_use]
        style = STYLES.get(label, {"color": "gray", "marker": "x", "linestyle": ":"})
        ax.plot(ks_to_use, ys,
                color=style["color"],
                marker=style["marker"],
                markersize=8,
                linestyle=style["linestyle"],
                linewidth=style.get("linewidth", 2.0),
                label=label_map.get(label, label))

    ax.set_xscale("log", base=2)
    ax.set_xticks(KS)
    ax.set_xticklabels(KS)
    ax.set_xlabel("k", fontsize=12)
    ax.set_ylabel("Recall@1@k", fontsize=12)
    title = dataset_label or name
    ax.set_title(f"{title} — all methods at all k\n"
                 "(solid = baselines; dashed = hierarchical strategies at b=4)",
                 fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.95, ncol=2)
    fig.tight_layout()
    out = FIGS / f"unified_recall_{name}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")
    return results


def plot_combined(datasets_with_results: dict):
    """3-panel figure with one panel per dataset."""
    n = len(datasets_with_results)
    if n == 0:
        return
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6.5), sharey=False)
    if n == 1:
        axes = [axes]

    label_map = {
        "PQ-2bit": "PQ (2 bits)",
        "PQ-4bit": "PQ (4 bits)",
        "TurboQuant-2bit": "TurboQuant (2 bits)",
        "TurboQuant-4bit": "TurboQuant (4 bits)",
        "ExRaBitQ-2bit": "ExRaBitQ (2 bits)",
        "ExRaBitQ-4bit": "ExRaBitQ (4 bits)",
        "TQ-native-4bit": "TQ-native-4bit",
        "TQ-top_down-4bit": "TQ-top_down-4bit",
        "TQ-bottom_up-4bit": "TQ-bottom_up-4bit",
        "TQ-middle_anchor-4bit": "TQ-middle_anchor-4bit",
    }

    seen_handles = {}
    seen_order = []

    for ax, (name, results) in zip(axes, datasets_with_results.items()):
        for label in DISPLAY_ORDER:
            if label not in results:
                continue
            ks_present = sorted(results[label].keys())
            ks_to_use = [k for k in KS if k in ks_present]
            if not ks_to_use:
                continue
            ys = [results[label][k] for k in ks_to_use]
            style = STYLES.get(label, {})
            line, = ax.plot(ks_to_use, ys,
                            color=style["color"], marker=style["marker"],
                            markersize=7, linestyle=style["linestyle"],
                            linewidth=style.get("linewidth", 2.0),
                            label=label_map[label])
            if label not in seen_handles:
                seen_handles[label] = line
                seen_order.append(label)
        ax.set_xscale("log", base=2)
        ax.set_xticks(KS)
        ax.set_xticklabels(KS)
        ax.set_xlabel("k", fontsize=11)
        ax.set_title(name, fontsize=11)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Recall@1@k", fontsize=11)

    # Single legend on the rightmost panel
    fig.legend(
        [seen_handles[lab] for lab in seen_order],
        [label_map[lab] for lab in seen_order],
        loc="lower center", ncol=5, fontsize=8, framealpha=0.95,
        bbox_to_anchor=(0.5, -0.05),
    )
    fig.suptitle("Unified recall@k: original methods + hierarchical TurboQuant strategies",
                 fontsize=13, y=1.0)
    fig.tight_layout()
    out = FIGS / "unified_recall_all.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    DATASETS = ["glove200_100k", "openai1536", "openai3072"]

    print("=== Unified recall plot ===")
    all_results = {}
    for name in DATASETS:
        results = plot_single(name)
        if results:
            all_results[name] = results

    if all_results:
        plot_combined(all_results)

    # Print summary table
    print("\n=== Summary ===")
    for name, results in all_results.items():
        print(f"\n[{name}]")
        for label in DISPLAY_ORDER:
            if label in results:
                rec = results[label]
                rec_str = "  ".join(f"R@{k}={rec[k]:.4f}" for k in KS if k in rec)
                print(f"  {label:<28s} {rec_str}")