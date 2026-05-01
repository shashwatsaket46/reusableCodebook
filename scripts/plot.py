#!/usr/bin/env python
"""Generate per-dataset and combined recall@k plots.

Auto-discovers all ExRaBitQ bit-widths from log files in results/logs/,
and TurboQuantProd bit-widths from pickles in results/tq_pickles/,
so the plot adapts to whatever was actually run.
"""
import pickle
import re
from pathlib import Path

import matplotlib.pyplot as plt

from config_utils import enabled_datasets, ensure_int_list, load_config

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "results" / "logs"
PKLS = ROOT / "results" / "pq_pickles"
TQ_PKLS = ROOT / "results" / "tq_pickles"
FIGS = ROOT / "results" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

EVAL_PAT = re.compile(r"EVAL bits=(\d+) nprobe=(\d+) k=(\d+) recall=([\d.]+)")


# ----------------------------------------------------------------------
# Labels and styling
# ----------------------------------------------------------------------

def _dataset_title(name, ds_cfg):
    dim = int(ds_cfg.get("dim", 0))
    n_query = int(ds_cfg.get("n_query", 0))
    return f"{name}, {n_query} queries", dim


def _pq_label(bits):
    return f"PQ-{bits}bit ({bits} bits/dim)"


def _exr_label(bits):
    """Total bits/dim for an ExRaBitQ run is the same as the `B` argument
    passed to test_search (1 short code bit + (B-1) long code bits)."""
    return f"ExRaBitQ-{bits}bit ({bits} bits/dim)"


def _tq_label(bits):
    return f"TurboQuantProd-{bits}bit ({bits} bits/dim)"


def _color_for(method, bits, plot_colors):
    """Pick a color from config; fall back to a deterministic palette."""
    palette = ["tab:blue", "tab:red", "tab:green", "tab:orange",
               "tab:purple", "tab:brown", "tab:olive", "tab:cyan",
               "tab:pink", "tab:gray"]
    key = f"{method}_{bits}bit"
    if key in plot_colors:
        return plot_colors[key]
    # Deterministic fallback so re-runs are stable
    return palette[(hash(key)) % len(palette)]


# ----------------------------------------------------------------------
# Discovery
# ----------------------------------------------------------------------

def _discover_ex_bits(name):
    """Find all bN.log files for this dataset; return sorted list of bits."""
    pattern = re.compile(rf"exrabitq_{re.escape(name)}_b(\d+)\.log$")
    bits = []
    for log_path in LOGS.glob(f"exrabitq_{name}_b*.log"):
        m = pattern.search(log_path.name)
        if m:
            bits.append(int(m.group(1)))
    return sorted(set(bits))


# ----------------------------------------------------------------------
# Style construction
# ----------------------------------------------------------------------

def _build_styles(pq_bits, ex_bits, tq_bits, plot_colors):
    """Return (label_order, styles_dict).

    PQ           -> solid line
    ExRaBitQ     -> dashed line
    TurboQuant   -> dotted line
    """
    order = []
    styles = {}

    markers = {2: "o", 4: "s", 1: "D", 3: "^", 5: "v", 6: "p", 8: "h"}

    for b in sorted(pq_bits):
        key = _pq_label(b)
        styles[key] = (markers.get(b, "o"), "-",
                       _color_for("pq", b, plot_colors))
        order.append(key)

    for b in sorted(ex_bits):
        key = _exr_label(b)
        styles[key] = (markers.get(b, "x"), "--",
                       _color_for("exrabitq", b, plot_colors))
        order.append(key)

    for b in sorted(tq_bits):
        key = _tq_label(b)
        styles[key] = (markers.get(b, "*"), ":",
                       _color_for("turboquantprod", b, plot_colors))
        order.append(key)

    return order, styles


# ----------------------------------------------------------------------
# Data collection
# ----------------------------------------------------------------------

def collect(name, plot_colors):
    """Load PQ pickle + auto-discovered ExRaBitQ logs + TurboQuant pickle.

    For ExRaBitQ each B picks the largest nprobe seen in the log
    (= exhaustive search, robust to nlist choice).
    """
    res = {}

    # ----- PQ pickle (required) -----
    pq_path = PKLS / f"pq_results_{name}.pkl"
    if not pq_path.exists():
        raise FileNotFoundError(pq_path)
    raw_pq = pickle.load(open(pq_path, "rb"))
    pq_bits_present = []
    for k, v in raw_pq.items():
        m = re.match(r"PQ-(\d+)bit", k)
        if m:
            b = int(m.group(1))
            res[_pq_label(b)] = v
            pq_bits_present.append(b)
        else:
            # unknown key — pass through but don't include in styling
            res[k] = v

    # ----- ExRaBitQ logs (optional) -----
    nprobes_used = {}
    ex_bits_present = _discover_ex_bits(name)
    for B in ex_bits_present:
        log = LOGS / f"exrabitq_{name}_b{B}.log"
        by_np = {}
        for bb, np_, k, r in EVAL_PAT.findall(log.read_text()):
            by_np.setdefault(int(np_), {})[int(k)] = float(r)
        if by_np:
            best = max(by_np)
            res[_exr_label(B)] = by_np[best]
            nprobes_used[B] = best

    # ----- TurboQuant pickle (optional) -----
    tq_bits_present = []
    tq_path = TQ_PKLS / f"tq_results_{name}.pkl"
    if tq_path.exists():
        raw_tq = pickle.load(open(tq_path, "rb"))
        for k, v in raw_tq.items():
            m = re.match(r"TurboQuantProd-(\d+)bit", k)
            if m:
                b = int(m.group(1))
                res[_tq_label(b)] = v
                tq_bits_present.append(b)

    order, styles = _build_styles(
        pq_bits_present, ex_bits_present, tq_bits_present, plot_colors
    )
    return res, nprobes_used, order, styles


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------

def _title_for(name, ds_cfg, nprobes_used):
    title, dim = _dataset_title(name, ds_cfg)
    if nprobes_used:
        unique = sorted(set(nprobes_used.values()))
        npr = unique[0] if len(unique) == 1 else "/".join(str(n) for n in unique)
        return f"{title} (d={dim}) — exhaustive (nprobe={npr})"
    return f"{title} (d={dim})"


def plot_single(name, ds_cfg, ks, y_lim, plot_colors):
    res, nprobes_used, order, styles = collect(name, plot_colors)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for n in order:
        if n not in res:
            continue
        m, ls, c = styles[n]
        ax.plot(ks, [res[n][k] for k in ks], marker=m, linestyle=ls,
                color=c, linewidth=2.2, markersize=8, label=n)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ks)
    ax.set_xticklabels(ks)
    ax.set_xlabel("k", fontsize=11)
    ax.set_ylabel("Recall1@k", fontsize=11)
    ax.set_ylim(*y_lim)
    ax.set_title(_title_for(name, ds_cfg, nprobes_used), fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
    fig.tight_layout()
    out = FIGS / f"recall_{name}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")
    return res, order, styles


def plot_combined(all_res_orders, datasets_cfg, ks, y_lim):
    """all_res_orders: dict[name] = (res, order, styles)"""
    n_panels = len(all_res_orders)
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5.5),
                             sharey=True)
    if n_panels == 1:
        axes = [axes]

    seen_labels = []
    legend_handles = {}

    for ax, (name, (res, order, styles)) in zip(axes, all_res_orders.items()):
        title, dim = _dataset_title(name, datasets_cfg[name])
        for n in order:
            if n not in res:
                continue
            m, ls, c = styles[n]
            line, = ax.plot(ks, [res[n][k] for k in ks], marker=m, linestyle=ls,
                            color=c, linewidth=2.2, markersize=8, label=n)
            if n not in legend_handles:
                legend_handles[n] = line
                seen_labels.append(n)
        ax.set_xscale("log", base=2)
        ax.set_xticks(ks)
        ax.set_xticklabels(ks)
        ax.set_xlabel("k", fontsize=11)
        ax.set_title(f"{title} (d={dim})", fontsize=12)
        ax.set_ylim(*y_lim)
        ax.grid(True, linestyle="--", alpha=0.4)

    axes[0].set_ylabel("Recall1@k", fontsize=11)
    axes[-1].legend(
        [legend_handles[n] for n in seen_labels], seen_labels,
        loc="lower right", fontsize=9, framealpha=0.95,
    )
    fig.suptitle("PQ vs Extended RaBitQ vs TurboQuantProd — exhaustive search",
                 fontsize=13, y=1.00)
    fig.tight_layout()
    out = FIGS / "recall_all_datasets.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

if __name__ == "__main__":
    cfg = load_config()
    enabled = enabled_datasets(cfg)
    ks = ensure_int_list(cfg.get("eval_ks", []), "eval_ks")
    y_lim_cfg = cfg.get("plot", {}).get("y_lim", [0.50, 1.01])
    y_lim = (float(y_lim_cfg[0]), float(y_lim_cfg[1]))
    plot_colors = cfg.get("plot", {}).get("colors", {})
    datasets_cfg = cfg["datasets"]

    all_res_orders = {}
    for name in enabled:
        try:
            res, order, styles = plot_single(
                name, datasets_cfg[name], ks, y_lim, plot_colors
            )
            all_res_orders[name] = (res, order, styles)
            print(f"[{name}]")
            for n, r in res.items():
                print(f"  {n:<40s}",
                      " ".join(f"R@{k}={r[k]:.4f}" for k in ks))
        except FileNotFoundError as e:
            print(f"[{name}] missing inputs: {e}; skipping")

    if all_res_orders:
        plot_combined(all_res_orders, datasets_cfg, ks, y_lim)
    print("\nfigures in", FIGS)