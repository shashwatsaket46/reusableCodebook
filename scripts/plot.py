#!/usr/bin/env python
"""Generate per-dataset and combined recall@k plots."""
import pickle
import re
from pathlib import Path

import matplotlib.pyplot as plt

from config_utils import enabled_datasets, ensure_int_list, load_config

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "results" / "logs"
PKLS = ROOT / "results" / "pq_pickles"
FIGS = ROOT / "results" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

EVAL_PAT = re.compile(r"EVAL bits=(\d+) nprobe=(\d+) k=(\d+) recall=([\d.]+)")

def _dataset_title(name, ds_cfg):
    dim = int(ds_cfg.get("dim", 0))
    n_query = int(ds_cfg.get("n_query", 0))
    return f"{name}, {n_query} queries", dim


def _exr_label(bits):
    return f"ExRaBitQ-{bits}bit"


def _build_order_and_styles(cfg):
    pq_bits = ensure_int_list(cfg.get("pq_bits", []), "pq_bits")
    ex_bits = ensure_int_list(cfg.get("exrabitq_bits", []), "exrabitq_bits")
    colors = cfg.get("plot", {}).get("colors", {})
    styles = {}
    order = []

    for b in pq_bits:
        key = f"PQ-{b}bit"
        marker = "o" if b == 2 else "s"
        color = colors.get(f"pq_{b}bit", "tab:blue" if b == 2 else "tab:red")
        styles[key] = (marker, "-", color)
        order.append(key)

    for b in ex_bits:
        key = _exr_label(b)
        pair = 2 if b in (2, 3) else 4
        color = colors.get(f"exrabitq_{pair}bit", "tab:blue" if pair == 2 else "tab:red")
        styles[key] = ("x", "--", color)
        order.append(key)

    return order, styles, ex_bits

def collect(name, ex_bits):
    """Load PQ pickle + ExRaBitQ logs. For each B, picks the largest nprobe
    seen in the log (= exhaustive search, robust to nlist choice)."""
    pq_path = PKLS / f"pq_results_{name}.pkl"
    if not pq_path.exists():
        raise FileNotFoundError(pq_path)
    res = pickle.load(open(pq_path, "rb"))
    nprobes_used = {}   # for the title later
    for B in ex_bits:
        lab = _exr_label(B)
        log = LOGS / f"exrabitq_{name}_b{B}.log"
        if not log.exists():
            continue
        # Group EVAL rows by nprobe, take the largest (exhaustive)
        by_np = {}
        for bb, np_, k, r in EVAL_PAT.findall(log.read_text()):
            by_np.setdefault(int(np_), {})[int(k)] = float(r)
        if by_np:
            best = max(by_np)
            res[lab] = by_np[best]
            nprobes_used[B] = best
    return res, nprobes_used

def _title_for(name, ds_cfg, nprobes_used):
    title, dim = _dataset_title(name, ds_cfg)
    if nprobes_used:
        unique = sorted(set(nprobes_used.values()))
        npr = unique[0] if len(unique) == 1 else "/".join(str(n) for n in unique)
        return f"{title} (d={dim}) — exhaustive (nprobe={npr})"
    return f"{title} (d={dim})"

def plot_single(name, ds_cfg, ks, y_lim, order, styles, ex_bits):
    res, nprobes_used = collect(name, ex_bits)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for n in order:
        if n not in res: continue
        m, ls, c = styles[n]
        ax.plot(ks, [res[n][k] for k in ks], marker=m, linestyle=ls,
                color=c, linewidth=2.2, markersize=8, label=n)
    ax.set_xscale("log", base=2); ax.set_xticks(ks); ax.set_xticklabels(ks)
    ax.set_xlabel("k", fontsize=11)
    ax.set_ylabel("Recall@k  (top-1 GT hit rate)", fontsize=11)
    ax.set_ylim(*y_lim)
    ax.set_title(_title_for(name, ds_cfg, nprobes_used), fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.4); ax.legend(loc="lower right")
    fig.tight_layout()
    out = FIGS / f"recall_{name}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")
    return res

def plot_combined(all_res, datasets_cfg, ks, y_lim, order, styles):
    n_panels = len(all_res)
    fig, axes = plt.subplots(1, n_panels, figsize=(6*n_panels, 5.5),
                             sharey=True)
    if n_panels == 1: axes = [axes]
    for ax, (name, res) in zip(axes, all_res.items()):
        title, dim = _dataset_title(name, datasets_cfg[name])
        for n in order:
            if n not in res: continue
            m, ls, c = styles[n]
            ax.plot(ks, [res[n][k] for k in ks], marker=m, linestyle=ls,
                    color=c, linewidth=2.2, markersize=8, label=n)
        ax.set_xscale("log", base=2); ax.set_xticks(ks); ax.set_xticklabels(ks)
        ax.set_xlabel("k", fontsize=11)
        ax.set_title(f"{title} (d={dim})", fontsize=12)
        ax.set_ylim(*y_lim)
        ax.grid(True, linestyle="--", alpha=0.4)
    axes[0].set_ylabel("Recall@k  (top-1 GT hit rate)", fontsize=11)
    axes[-1].legend(loc="lower right", fontsize=9, framealpha=0.95)
    fig.suptitle("PQ vs Extended RaBitQ — exhaustive search",
                 fontsize=13, y=1.00)
    fig.tight_layout()
    out = FIGS / "recall_all_datasets.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")

if __name__ == "__main__":
    cfg = load_config()
    enabled = enabled_datasets(cfg)
    ks = ensure_int_list(cfg.get("eval_ks", []), "eval_ks")
    y_lim_cfg = cfg.get("plot", {}).get("y_lim", [0.50, 1.01])
    y_lim = (float(y_lim_cfg[0]), float(y_lim_cfg[1]))
    datasets_cfg = cfg["datasets"]
    order, styles, ex_bits = _build_order_and_styles(cfg)

    all_res = {}
    for name in enabled:
        try:
            res = plot_single(name, datasets_cfg[name], ks, y_lim, order, styles, ex_bits)
            all_res[name] = res
            print(f"[{name}]")
            for n, r in res.items():
                print(f"  {n:<40s}",
                      " ".join(f"R@{k}={r[k]:.4f}" for k in ks))
        except FileNotFoundError as e:
            print(f"[{name}] missing inputs: {e}; skipping")
    if all_res:
        plot_combined(all_res, datasets_cfg, ks, y_lim, order, styles)
    print("\nfigures in", FIGS)