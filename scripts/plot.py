#!/usr/bin/env python
"""Generate per-dataset and combined recall@k plots."""
import os, re, pickle
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "results" / "logs"
PKLS = ROOT / "results" / "pq_pickles"
FIGS = ROOT / "results" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

ENABLED = os.environ.get("DATASETS",
                         "glove200_100k openai1536 openai3072").split()
KS = [1, 2, 4, 8, 16, 32]
EVAL_PAT = re.compile(r"EVAL bits=(\d+) nprobe=(\d+) k=(\d+) recall=([\d.]+)")

DATASET_TITLES = {
    "glove200_100k": ("GloVe-200, 10k queries",   200),
    "openai1536":    ("OpenAI-1536, 1k queries",  1536),
    "openai3072":    ("OpenAI-3072, 1k queries",  3072),
}
EXR_LABELS = {3: "ExRaBitQ-2bit+sign (3 bits/dim)",
              4: "ExRaBitQ-4bit+sign (4 bits/dim)"}
ORDER = ["PQ-2bit", "ExRaBitQ-2bit+sign (3 bits/dim)",
         "PQ-4bit", "ExRaBitQ-4bit+sign (4 bits/dim)"]
STYLES = {
    "PQ-2bit":                            ("o", "-",  "tab:blue"),
    "ExRaBitQ-2bit+sign (3 bits/dim)":    ("x", "--", "tab:blue"),
    "PQ-4bit":                            ("s", "-",  "tab:red"),
    "ExRaBitQ-4bit (4 bits/dim)":    ("x", "--", "tab:red"),
}

def collect(name):
    """Load PQ pickle + ExRaBitQ logs. For each B, picks the largest nprobe
    seen in the log (= exhaustive search, robust to nlist choice)."""
    pq_path = PKLS / f"pq_results_{name}.pkl"
    if not pq_path.exists():
        raise FileNotFoundError(pq_path)
    res = pickle.load(open(pq_path, "rb"))
    nprobes_used = {}   # for the title later
    for B, lab in EXR_LABELS.items():
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

def _title_for(name, nprobes_used):
    title, dim = DATASET_TITLES[name]
    if nprobes_used:
        unique = sorted(set(nprobes_used.values()))
        npr = unique[0] if len(unique) == 1 else "/".join(str(n) for n in unique)
        return f"{title} (d={dim}) — exhaustive (nprobe={npr})"
    return f"{title} (d={dim})"

def plot_single(name):
    res, nprobes_used = collect(name)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for n in ORDER:
        if n not in res: continue
        m, ls, c = STYLES[n]
        ax.plot(KS, [res[n][k] for k in KS], marker=m, linestyle=ls,
                color=c, linewidth=2.2, markersize=8, label=n)
    ax.set_xscale("log", base=2); ax.set_xticks(KS); ax.set_xticklabels(KS)
    ax.set_xlabel("k", fontsize=11)
    ax.set_ylabel("Recall@k  (top-1 GT hit rate)", fontsize=11)
    ax.set_ylim(0.50, 1.01)
    ax.set_title(_title_for(name, nprobes_used), fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.4); ax.legend(loc="lower right")
    fig.tight_layout()
    out = FIGS / f"recall_{name}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")
    return res

def plot_combined(all_res):
    n_panels = len(all_res)
    fig, axes = plt.subplots(1, n_panels, figsize=(6*n_panels, 5.5),
                             sharey=True)
    if n_panels == 1: axes = [axes]
    for ax, (name, res) in zip(axes, all_res.items()):
        title, dim = DATASET_TITLES[name]
        for n in ORDER:
            if n not in res: continue
            m, ls, c = STYLES[n]
            ax.plot(KS, [res[n][k] for k in KS], marker=m, linestyle=ls,
                    color=c, linewidth=2.2, markersize=8, label=n)
        ax.set_xscale("log", base=2); ax.set_xticks(KS); ax.set_xticklabels(KS)
        ax.set_xlabel("k", fontsize=11)
        ax.set_title(f"{title} (d={dim})", fontsize=12)
        ax.set_ylim(0.50, 1.01)
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
    all_res = {}
    for name in ENABLED:
        if name not in DATASET_TITLES:
            print(f"unknown dataset {name!r}; skipping"); continue
        try:
            res = plot_single(name)
            all_res[name] = res
            print(f"[{name}]")
            for n, r in res.items():
                print(f"  {n:<40s}",
                      " ".join(f"R@{k}={r[k]:.4f}" for k in KS))
        except FileNotFoundError as e:
            print(f"[{name}] missing inputs: {e}; skipping")
    if all_res:
        plot_combined(all_res)
    print("\nfigures in", FIGS)