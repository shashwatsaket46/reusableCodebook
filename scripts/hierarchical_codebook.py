#!/usr/bin/env python
"""Hierarchical Lloyd-Max codebooks for the unit Gaussian.

The TurboQuant pipeline standardizes rotated coordinates by sqrt(d), making
them approximately N(0, 1). The Lloyd-Max codebook is therefore designed for
the unit Gaussian — *independent of d*. Same codebook works for d=200,
d=1536, d=3072.

Construction (bottom-up):
  1. Solve unconstrained Lloyd-Max for b = b_min (e.g., 2).
  2. For each successive b, keep all previous levels FIXED and add new
     levels via constrained Lloyd-Max iteration.

Each lower-bit codebook is an exact subset of every higher-bit codebook.
The lowest bit-width is unconstrained-optimal; higher bit-widths bear the
distortion penalty of the nesting constraint.

Outputs:
  - results/codebooks/hierarchical_gaussian.npz
  - results/figures/hierarchical_codebook_gaussian.png
  - results/figures/hierarchical_vs_native.png
  - results/figures/hierarchical_mse_bars.png
  - results/codebooks/hierarchical_metrics.csv
"""
from __future__ import annotations

from pathlib import Path
import csv

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
CODEBOOK_DIR = ROOT / "results" / "codebooks"
FIGS = ROOT / "results" / "figures"
CODEBOOK_DIR.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

def build_middle_anchor_codebooks_compat(bit_widths, seed=42, anchor_bits=3):
    """Wrapper for compatibility with run_*_recall.py."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from middle_anchor import build_middle_anchor_codebooks
    return build_middle_anchor_codebooks(bit_widths, anchor_bits=anchor_bits, seed=seed)

# ============================================================================
# Sampling
# ============================================================================

def sample_gaussian(n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample from N(0, 1)."""
    return rng.standard_normal(n)


# ============================================================================
# Lloyd-Max iteration helpers
# ============================================================================

def voronoi_boundaries(levels: np.ndarray) -> np.ndarray:
    """Midpoints between adjacent (sorted) levels."""
    return (levels[:-1] + levels[1:]) / 2.0


def assign_to_levels(samples: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """For each sample, return index of nearest level."""
    if len(levels) == 1:
        return np.zeros_like(samples, dtype=np.int64)
    return np.digitize(samples, voronoi_boundaries(levels))


# ============================================================================
# Lloyd-Max — unconstrained and constrained
# ============================================================================

def lloyd_max_unconstrained(
        n_levels: int,
        n_iter: int = 300,
        n_samples: int = 500_000,
        seed: int = 42,
        init: np.ndarray | None = None,
) -> np.ndarray:
    """Standard Lloyd-Max iteration for N(0, 1)."""
    rng = np.random.default_rng(seed)
    samples = sample_gaussian(n_samples, rng)

    if init is None:
        # Initialize within ±3σ — covers >99% of mass, no stuck cells
        levels = np.linspace(-3, 3, n_levels)
    else:
        levels = np.array(init, dtype=float).copy()
    levels.sort()

    for _ in range(n_iter):
        idx = assign_to_levels(samples, levels)
        new_levels = np.empty_like(levels)
        for i in range(n_levels):
            mask = idx == i
            new_levels[i] = samples[mask].mean() if mask.any() else levels[i]
        new_levels.sort()
        if np.max(np.abs(new_levels - levels)) < 1e-9:
            return new_levels
        levels = new_levels
    return levels


def lloyd_max_constrained(
        fixed_levels: np.ndarray,
        new_levels_init: np.ndarray,
        n_iter: int = 300,
        n_samples: int = 500_000,
        seed: int = 42,
) -> np.ndarray:
    """Lloyd-Max with a subset of levels held FIXED.

    All `fixed_levels` are kept exactly. Only `new_levels` update to centroids
    of their assigned cells. Returns full sorted (fixed + new).
    """
    rng = np.random.default_rng(seed)
    samples = sample_gaussian(n_samples, rng)

    fixed = np.array(fixed_levels, dtype=float)
    new = np.array(new_levels_init, dtype=float)
    n_total = len(fixed) + len(new)

    for _ in range(n_iter):
        all_levels = np.concatenate([fixed, new])
        order = np.argsort(all_levels)
        sorted_levels = all_levels[order]
        is_fixed_sorted = order < len(fixed)

        idx = assign_to_levels(samples, sorted_levels)

        new_updated = []
        for k in range(n_total):
            if is_fixed_sorted[k]:
                continue
            mask = idx == k
            if mask.any():
                centroid = samples[mask].mean()
            else:
                centroid = sorted_levels[k]
            new_updated.append(centroid)

        new_updated = np.array(new_updated)
        if new.shape == new_updated.shape and np.max(np.abs(new_updated - new)) < 1e-9:
            new = new_updated
            break
        new = new_updated

    return np.sort(np.concatenate([fixed, new]))


# ============================================================================
# Bottom-up hierarchical construction
# ============================================================================

def _initialize_new_levels(prev_levels: np.ndarray, n_new: int) -> np.ndarray:
    """Pick n_new initial positions for new levels.

    Heuristic: identify the n_new largest "cells" (gaps between adjacent levels,
    plus the regions [-inf, prev_min] and [prev_max, +inf]) and place one new
    level at the midpoint of each.

    For the unbounded Gaussian, we use ±3σ as the soft boundaries.
    """
    extended = np.concatenate([[-3.0], prev_levels, [3.0]])
    cell_mids = (extended[:-1] + extended[1:]) / 2.0
    cell_widths = np.diff(extended)
    order = np.argsort(-cell_widths)
    chosen = order[:n_new]
    return np.sort(cell_mids[chosen])


def build_hierarchical_codebooks(
        bit_widths: list[int], seed: int = 42
) -> dict[int, np.ndarray]:
    """Build a sequence of nested codebooks for N(0, 1).

    Returns dict {b: levels} where levels[b] is sorted, len = 2^b,
    and levels[b1] subset of levels[b2] for b1 < b2.
    """
    bit_widths = sorted(bit_widths)
    codebooks = {}

    b0 = bit_widths[0]
    codebooks[b0] = lloyd_max_unconstrained(2 ** b0, seed=seed)

    for b_prev, b in zip(bit_widths[:-1], bit_widths[1:]):
        prev_levels = codebooks[b_prev]
        n_new = 2 ** b - 2 ** b_prev
        new_init = _initialize_new_levels(prev_levels, n_new)
        codebooks[b] = lloyd_max_constrained(
            fixed_levels=prev_levels, new_levels_init=new_init, seed=seed,
        )

    return codebooks


def build_top_down_codebooks(
        bit_widths: list[int], seed: int = 42
) -> dict[int, np.ndarray]:
    """Top-down: solve b_max unconstrained, decimate to lower bit-widths.

    Higher bit-widths are unconstrained-optimal; lower bit-widths bear penalty.
    """
    bit_widths = sorted(bit_widths)
    b_max = bit_widths[-1]
    levels_max = lloyd_max_unconstrained(2 ** b_max, seed=seed)
    return {b: levels_max[::2 ** (b_max - b)] for b in bit_widths}


# ============================================================================
# Evaluation
# ============================================================================

def empirical_mse(levels: np.ndarray, n_samples: int = 1_000_000,
                  seed: int = 0) -> float:
    """Empirical MSE of the codebook on N(0, 1) samples."""
    rng = np.random.default_rng(seed)
    samples = sample_gaussian(n_samples, rng)
    if len(levels) == 1:
        idx = np.zeros_like(samples, dtype=np.int64)
    else:
        idx = np.digitize(samples, voronoi_boundaries(levels))
    quantized = levels[idx]
    return float(np.mean((samples - quantized) ** 2))


def verify_nesting(codebooks: dict[int, np.ndarray]) -> bool:
    """Check that lower-b codebooks are exact subsets of higher-b codebooks."""
    bs = sorted(codebooks.keys())
    for i, b_low in enumerate(bs):
        for b_high in bs[i + 1:]:
            low = codebooks[b_low]
            high = codebooks[b_high]
            for lvl in low:
                if not np.any(np.isclose(high, lvl, atol=1e-10)):
                    return False
    return True


# ============================================================================
# Main: build, evaluate, plot, save
# ============================================================================

def main():
    BITS = [2, 3, 4, 5]

    print("=" * 78)
    print("Hierarchical Lloyd-Max codebooks for N(0, 1)")
    print("(matches TurboQuant's standardized rotated-coordinate distribution)")
    print("=" * 78)

    # Native (unconstrained) Lloyd-Max for each bit-width — for comparison
    print("\nNative (unconstrained) Lloyd-Max levels:")
    native = {}
    for b in BITS:
        native[b] = lloyd_max_unconstrained(2 ** b)
        print(f"  b={b}  ({2**b:>2d} levels): {np.array2string(native[b], precision=4, suppress_small=True)}")

    # Bottom-up hierarchical (nested) Lloyd-Max
    print("\nBottom-up hierarchical (nested) Lloyd-Max levels:")
    hier_bu = build_hierarchical_codebooks(BITS)
    for b in BITS:
        print(f"  b={b}  ({2**b:>2d} levels): {np.array2string(hier_bu[b], precision=4, suppress_small=True)}")
    print(f"  Nesting verified: {verify_nesting(hier_bu)}")

    # Top-down hierarchical (decimated from b_max)
    print("\nTop-down hierarchical Lloyd-Max levels (decimated from b_max):")
    hier_td = build_top_down_codebooks(BITS)
    for b in BITS:
        print(f"  b={b}  ({2**b:>2d} levels): {np.array2string(hier_td[b], precision=4, suppress_small=True)}")
    print(f"  Nesting verified: {verify_nesting(hier_td)}")

    print("\n" + "=" * 78)
    print("MSE comparison: native vs bottom-up nested vs top-down nested")
    print("=" * 78)
    print(f"{'b':>3s} {'native':>12s} {'bottom-up':>12s} {'top-down':>12s} "
          f"{'BU gap%':>10s} {'TD gap%':>10s}")
    print("-" * 70)
    rows = [("bits", "native_mse", "hier_bu_mse", "hier_td_mse",
             "bu_gap_pct", "td_gap_pct")]
    for b in BITS:
        mse_n = empirical_mse(native[b])
        mse_bu = empirical_mse(hier_bu[b])
        mse_td = empirical_mse(hier_td[b])
        bu_pct = 100 * (mse_bu - mse_n) / mse_n if mse_n > 0 else 0
        td_pct = 100 * (mse_td - mse_n) / mse_n if mse_n > 0 else 0
        print(f"{b:>3d} {mse_n:>12.6f} {mse_bu:>12.6f} {mse_td:>12.6f} "
              f"{bu_pct:>+9.2f}% {td_pct:>+9.2f}%")
        rows.append((b, mse_n, mse_bu, mse_td, bu_pct, td_pct))

    # Save codebooks
    save_dict = {f"native_b{b}": native[b] for b in BITS}
    save_dict.update({f"bu_b{b}": hier_bu[b] for b in BITS})
    save_dict.update({f"td_b{b}": hier_td[b] for b in BITS})
    np.savez(CODEBOOK_DIR / "hierarchical_gaussian.npz", **save_dict)
    print(f"\nCodebooks saved: {CODEBOOK_DIR / 'hierarchical_gaussian.npz'}")

    # Save metrics CSV
    metrics_path = CODEBOOK_DIR / "hierarchical_metrics.csv"
    with open(metrics_path, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"Metrics CSV:     {metrics_path}")

    # ------------------------------------------------------------------
    # Visualization 1: bottom-up codebook nesting structure
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 6))
    for b in BITS:
        levels = hier_bu[b]
        y = b * np.ones_like(levels)
        ax.scatter(levels, y, s=80, zorder=3, label=f"b={b} ({2**b} levels)")
        for lvl in levels:
            ax.text(lvl, b - 0.18, f"{lvl:+.2f}",
                    ha="center", va="top", fontsize=7, alpha=0.7)
    for i, b in enumerate(BITS[:-1]):
        for lvl in hier_bu[b]:
            ax.plot([lvl, lvl], [b, b + 1], "k-", alpha=0.25, linewidth=1)
    ax.set_xlabel("Codebook level (in standardized coordinate space)")
    ax.set_ylabel("bits b")
    ax.set_yticks(BITS)
    ax.set_xlim(-3.5, 3.5)
    ax.set_title(
        "Bottom-up hierarchical Lloyd-Max codebooks for N(0, 1)\n"
        "Vertical lines: a level at b is preserved at b+1 (nesting structure)"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    out = FIGS / "hierarchical_codebook_gaussian.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved:    {out}")

    # ------------------------------------------------------------------
    # Visualization 2: native vs bottom-up vs top-down side-by-side
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for ax, codebook_set, title in [
        (axes[0], native, "Native (unconstrained) Lloyd-Max"),
        (axes[1], hier_bu, "Bottom-up nested Lloyd-Max"),
        (axes[2], hier_td, "Top-down nested Lloyd-Max"),
    ]:
        for b in BITS:
            levels = codebook_set[b]
            y = b * np.ones_like(levels)
            ax.scatter(levels, y, s=70, zorder=3, label=f"b={b}")
        x_ref = np.linspace(-3, 3, 500)
        pdf = np.exp(-x_ref ** 2 / 2) / np.sqrt(2 * np.pi)
        ax.fill_between(x_ref, 1.5, 1.5 + pdf * 1.2, color='gray', alpha=0.2)
        ax.set_xlim(-3.5, 3.5)
        ax.set_yticks(BITS)
        ax.set_xlabel("Codebook level")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=9)
    axes[0].set_ylabel("bits b")
    fig.suptitle("Native vs hierarchical codebook strategies (N(0, 1) pdf in gray)",
                 fontsize=12)
    fig.tight_layout()
    out2 = FIGS / "hierarchical_vs_native.png"
    fig.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved:    {out2}")

    # ------------------------------------------------------------------
    # Visualization 3: MSE bar plot — the headline result
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(BITS))
    width = 0.27
    native_mses = [empirical_mse(native[b]) for b in BITS]
    bu_mses = [empirical_mse(hier_bu[b]) for b in BITS]
    td_mses = [empirical_mse(hier_td[b]) for b in BITS]
    ax.bar(x - width, native_mses, width,
           label="Native (unconstrained)", color="tab:gray", alpha=0.85)
    bars2 = ax.bar(x, bu_mses, width,
                   label="Bottom-up nested", color="tab:blue", alpha=0.85)
    bars3 = ax.bar(x + width, td_mses, width,
                   label="Top-down nested", color="tab:orange", alpha=0.85)
    for i, b in enumerate(BITS):
        bu_pct = 100 * (bu_mses[i] - native_mses[i]) / native_mses[i]
        td_pct = 100 * (td_mses[i] - native_mses[i]) / native_mses[i]
        if bu_pct > 1:
            ax.text(x[i], bu_mses[i],
                    f"+{bu_pct:.0f}%", ha="center", va="bottom", fontsize=8, color="tab:blue")
        if td_pct > 1:
            ax.text(x[i] + width, td_mses[i],
                    f"+{td_pct:.0f}%", ha="center", va="bottom", fontsize=8, color="tab:orange")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"b={b}" for b in BITS])
    ax.set_xlabel("bits per coordinate")
    ax.set_ylabel("Empirical MSE per coordinate")
    ax.set_title("Distortion cost of nesting — Lloyd-Max for N(0, 1)")
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out3 = FIGS / "hierarchical_mse_bars.png"
    fig.savefig(out3, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved:    {out3}")


if __name__ == "__main__":
    main()