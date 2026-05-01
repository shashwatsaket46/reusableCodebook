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
    plus the regions [-∞, prev_min] and [prev_max, +∞]) and place one new level
    at the midpoint of each.

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
    and levels[b1] ⊂ levels[b2] for b1 < b2.
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


# ============================================================================
# Evaluation
# ============================================================================

def empirical_mse(levels: np.ndarray, n_samples: int = 1_000_000,
                  seed: int = 0) -> float:
    """Empirical MSE of the codebook on N(0, 1) samples."""
    rng = np.random.default_rng(seed)
    samples = sample_gaussian(n_samples, rng)
    boundaries = voronoi_boundaries(levels) if len(levels) > 1 else np.array([])
    idx = np.digitize(samples, boundaries) if len(levels) > 1 else np.zeros_like(samples, dtype=np.int64)
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

    # Hierarchical (nested) Lloyd-Max
    print("\nHierarchical (nested) Lloyd-Max levels:")
    hier = build_hierarchical_codebooks(BITS)
    for b in BITS:
        print(f"  b={b}  ({2**b:>2d} levels): {np.array2string(hier[b], precision=4, suppress_small=True)}")

    ok = verify_nesting(hier)
    print(f"\n  Nesting verified: {ok}")

    print("\n" + "=" * 78)
    print("MSE comparison: native vs hierarchical Lloyd-Max")
    print("=" * 78)
    print(f"{'b':>3s} {'native MSE':>14s} {'hier MSE':>14s} "
          f"{'gap':>12s} {'gap %':>10s}")
    print("-" * 60)
    rows = [("bits", "native_mse", "hier_mse", "gap", "gap_pct")]
    for b in BITS:
        mse_native = empirical_mse(native[b])
        mse_hier = empirical_mse(hier[b])
        gap = mse_hier - mse_native
        gap_pct = 100 * gap / mse_native if mse_native > 0 else 0
        print(f"{b:>3d} {mse_native:>14.6f} {mse_hier:>14.6f} "
              f"{gap:>+12.6f} {gap_pct:>+9.2f}%")
        rows.append((b, mse_native, mse_hier, gap, gap_pct))

    # Save codebooks
    save_dict = {f"native_b{b}": native[b] for b in BITS}
    save_dict.update({f"hier_b{b}": hier[b] for b in BITS})
    np.savez(CODEBOOK_DIR / "hierarchical_gaussian.npz", **save_dict)
    print(f"\nCodebooks saved: {CODEBOOK_DIR / 'hierarchical_gaussian.npz'}")

    # Save metrics CSV
    metrics_path = CODEBOOK_DIR / "hierarchical_metrics.csv"
    with open(metrics_path, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"Metrics CSV:     {metrics_path}")

    # ------------------------------------------------------------------
    # Visualization 1: codebook nesting structure
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 6))
    for b in BITS:
        levels = hier[b]
        y = b * np.ones_like(levels)
        ax.scatter(levels, y, s=80, zorder=3, label=f"b={b} ({2**b} levels)")
        for lvl in levels:
            ax.text(lvl, b - 0.18, f"{lvl:+.2f}",
                    ha="center", va="top", fontsize=7, alpha=0.7)
    # Connect each level to its presence at higher bits (vertical lines)
    for i, b in enumerate(BITS[:-1]):
        for lvl in hier[b]:
            ax.plot([lvl, lvl], [b, b + 1], "k-", alpha=0.25, linewidth=1)
    ax.set_xlabel("Codebook level (in standardized coordinate space)")
    ax.set_ylabel("bits b")
    ax.set_yticks(BITS)
    ax.set_xlim(-3.5, 3.5)
    ax.set_title(
        "Hierarchical Lloyd-Max codebooks for N(0, 1)\n"
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
    # Visualization 2: native vs hierarchical levels side-by-side
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, codebook_set, title in [
        (axes[0], native, "Native (unconstrained) Lloyd-Max"),
        (axes[1], hier, "Hierarchical (nested) Lloyd-Max"),
    ]:
        for b in BITS:
            levels = codebook_set[b]
            y = b * np.ones_like(levels)
            ax.scatter(levels, y, s=70, zorder=3, label=f"b={b}")
        # Reference: pdf of N(0, 1) at the bottom
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
    fig.suptitle("Native vs hierarchical codebook structure (N(0, 1) pdf in gray)",
                 fontsize=12)
    fig.tight_layout()
    out2 = FIGS / "hierarchical_vs_native.png"
    fig.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved:    {out2}")

    # ------------------------------------------------------------------
    # Visualization 3: MSE bar plot — the headline result
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(BITS))
    width = 0.35
    native_mses = [empirical_mse(native[b]) for b in BITS]
    hier_mses = [empirical_mse(hier[b]) for b in BITS]
    bars1 = ax.bar(x - width / 2, native_mses, width,
                   label="Native Lloyd-Max", color="tab:blue", alpha=0.85)
    bars2 = ax.bar(x + width / 2, hier_mses, width,
                   label="Hierarchical Lloyd-Max", color="tab:orange", alpha=0.85)
    # Annotate gap_pct
    for i, b in enumerate(BITS):
        pct = 100 * (hier_mses[i] - native_mses[i]) / native_mses[i]
        ax.text(x[i] + width / 2, hier_mses[i],
                f"+{pct:.1f}%", ha="center", va="bottom", fontsize=9, color="darkred")
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