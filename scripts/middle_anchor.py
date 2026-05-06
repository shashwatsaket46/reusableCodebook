#!/usr/bin/env python
"""Middle-anchor hierarchical Lloyd-Max codebooks for N(0, 1).

Construction:
  1. Fix b_anchor (default b=3) as unconstrained-optimal.
  2. For b > b_anchor: add new levels via constrained Lloyd-Max (upward).
  3. For b < b_anchor: SELECT a subset of b_anchor's levels via greedy MSE
     minimization (downward).

Result: every codebook contains b_anchor's levels. Only b_anchor itself is
unconstrained-optimal. Lower and higher bit-widths bear penalty, but spread
more evenly across the design space than bottom-up or top-down alone.
"""
from __future__ import annotations

from pathlib import Path
import csv
import sys

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from hierarchical_codebook import (
    lloyd_max_unconstrained,
    lloyd_max_constrained,
    build_hierarchical_codebooks,
    build_top_down_codebooks,
    empirical_mse,
    verify_nesting,
    _initialize_new_levels,
)

CODEBOOK_DIR = ROOT / "results" / "codebooks"
FIGS = ROOT / "results" / "figures"
CODEBOOK_DIR.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Greedy subset selection: pick k of n levels minimizing MSE
# ============================================================================

def select_subset_greedy(
        levels: np.ndarray,
        k: int,
        n_samples: int = 500_000,
        seed: int = 42,
) -> np.ndarray:
    """Select k levels from `levels` that minimize empirical MSE on N(0,1).

    Greedy backward elimination: start from full set, repeatedly remove the
    level whose removal minimizes the resulting MSE. Stops when k remain.
    """
    levels = np.array(levels, dtype=float)
    if k >= len(levels):
        return np.sort(levels.copy())
    rng = np.random.default_rng(seed)
    samples = rng.standard_normal(n_samples)

    keep_mask = np.ones(len(levels), dtype=bool)
    while keep_mask.sum() > k:
        best_mse = np.inf
        best_idx_to_remove = -1
        kept_indices = np.where(keep_mask)[0]
        for idx in kept_indices:
            trial_mask = keep_mask.copy()
            trial_mask[idx] = False
            trial_levels = np.sort(levels[trial_mask])
            # Compute MSE without re-sampling each time
            if len(trial_levels) == 1:
                mse = float(np.mean((samples - trial_levels[0]) ** 2))
            else:
                boundaries = (trial_levels[:-1] + trial_levels[1:]) / 2.0
                idx_q = np.digitize(samples, boundaries)
                quantized = trial_levels[idx_q]
                mse = float(np.mean((samples - quantized) ** 2))
            if mse < best_mse:
                best_mse = mse
                best_idx_to_remove = idx
        keep_mask[best_idx_to_remove] = False

    return np.sort(levels[keep_mask])


def build_middle_anchor_codebooks(
        bit_widths: list[int], anchor_bits: int = 3, seed: int = 42,
) -> dict[int, np.ndarray]:
    """Middle-anchor: fix b=anchor_bits unconstrained, expand both directions.

    Returns dict {b: levels} with all codebooks containing the anchor levels.
    """
    bit_widths = sorted(bit_widths)
    if anchor_bits not in bit_widths:
        raise ValueError(
            f"anchor_bits {anchor_bits} must be in bit_widths {bit_widths}"
        )
    if anchor_bits == bit_widths[0]:
        raise ValueError(
            "middle_anchor with anchor=min is just bottom_up — use that instead"
        )
    if anchor_bits == bit_widths[-1]:
        raise ValueError(
            "middle_anchor with anchor=max is just top_down — use that instead"
        )

    codebooks = {}

    # Anchor: unconstrained-optimal
    anchor_levels = lloyd_max_unconstrained(2 ** anchor_bits, seed=seed)
    codebooks[anchor_bits] = anchor_levels

    # Lower than anchor: greedy subset selection (downward chain)
    bs_below = sorted([b for b in bit_widths if b < anchor_bits], reverse=True)
    for b in bs_below:
        codebooks[b] = select_subset_greedy(anchor_levels, 2 ** b, seed=seed)

    # Higher than anchor: constrained add (upward chain)
    bs_above = sorted([b for b in bit_widths if b > anchor_bits])
    prev_levels = anchor_levels
    for b in bs_above:
        n_new = 2 ** b - len(prev_levels)
        new_init = _initialize_new_levels(prev_levels, n_new)
        codebooks[b] = lloyd_max_constrained(
            fixed_levels=prev_levels, new_levels_init=new_init, seed=seed,
        )
        prev_levels = codebooks[b]

    return codebooks


# ============================================================================
# Main: compare all four strategies
# ============================================================================

def main():
    BITS = [2, 3, 4, 5]
    ANCHOR_BITS = 3

    print("=" * 78)
    print("Three nested Lloyd-Max strategies for N(0, 1) — comparison")
    print("=" * 78)

    print("\nBuilding codebooks...")
    native = {b: lloyd_max_unconstrained(2 ** b) for b in BITS}
    bu = build_hierarchical_codebooks(BITS)
    td = build_top_down_codebooks(BITS)
    ma = build_middle_anchor_codebooks(BITS, anchor_bits=ANCHOR_BITS)

    print(f"  Bottom-up nesting verified: {verify_nesting(bu)}")
    print(f"  Top-down nesting verified: {verify_nesting(td)}")
    print(f"  Middle-anchor (b={ANCHOR_BITS}) nesting verified: {verify_nesting(ma)}")

    print("\nLevels by strategy:")
    for name, cbs in [("native", native), ("bottom_up", bu),
                      ("top_down", td), ("middle_anchor", ma)]:
        print(f"\n  {name}:")
        for b in BITS:
            print(f"    b={b}  ({2**b:>2d} levels): "
                  f"{np.array2string(cbs[b], precision=3, suppress_small=True)}")

    print("\n" + "=" * 78)
    print(f"MSE comparison (anchor for middle = b={ANCHOR_BITS})")
    print("=" * 78)
    print(f"{'b':>3s} {'native':>10s} {'bottom_up':>10s} {'top_down':>10s} "
          f"{'mid_anchor':>11s} {'BU%':>7s} {'TD%':>7s} {'MA%':>7s}")
    print("-" * 80)
    rows = [("bits", "native", "bottom_up", "top_down", "middle_anchor",
             "bu_pct", "td_pct", "ma_pct")]
    for b in BITS:
        n = empirical_mse(native[b])
        bu_v = empirical_mse(bu[b])
        td_v = empirical_mse(td[b])
        ma_v = empirical_mse(ma[b])
        bu_pct = 100 * (bu_v - n) / n
        td_pct = 100 * (td_v - n) / n
        ma_pct = 100 * (ma_v - n) / n
        print(f"{b:>3d} {n:>10.5f} {bu_v:>10.5f} {td_v:>10.5f} "
              f"{ma_v:>11.5f} {bu_pct:>+6.1f}% {td_pct:>+6.1f}% {ma_pct:>+6.1f}%")
        rows.append((b, n, bu_v, td_v, ma_v, bu_pct, td_pct, ma_pct))

    save_dict = {f"native_b{b}": native[b] for b in BITS}
    save_dict.update({f"bu_b{b}": bu[b] for b in BITS})
    save_dict.update({f"td_b{b}": td[b] for b in BITS})
    save_dict.update({f"ma_b{b}": ma[b] for b in BITS})
    np.savez(CODEBOOK_DIR / "all_strategies.npz", **save_dict)
    print(f"\nSaved: {CODEBOOK_DIR / 'all_strategies.npz'}")

    csv_path = CODEBOOK_DIR / "all_strategies_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"Saved: {csv_path}")

    # Plot: MSE penalty across all four strategies
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(BITS))
    width = 0.2

    native_mses = [empirical_mse(native[b]) for b in BITS]
    bu_mses = [empirical_mse(bu[b]) for b in BITS]
    td_mses = [empirical_mse(td[b]) for b in BITS]
    ma_mses = [empirical_mse(ma[b]) for b in BITS]

    ax.bar(x - 1.5 * width, native_mses, width,
           label="native", color="tab:gray", alpha=0.85)
    ax.bar(x - 0.5 * width, bu_mses, width,
           label="bottom-up", color="tab:blue", alpha=0.85)
    ax.bar(x + 0.5 * width, ma_mses, width,
           label=f"middle-anchor (b={ANCHOR_BITS})", color="tab:green", alpha=0.85)
    ax.bar(x + 1.5 * width, td_mses, width,
           label="top-down", color="tab:orange", alpha=0.85)

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"b={b}" for b in BITS])
    ax.set_xlabel("bits per coordinate", fontsize=12)
    ax.set_ylabel("MSE per coord (log)", fontsize=12)
    ax.set_title(f"Nested Lloyd-Max strategies — MSE penalty by bit-width\n"
                 f"(middle-anchor at b={ANCHOR_BITS})", fontsize=12)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out = FIGS / "all_strategies_mse.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()