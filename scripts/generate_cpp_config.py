#!/usr/bin/env python
"""Generate C++ experiment config header from config.yaml."""
from __future__ import annotations

from pathlib import Path

from config_utils import ensure_int_list, load_config

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "cpp" / "experiment_config.hpp"


def main() -> int:
    cfg = load_config()
    eval_ks = ensure_int_list(cfg.get("eval_ks", []), "eval_ks")
    topk = int(cfg.get("cpp", {}).get("topk", max(eval_ks)))
    rounds = int(cfg.get("search_rounds", 3))
    high_acc = bool(cfg.get("cpp", {}).get("high_acc_fast_scan", True))

    if not high_acc:
        raise ValueError("cpp.high_acc_fast_scan=false is unsupported for this repo")
    if topk < max(eval_ks):
        raise ValueError("cpp.topk must be >= max(eval_ks)")
    if rounds <= 0:
        raise ValueError("search_rounds must be > 0")

    body = f"""#pragma once
#include <array>
#include <cstddef>

#define HIGH_ACC_FAST_SCAN

static constexpr std::array<size_t, {len(eval_ks)}> EXP_EVAL_KS = {{{', '.join(str(k) for k in eval_ks)}}};
static constexpr size_t EXP_EVAL_KMAX = {max(eval_ks)};
static constexpr size_t EXP_TOPK = {topk};
static constexpr size_t EXP_ROUND = {rounds};
"""
    OUT.write_text(body, encoding="utf-8")
    print(f"generated {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

