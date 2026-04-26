#!/usr/bin/env python
"""Shared config loading helpers for experiment scripts."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.yaml"


def load_config() -> dict[str, Any]:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if "datasets" not in cfg or not isinstance(cfg["datasets"], dict):
        raise ValueError("config.yaml must define a 'datasets' mapping")
    return cfg


def enabled_datasets(cfg: dict[str, Any]) -> list[str]:
    env = os.environ.get("DATASETS", "").strip()
    if env:
        names = env.split()
    else:
        names = list(cfg["datasets"].keys())
    unknown = [n for n in names if n not in cfg["datasets"]]
    if unknown:
        raise ValueError(f"Unknown datasets in DATASETS: {unknown}")
    return names


def ensure_int_list(values: Any, field_name: str) -> list[int]:
    if not isinstance(values, list) or not values:
        raise ValueError(f"{field_name} must be a non-empty list")
    out = [int(v) for v in values]
    if any(v <= 0 for v in out):
        raise ValueError(f"{field_name} values must be positive")
    return out

