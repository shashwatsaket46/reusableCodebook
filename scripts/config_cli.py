#!/usr/bin/env python
"""CLI bridge so shell scripts can read values from config.yaml."""
from __future__ import annotations

import sys

from config_utils import enabled_datasets, ensure_int_list, load_config


def _print_list(vals: list[int | str]) -> None:
    print(" ".join(str(v) for v in vals))


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: config_cli.py <command> [args...]", file=sys.stderr)
        return 2

    cfg = load_config()
    cmd = sys.argv[1]

    if cmd == "datasets":
        _print_list(enabled_datasets(cfg))
        return 0
    if cmd == "dataset_nlist":
        if len(sys.argv) != 3:
            print("usage: config_cli.py dataset_nlist <name>", file=sys.stderr)
            return 2
        print(int(cfg["datasets"][sys.argv[2]]["nlist"]))
        return 0
    if cmd == "exrabitq_bits":
        _print_list(ensure_int_list(cfg.get("exrabitq_bits", []), "exrabitq_bits"))
        return 0
    if cmd == "pq_bits":
        _print_list(ensure_int_list(cfg.get("pq_bits", []), "pq_bits"))
        return 0

    print(f"unknown command: {cmd}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

