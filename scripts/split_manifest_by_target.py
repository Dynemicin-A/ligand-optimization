#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split a .pt manifest by target_id to avoid target leakage.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--heldout-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--train-name", type=str, default="train_manifest.txt")
    parser.add_argument("--heldout-name", type=str, default="heldout_manifest.txt")
    parser.add_argument("--summary-name", type=str, default="split_summary.csv")
    return parser.parse_args()


def read_manifest(path: Path) -> list[Path]:
    base = path.parent
    paths = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        item = Path(line)
        paths.append(item if item.is_absolute() else base / item)
    if not paths:
        raise ValueError(f"empty manifest: {path}")
    return paths


def target_id(path: Path) -> str:
    record = torch.load(path, map_location="cpu", weights_only=False)
    target = record.get("target_id")
    if target:
        return str(target)
    return path.parent.name


def main() -> None:
    args = parse_args()
    if not 0.0 < args.heldout_fraction < 1.0:
        raise ValueError("--heldout-fraction must be in (0, 1)")
    args.outdir.mkdir(parents=True, exist_ok=True)

    by_target: dict[str, list[Path]] = defaultdict(list)
    for path in read_manifest(args.manifest):
        by_target[target_id(path)].append(path)

    targets = sorted(by_target)
    rng = random.Random(args.seed)
    rng.shuffle(targets)
    n_heldout = max(1, round(len(targets) * args.heldout_fraction))
    heldout_targets = set(targets[:n_heldout])

    train_paths = []
    heldout_paths = []
    rows = []
    for target in sorted(by_target):
        split = "heldout" if target in heldout_targets else "train"
        paths = by_target[target]
        if split == "heldout":
            heldout_paths.extend(paths)
        else:
            train_paths.extend(paths)
        rows.append({"target_id": target, "split": split, "records": len(paths)})

    (args.outdir / args.train_name).write_text("\n".join(str(path) for path in train_paths) + "\n")
    (args.outdir / args.heldout_name).write_text("\n".join(str(path) for path in heldout_paths) + "\n")
    with (args.outdir / args.summary_name).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["target_id", "split", "records"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"targets: {len(targets)}")
    print(f"train_targets: {len(targets) - len(heldout_targets)}")
    print(f"heldout_targets: {len(heldout_targets)}")
    print(f"train_records: {len(train_paths)}")
    print(f"heldout_records: {len(heldout_paths)}")
    print(f"outdir: {args.outdir}")


if __name__ == "__main__":
    main()
