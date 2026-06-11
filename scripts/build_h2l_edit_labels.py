#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import torch
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pco_backbone.records import build_edit_labels, coerce_model_record  # noqa: E402


LABEL_NAMES = {
    0: "copy",
    1: "mutate",
    2: "move",
    3: "grow",
}


def read_manifest(path: Path) -> list[Path]:
    base = path.parent
    paths = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        item = Path(line)
        paths.append(item if item.is_absolute() else base / item)
    return paths


def relative_or_absolute(path: Path, base: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(path.resolve())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build weak source-target edit labels for H2L records.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--copy-threshold", type=float, default=1.25)
    parser.add_argument("--move-threshold", type=float, default=3.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--copy-existing-metadata", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    paths = read_manifest(args.manifest)
    if args.limit is not None:
        paths = paths[: args.limit]
    written: list[Path] = []
    label_counts: Counter[int] = Counter()
    delete_counts: Counter[int] = Counter()
    failures: list[dict[str, str]] = []

    for idx, path in enumerate(tqdm(paths, desc="edit-labels")):
        try:
            raw = torch.load(path, map_location="cpu", weights_only=False)
            record = coerce_model_record(raw, keep_metadata=True)
            labels = build_edit_labels(
                ligand_atom_type=record["ligand_atom_type"],
                ligand_pos=record["ligand_pos"],
                source_atom_type=record["source_atom_type"],
                source_pos=record["source_pos"],
                copy_threshold=args.copy_threshold,
                move_threshold=args.move_threshold,
            )
            out_record = dict(raw) if args.copy_existing_metadata else dict(record)
            out_record.update(labels)
            label_counts.update(int(value) for value in labels["ligand_edit_label"].tolist())
            delete_counts.update(int(value) for value in labels["source_delete_label"].tolist())
            out_path = args.outdir / f"{idx:08d}_{path.name}"
            torch.save(out_record, out_path)
            written.append(out_path)
        except Exception as exc:
            failures.append(
                {
                    "path": str(path),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    manifest_path = args.outdir / "manifest.txt"
    manifest_path.write_text("\n".join(relative_or_absolute(path, args.outdir) for path in written) + "\n")
    summary = {
        "source_manifest": str(args.manifest),
        "outdir": str(args.outdir),
        "written": len(written),
        "failed": len(failures),
        "copy_threshold": args.copy_threshold,
        "move_threshold": args.move_threshold,
        "ligand_edit_label_counts": {
            LABEL_NAMES.get(key, str(key)): value for key, value in sorted(label_counts.items())
        },
        "source_delete_label_counts": {
            "kept": delete_counts.get(0, 0),
            "delete": delete_counts.get(1, 0),
        },
        "failures": failures[:50],
    }
    (args.outdir / "edit_label_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if failures:
        shutil.copyfile(args.outdir / "edit_label_summary.json", args.outdir / "edit_label_failures.json")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
