#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path


UNIT_TO_MOLAR = {
    "m": 1.0,
    "mol": 1.0,
    "mm": 1e-3,
    "millimolar": 1e-3,
    "um": 1e-6,
    "µm": 1e-6,
    "μm": 1e-6,
    "nm": 1e-9,
    "pm": 1e-12,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build low-activity to high-activity ChEMBL H2L pair CSVs.")
    parser.add_argument("--root", type=Path, required=True, help="Extracted chembl_h2l transfer directory.")
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--min-delta", type=float, default=0.5, help="Minimum pActivity improvement.")
    parser.add_argument("--max-pairs-per-series", type=int, default=64)
    parser.add_argument("--include-splits", nargs="*", default=["train"], help="Dataset splits to include.")
    parser.add_argument("--pairs-name", default="pairs.csv")
    parser.add_argument("--summary-name", default="summary.csv")
    return parser.parse_args()


def norm_unit(unit: str) -> str:
    return unit.strip().lower().replace(" ", "").replace("µ", "u").replace("μ", "u")


def activity_score(row: dict[str, str]) -> float | None:
    raw = row.get("affinity", "").strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    if not math.isfinite(value) or value <= 0:
        return None

    affinity_type = row.get("affinity_type", "").strip().lower()
    unit = norm_unit(row.get("affinity_units", ""))
    if affinity_type.startswith("p") or unit.startswith("p"):
        return value

    factor = UNIT_TO_MOLAR.get(unit)
    if factor is None:
        return None
    molar = value * factor
    if molar <= 0:
        return None
    return -math.log10(molar)


def csv_files(root: Path, splits: set[str]) -> list[Path]:
    files = []
    for path in sorted(root.glob("round*/**/*.csv")):
        if path.name.endswith("_scaffold_info_20241120.csv"):
            continue
        if path.stem in splits:
            files.append(path)
    return files


def ligand_path_for_row(csv_path: Path, row: dict[str, str]) -> Path | None:
    split = csv_path.stem
    series_id = row.get("sries_id", "").strip()
    identifier = row.get("identifier", "").strip()
    if not series_id or not identifier:
        return None
    path = csv_path.parent / split / series_id / f"{identifier}.sdf"
    return path if path.exists() else None


def protein_path_for_row(csv_path: Path, row: dict[str, str]) -> Path | None:
    split = csv_path.stem
    series_id = row.get("sries_id", "").strip()
    if not series_id:
        return None
    path = csv_path.parent / split / series_id / f"{series_id}_prep.pdb"
    return path if path.exists() else None


def iter_rows(root: Path, splits: set[str]):
    for csv_path in csv_files(root, splits):
        round_id = csv_path.parent.name
        split = csv_path.stem
        with csv_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                score = activity_score(row)
                ligand_path = ligand_path_for_row(csv_path, row)
                protein_path = protein_path_for_row(csv_path, row)
                if score is None or ligand_path is None or protein_path is None:
                    continue
                yield {
                    **row,
                    "round_id": round_id,
                    "split": split,
                    "activity_score": score,
                    "ligand_path": ligand_path,
                    "protein_path": protein_path,
                }


def build_pairs(rows: list[dict], min_delta: float, max_pairs_per_series: int) -> tuple[list[dict], list[dict]]:
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["round_id"], row["split"], row["sries_id"])].append(row)

    pairs = []
    summary = []
    for (round_id, split, series_id), group in sorted(grouped.items()):
        unique = {}
        for row in group:
            unique[row["identifier"]] = row
        group = sorted(unique.values(), key=lambda r: (r["activity_score"], r["identifier"]))
        candidates = []
        for low in group:
            for high in group:
                delta = high["activity_score"] - low["activity_score"]
                if delta >= min_delta:
                    candidates.append((delta, low, high))
        candidates.sort(key=lambda item: (-item[0], item[1]["identifier"], item[2]["identifier"]))
        selected = candidates[:max_pairs_per_series]
        for pair_index, (delta, low, high) in enumerate(selected):
            record_id = f"{round_id}_{split}_{series_id}_{low['identifier']}_to_{high['identifier']}_{pair_index:03d}"
            pairs.append(
                {
                    "record_id": record_id,
                    "round_id": round_id,
                    "split": split,
                    "target_id": low.get("pdbid", ""),
                    "series_id": series_id,
                    "protein_path": str(low["protein_path"].resolve()),
                    "source_ligand_path": str(low["ligand_path"].resolve()),
                    "target_ligand_path": str(high["ligand_path"].resolve()),
                    "source_identifier": low["identifier"],
                    "target_identifier": high["identifier"],
                    "source_activity": f"{low['activity_score']:.6g}",
                    "target_activity": f"{high['activity_score']:.6g}",
                    "delta_activity": f"{delta:.6g}",
                    "source_affinity": low.get("affinity", ""),
                    "target_affinity": high.get("affinity", ""),
                    "affinity_units": high.get("affinity_units", ""),
                    "affinity_type": high.get("affinity_type", ""),
                    "uniprot": high.get("uniprot", ""),
                }
            )
        summary.append(
            {
                "round_id": round_id,
                "split": split,
                "series_id": series_id,
                "molecules": len(group),
                "candidate_pairs": len(candidates),
                "selected_pairs": len(selected),
                "min_activity": f"{group[0]['activity_score']:.6g}" if group else "",
                "max_activity": f"{group[-1]['activity_score']:.6g}" if group else "",
            }
        )
    return pairs, summary


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    splits = set(args.include_splits)
    rows = list(iter_rows(args.root, splits))
    pairs, summary = build_pairs(rows, args.min_delta, args.max_pairs_per_series)

    pair_fields = [
        "record_id",
        "round_id",
        "split",
        "target_id",
        "series_id",
        "protein_path",
        "source_ligand_path",
        "target_ligand_path",
        "source_identifier",
        "target_identifier",
        "source_activity",
        "target_activity",
        "delta_activity",
        "source_affinity",
        "target_affinity",
        "affinity_units",
        "affinity_type",
        "uniprot",
    ]
    summary_fields = [
        "round_id",
        "split",
        "series_id",
        "molecules",
        "candidate_pairs",
        "selected_pairs",
        "min_activity",
        "max_activity",
    ]
    write_csv(args.outdir / args.pairs_name, pairs, pair_fields)
    write_csv(args.outdir / args.summary_name, summary, summary_fields)

    split_counts = Counter(row["split"] for row in pairs)
    print(f"input_molecules: {len(rows)}")
    print(f"series: {len(summary)}")
    print(f"pairs: {len(pairs)}")
    print("pair_splits: " + ", ".join(f"{k}={v}" for k, v in sorted(split_counts.items())))
    print(f"pairs_csv: {args.outdir / args.pairs_name}")
    print(f"summary_csv: {args.outdir / args.summary_name}")


if __name__ == "__main__":
    main()
