#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold


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
    parser.add_argument(
        "--min-tanimoto",
        type=float,
        default=0.0,
        help="Minimum source/target Morgan Tanimoto similarity. Use 0 to disable.",
    )
    parser.add_argument(
        "--require-same-scaffold",
        action="store_true",
        help="Require source and target to share the same nonempty Murcko scaffold.",
    )
    parser.add_argument(
        "--max-heavy-delta",
        type=int,
        default=None,
        help="Maximum absolute target-source heavy atom count difference.",
    )
    parser.add_argument(
        "--max-source-reuse-per-series",
        type=int,
        default=None,
        help="Maximum selected pairs per source ligand within one series.",
    )
    parser.add_argument(
        "--max-target-reuse-per-series",
        type=int,
        default=None,
        help="Maximum selected pairs per target ligand within one series.",
    )
    parser.add_argument(
        "--allow-affinity-mismatch",
        action="store_true",
        help="Allow pair source/target affinity type or unit mismatch. Disabled by default for clean H2L.",
    )
    parser.add_argument("--include-splits", nargs="*", default=["train"], help="Dataset splits to include.")
    parser.add_argument("--pairs-name", default="pairs.csv")
    parser.add_argument("--summary-name", default="summary.csv")
    return parser.parse_args()


def norm_unit(unit: str) -> str:
    return unit.strip().lower().replace(" ", "").replace("µ", "u").replace("μ", "u")


def norm_affinity_type(affinity_type: str) -> str:
    return affinity_type.strip().lower()


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


MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def load_mol(path: Path) -> Chem.Mol | None:
    supplier = Chem.SDMolSupplier(str(path), sanitize=True, removeHs=False)
    for mol in supplier:
        if mol is not None:
            return mol
    return None


def mol_props(path: Path) -> dict:
    mol = load_mol(path)
    if mol is None:
        return {"load_failed": True}
    heavy = Chem.RemoveHs(mol)
    scaffold_mol = MurckoScaffold.GetScaffoldForMol(heavy)
    scaffold = Chem.MolToSmiles(scaffold_mol, isomericSmiles=False) if scaffold_mol.GetNumAtoms() else ""
    return {
        "load_failed": False,
        "heavy_atoms": int(heavy.GetNumAtoms()),
        "fingerprint": MORGAN_GENERATOR.GetFingerprint(heavy),
        "scaffold": scaffold,
    }


def pair_similarity(low: dict, high: dict) -> float:
    low_fp = low["_mol_props"]["fingerprint"]
    high_fp = high["_mol_props"]["fingerprint"]
    return float(DataStructs.TanimotoSimilarity(low_fp, high_fp))


def same_nonempty_scaffold(low: dict, high: dict) -> bool:
    low_scaffold = low["_mol_props"].get("scaffold", "")
    high_scaffold = high["_mol_props"].get("scaffold", "")
    return bool(low_scaffold and high_scaffold and low_scaffold == high_scaffold)


def same_affinity_context(low: dict, high: dict) -> bool:
    return (
        norm_affinity_type(low.get("affinity_type", "")) == norm_affinity_type(high.get("affinity_type", ""))
        and norm_unit(low.get("affinity_units", "")) == norm_unit(high.get("affinity_units", ""))
    )


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
                props = mol_props(ligand_path)
                if props.get("load_failed"):
                    continue
                yield {
                    **row,
                    "round_id": round_id,
                    "split": split,
                    "activity_score": score,
                    "ligand_path": ligand_path,
                    "protein_path": protein_path,
                    "_mol_props": props,
                }


def build_pairs(
    rows: list[dict],
    min_delta: float,
    max_pairs_per_series: int,
    *,
    min_tanimoto: float,
    require_same_scaffold: bool,
    max_heavy_delta: int | None,
    max_source_reuse_per_series: int | None,
    max_target_reuse_per_series: int | None,
    allow_affinity_mismatch: bool,
) -> tuple[list[dict], list[dict]]:
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
        rejected = Counter()
        for low in group:
            for high in group:
                delta = high["activity_score"] - low["activity_score"]
                if delta < min_delta:
                    continue
                if not allow_affinity_mismatch and not same_affinity_context(low, high):
                    rejected["affinity_mismatch"] += 1
                    continue
                tanimoto = pair_similarity(low, high)
                if tanimoto < min_tanimoto:
                    rejected["low_tanimoto"] += 1
                    continue
                scaffold_match = same_nonempty_scaffold(low, high)
                if require_same_scaffold and not scaffold_match:
                    rejected["scaffold_mismatch"] += 1
                    continue
                source_heavy = low["_mol_props"]["heavy_atoms"]
                target_heavy = high["_mol_props"]["heavy_atoms"]
                heavy_delta = target_heavy - source_heavy
                if max_heavy_delta is not None and abs(heavy_delta) > max_heavy_delta:
                    rejected["heavy_delta"] += 1
                    continue
                candidates.append((delta, tanimoto, scaffold_match, heavy_delta, source_heavy, target_heavy, low, high))
        candidates.sort(key=lambda item: (-item[0], -item[1], abs(item[3]), item[6]["identifier"], item[7]["identifier"]))
        selected = []
        source_reuse: Counter[str] = Counter()
        target_reuse: Counter[str] = Counter()
        for candidate in candidates:
            if len(selected) >= max_pairs_per_series:
                break
            low = candidate[6]
            high = candidate[7]
            source_id = low["identifier"]
            target_id = high["identifier"]
            if max_source_reuse_per_series is not None and source_reuse[source_id] >= max_source_reuse_per_series:
                rejected["source_reuse"] += 1
                continue
            if max_target_reuse_per_series is not None and target_reuse[target_id] >= max_target_reuse_per_series:
                rejected["target_reuse"] += 1
                continue
            selected.append(candidate)
            source_reuse[source_id] += 1
            target_reuse[target_id] += 1
        for pair_index, (delta, tanimoto, scaffold_match, heavy_delta, source_heavy, target_heavy, low, high) in enumerate(selected):
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
                    "source_affinity_units": low.get("affinity_units", ""),
                    "target_affinity_units": high.get("affinity_units", ""),
                    "source_affinity_type": low.get("affinity_type", ""),
                    "target_affinity_type": high.get("affinity_type", ""),
                    "affinity_units": high.get("affinity_units", ""),
                    "affinity_type": high.get("affinity_type", ""),
                    "tanimoto": f"{tanimoto:.6g}",
                    "same_scaffold": "1" if scaffold_match else "0",
                    "source_heavy_atoms": str(source_heavy),
                    "target_heavy_atoms": str(target_heavy),
                    "heavy_atom_delta": str(heavy_delta),
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
                "rejected_low_tanimoto": rejected.get("low_tanimoto", 0),
                "rejected_scaffold_mismatch": rejected.get("scaffold_mismatch", 0),
                "rejected_heavy_delta": rejected.get("heavy_delta", 0),
                "rejected_affinity_mismatch": rejected.get("affinity_mismatch", 0),
                "rejected_source_reuse": rejected.get("source_reuse", 0),
                "rejected_target_reuse": rejected.get("target_reuse", 0),
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


def group_by_split(rows: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["split"]].append(row)
    return grouped


def main() -> None:
    args = parse_args()
    splits = set(args.include_splits)
    rows = list(iter_rows(args.root, splits))
    pairs, summary = build_pairs(
        rows,
        args.min_delta,
        args.max_pairs_per_series,
        min_tanimoto=args.min_tanimoto,
        require_same_scaffold=args.require_same_scaffold,
        max_heavy_delta=args.max_heavy_delta,
        max_source_reuse_per_series=args.max_source_reuse_per_series,
        max_target_reuse_per_series=args.max_target_reuse_per_series,
        allow_affinity_mismatch=args.allow_affinity_mismatch,
    )

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
        "source_affinity_units",
        "target_affinity_units",
        "source_affinity_type",
        "target_affinity_type",
        "affinity_units",
        "affinity_type",
        "tanimoto",
        "same_scaffold",
        "source_heavy_atoms",
        "target_heavy_atoms",
        "heavy_atom_delta",
        "uniprot",
    ]
    summary_fields = [
        "round_id",
        "split",
        "series_id",
        "molecules",
        "candidate_pairs",
        "selected_pairs",
        "rejected_low_tanimoto",
        "rejected_scaffold_mismatch",
        "rejected_heavy_delta",
        "rejected_affinity_mismatch",
        "rejected_source_reuse",
        "rejected_target_reuse",
        "min_activity",
        "max_activity",
    ]
    write_csv(args.outdir / args.pairs_name, pairs, pair_fields)
    for split, split_rows in sorted(group_by_split(pairs).items()):
        write_csv(args.outdir / f"{split}_pairs.csv", split_rows, pair_fields)
    write_csv(args.outdir / args.summary_name, summary, summary_fields)

    split_counts = Counter(row["split"] for row in pairs)
    print(f"input_molecules: {len(rows)}")
    print(f"series: {len(summary)}")
    print(f"pairs: {len(pairs)}")
    print("pair_splits: " + ", ".join(f"{k}={v}" for k, v in sorted(split_counts.items())))
    print(f"pairs_csv: {args.outdir / args.pairs_name}")
    for split, count in sorted(split_counts.items()):
        print(f"{split}_pairs_csv: {args.outdir / f'{split}_pairs.csv'} ({count})")
    print(f"summary_csv: {args.outdir / args.summary_name}")


if __name__ == "__main__":
    main()
