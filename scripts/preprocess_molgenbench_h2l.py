#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pco_backbone.chem import (  # noqa: E402
    crop_protein_to_ligand,
    load_first_mol,
    mol_to_record_tensors,
    parse_pdb_atoms,
)


@dataclass(frozen=True)
class H2LSeries:
    target_id: str
    series_id: str
    protein_path: Path
    source_ligand_path: Path
    target_ligand_path: Path


@dataclass(frozen=True)
class SeriesMol:
    label: str
    index: int
    mol: Chem.Mol
    path: Path
    target_index: int | None
    score: float | None


@dataclass(frozen=True)
class PairDef:
    source_label: str
    source_index: int
    source_mol: Chem.Mol
    source_path: Path
    target_index: int
    target_mol: Chem.Mol
    source_score: float | None
    target_score: float | None
    negative_label: str | None = None
    negative_index: int | None = None
    negative_mol: Chem.Mol | None = None
    negative_path: Path | None = None
    negative_score: float | None = None
    negative_similarity: float | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess MolGenBench v3 H2L reference series into .pt records.")
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT / "data/raw/molgenbench_v3/extracted/MolGenBench_Version3",
        help="Extracted MolGenBench_Version3 directory.",
    )
    parser.add_argument("--outdir", type=Path, default=ROOT / "data/processed_h2l/train")
    parser.add_argument("--manifest-name", type=str, default="manifest.txt")
    parser.add_argument("--pairs-name", type=str, default="pairs.csv")
    parser.add_argument("--failures-name", type=str, default="failures.csv")
    parser.add_argument("--pocket-radius", type=float, default=10.0)
    parser.add_argument("--max-records", type=int, default=None, help="Optional cap for smoke preprocessing.")
    parser.add_argument(
        "--pair-mode",
        choices=["reference_to_targets", "all_ordered"],
        default="reference_to_targets",
        help="Use original reference source -> each target, or all lower-activity -> higher-activity pairs per series.",
    )
    parser.add_argument("--min-activity-delta", type=float, default=0.0)
    parser.add_argument("--max-pairs-per-series", type=int, default=None)
    parser.add_argument("--augment-copies", type=int, default=1)
    parser.add_argument("--augment-seed", type=int, default=2024)
    parser.add_argument("--global-random-rotate", action="store_true")
    parser.add_argument("--global-translate-sigma", type=float, default=0.0)
    parser.add_argument("--ligand-noise-sigma", type=float, default=0.0)
    parser.add_argument("--source-noise-sigma", type=float, default=0.0)
    parser.add_argument("--protein-noise-sigma", type=float, default=0.0)
    parser.add_argument("--negative-noise-sigma", type=float, default=0.03)
    parser.add_argument(
        "--hard-negative-mode",
        choices=["none", "within_series"],
        default="none",
        help="Attach one same-pocket low-activity hard negative per positive pair when available.",
    )
    parser.add_argument("--hard-negative-min-target-similarity", type=float, default=0.2)
    parser.add_argument("--hard-negative-score-slack", type=float, default=0.0)
    return parser.parse_args()


def discover_series(root: Path) -> list[H2LSeries]:
    series: list[H2LSeries] = []
    for target_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        target_id = target_dir.name
        protein_path = target_dir / f"{target_id}_pocket10.pdb"
        h2l_dir = target_dir / "reference_active_molecules" / "Hit2Lead"
        if not protein_path.exists() or not h2l_dir.exists():
            continue
        for source_path in sorted(h2l_dir.glob(f"{target_id}_*_reference_ligand_pose_with_h.sdf")):
            prefix = source_path.name.removesuffix("_reference_ligand_pose_with_h.sdf")
            series_id = prefix.removeprefix(f"{target_id}_")
            target_path = h2l_dir / f"{prefix}_with_common_scaffold.sdf"
            if target_path.exists():
                series.append(
                    H2LSeries(
                        target_id=target_id,
                        series_id=series_id,
                        protein_path=protein_path,
                        source_ligand_path=source_path,
                        target_ligand_path=target_path,
                    )
                )
    return series


def load_sdf_mols(path: Path) -> list[Chem.Mol]:
    return [mol for mol in Chem.SDMolSupplier(str(path), sanitize=True, removeHs=False) if mol is not None]


def prop(mol: Chem.Mol, name: str) -> str:
    return mol.GetProp(name) if mol.HasProp(name) else ""


def affinity_value(mol: Chem.Mol) -> float | None:
    text = prop(mol, "Affinity")
    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    if not match:
        return None
    try:
        value = float(match.group(0))
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def activity_score(mol: Chem.Mol) -> float | None:
    value = affinity_value(mol)
    if value is None:
        return None
    affinity_type = prop(mol, "Affinity Type").lower()
    unit = prop(mol, "Affinity Unit").lower()
    higher_is_better = affinity_type.startswith("p") or unit.startswith("p")
    return value if higher_is_better else -value


_FP_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def mol_fingerprint(mol: Chem.Mol):
    return _FP_GENERATOR.GetFingerprint(Chem.RemoveHs(mol))


def mol_similarity(lhs: Chem.Mol, rhs: Chem.Mol) -> float:
    try:
        return float(DataStructs.TanimotoSimilarity(mol_fingerprint(lhs), mol_fingerprint(rhs)))
    except Exception:
        return 0.0


def random_rotation(generator: torch.Generator) -> torch.Tensor:
    q = torch.randn(4, generator=generator)
    q = q / q.norm().clamp_min(1e-8)
    w, x, y, z = q
    return torch.tensor(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=torch.float32,
    )


def augment_record(record: dict, args: argparse.Namespace, aug_idx: int, seed_offset: int) -> dict:
    if aug_idx == 0:
        return record
    out = {
        key: value.clone() if torch.is_tensor(value) else value
        for key, value in record.items()
    }
    generator = torch.Generator().manual_seed(args.augment_seed + seed_offset * 1009 + aug_idx)
    pos_keys = [key for key in ["protein_pos", "source_pos", "ligand_pos", "negative_ligand_pos"] if key in out]
    if args.global_random_rotate:
        rot = random_rotation(generator)
        for key in pos_keys:
            out[key] = out[key] @ rot.T
    if args.global_translate_sigma > 0:
        delta = torch.randn(1, 3, generator=generator) * args.global_translate_sigma
        for key in pos_keys:
            out[key] = out[key] + delta
    for key, sigma in [
        ("protein_pos", args.protein_noise_sigma),
        ("source_pos", args.source_noise_sigma),
        ("ligand_pos", args.ligand_noise_sigma),
        ("negative_ligand_pos", args.negative_noise_sigma),
    ]:
        if key in out and sigma > 0:
            out[key] = out[key] + torch.randn(out[key].shape, generator=generator) * sigma
    return out


def build_pairs(
    item: H2LSeries,
    source_mol: Chem.Mol,
    target_mols: list[Chem.Mol],
    args: argparse.Namespace,
) -> list[PairDef]:
    pool = build_pool(item, source_mol, target_mols)
    if args.pair_mode == "reference_to_targets":
        pairs = [
            PairDef(
                source_label="ref",
                source_index=0,
                source_mol=source_mol,
                source_path=item.source_ligand_path,
                target_index=target_index,
                target_mol=target_mol,
                source_score=activity_score(source_mol),
                target_score=activity_score(target_mol),
            )
            for target_index, target_mol in enumerate(target_mols)
        ]
        return attach_hard_negatives(pairs, pool, args)

    scored = [entry for entry in pool if entry.score is not None]
    if len(scored) < 2:
        pairs = [
            PairDef(
                source_label="ref",
                source_index=0,
                source_mol=source_mol,
                source_path=item.source_ligand_path,
                target_index=target_index,
                target_mol=target_mol,
                source_score=activity_score(source_mol),
                target_score=activity_score(target_mol),
            )
            for target_index, target_mol in enumerate(target_mols)
        ]
        return attach_hard_negatives(pairs, pool, args)

    pairs: list[PairDef] = []
    for low in scored:
        assert low.score is not None
        for high in scored:
            assert high.score is not None
            if high.target_index is None or low.mol is high.mol:
                continue
            if high.score > low.score + args.min_activity_delta:
                pairs.append(
                    PairDef(
                        source_label=low.label,
                        source_index=low.index,
                        source_mol=low.mol,
                        source_path=low.path,
                        target_index=high.target_index,
                        target_mol=high.mol,
                        source_score=low.score,
                        target_score=high.score,
                    )
                )
            if args.max_pairs_per_series is not None and len(pairs) >= args.max_pairs_per_series:
                return attach_hard_negatives(pairs, pool, args)
    return attach_hard_negatives(pairs, pool, args)


def build_pool(item: H2LSeries, source_mol: Chem.Mol, target_mols: list[Chem.Mol]) -> list[SeriesMol]:
    pool = [
        SeriesMol("ref", 0, source_mol, item.source_ligand_path, None, activity_score(source_mol))
    ]
    pool.extend(
        SeriesMol(
            f"t{target_index:04d}",
            target_index,
            target_mol,
            item.target_ligand_path,
            target_index,
            activity_score(target_mol),
        )
        for target_index, target_mol in enumerate(target_mols)
    )
    return pool


def attach_hard_negatives(pairs: list[PairDef], pool: list[SeriesMol], args: argparse.Namespace) -> list[PairDef]:
    if args.hard_negative_mode == "none":
        return pairs
    return [attach_hard_negative(pair, pool, args) for pair in pairs]


def attach_hard_negative(pair: PairDef, pool: list[SeriesMol], args: argparse.Namespace) -> PairDef:
    candidates: list[tuple[float, float, SeriesMol]] = []
    for candidate in pool:
        if candidate.mol is pair.target_mol:
            continue
        if candidate.score is not None and pair.target_score is not None and candidate.score >= pair.target_score:
            continue
        if (
            candidate.score is not None
            and pair.source_score is not None
            and candidate.score > pair.source_score + args.hard_negative_score_slack
        ):
            continue
        similarity = mol_similarity(candidate.mol, pair.target_mol)
        if similarity < args.hard_negative_min_target_similarity:
            continue
        score_key = candidate.score if candidate.score is not None else float("-inf")
        candidates.append((similarity, score_key, candidate))
    if not candidates:
        return pair
    similarity, _score_key, chosen = max(candidates, key=lambda item: (item[0], -item[1]))
    return PairDef(
        source_label=pair.source_label,
        source_index=pair.source_index,
        source_mol=pair.source_mol,
        source_path=pair.source_path,
        target_index=pair.target_index,
        target_mol=pair.target_mol,
        source_score=pair.source_score,
        target_score=pair.target_score,
        negative_label=chosen.label,
        negative_index=chosen.index,
        negative_mol=chosen.mol,
        negative_path=chosen.path,
        negative_score=chosen.score,
        negative_similarity=similarity,
    )


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    series_list = discover_series(args.root)
    if not series_list:
        raise ValueError(f"no MolGenBench H2L series found under {args.root}")

    manifest_paths: list[Path] = []
    pair_rows: list[dict[str, str]] = []
    failure_rows: list[dict[str, str]] = []
    max_records = args.max_records

    for item in tqdm(series_list, desc="molgenbench-h2l"):
        if max_records is not None and len(manifest_paths) >= max_records:
            break
        try:
            protein_full = parse_pdb_atoms(item.protein_path)
            source_mol = load_first_mol(item.source_ligand_path)
            target_mols = load_sdf_mols(item.target_ligand_path)
            pair_defs = build_pairs(item, source_mol, target_mols, args)
        except Exception as exc:
            failure_rows.append(
                {
                    "target_id": item.target_id,
                    "series_id": item.series_id,
                    "source_index": "",
                    "target_index": "",
                    "error": repr(exc),
                }
            )
            continue

        for pair_index, pair in enumerate(pair_defs):
            if max_records is not None and len(manifest_paths) >= max_records:
                break
            try:
                source = mol_to_record_tensors(pair.source_mol)
                target = mol_to_record_tensors(pair.target_mol)
                negative = mol_to_record_tensors(pair.negative_mol) if pair.negative_mol is not None else None
                protein = crop_protein_to_ligand(protein_full, source["pos"], args.pocket_radius)
                base_record_id = (
                    f"{item.target_id}_{item.series_id}_{pair.target_index:04d}"
                    if args.pair_mode == "reference_to_targets" and args.augment_copies == 1
                    else f"{item.target_id}_{item.series_id}_{pair.source_label}_to_t{pair.target_index:04d}"
                )
                for aug_idx in range(max(1, args.augment_copies)):
                    if max_records is not None and len(manifest_paths) >= max_records:
                        break
                    record_id = base_record_id if args.augment_copies == 1 else f"{base_record_id}_aug{aug_idx:02d}"
                    out_path = args.outdir / item.target_id / f"{record_id}.pt"
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    record = {
                        "record_id": record_id,
                        "target_id": item.target_id,
                        "series_id": item.series_id,
                        "pair_index": pair_index,
                        "source_index": pair.source_index,
                        "target_index": pair.target_index,
                        "augmentation_index": aug_idx,
                        "protein_atom_type": protein["atom_type"],
                        "protein_pos": protein["pos"],
                        "source_atom_type": source["atom_type"],
                        "source_pos": source["pos"],
                        "ligand_atom_type": target["atom_type"],
                        "ligand_pos": target["pos"],
                        "ligand_bond_edge_index": target["bond_edge_index"],
                        "ligand_bond_type": target["bond_type"],
                        "protein_path": str(item.protein_path.resolve()),
                        "source_ligand_path": str(pair.source_path.resolve()),
                        "target_ligand_path": str(item.target_ligand_path.resolve()),
                        "source_affinity": prop(pair.source_mol, "Affinity"),
                        "source_affinity_type": prop(pair.source_mol, "Affinity Type"),
                        "source_affinity_unit": prop(pair.source_mol, "Affinity Unit"),
                        "target_affinity": prop(pair.target_mol, "Affinity"),
                        "target_affinity_type": prop(pair.target_mol, "Affinity Type"),
                        "target_affinity_unit": prop(pair.target_mol, "Affinity Unit"),
                    }
                    if negative is not None:
                        record.update(
                            {
                                "negative_label": pair.negative_label,
                                "negative_index": pair.negative_index,
                                "negative_ligand_atom_type": negative["atom_type"],
                                "negative_ligand_pos": negative["pos"],
                                "negative_ligand_bond_edge_index": negative["bond_edge_index"],
                                "negative_ligand_bond_type": negative["bond_type"],
                                "negative_ligand_path": str(pair.negative_path.resolve()) if pair.negative_path else "",
                                "negative_affinity": prop(pair.negative_mol, "Affinity") if pair.negative_mol else "",
                                "negative_affinity_type": prop(pair.negative_mol, "Affinity Type") if pair.negative_mol else "",
                                "negative_affinity_unit": prop(pair.negative_mol, "Affinity Unit") if pair.negative_mol else "",
                                "negative_similarity_to_target": pair.negative_similarity,
                            }
                        )
                    record = augment_record(record, args, aug_idx, len(manifest_paths))
                    torch.save(record, out_path)
                    manifest_paths.append(out_path.resolve())
                    pair_rows.append(
                        {
                            "record_id": record_id,
                            "target_id": item.target_id,
                            "series_id": item.series_id,
                            "pair_index": str(pair_index),
                            "source_index": str(pair.source_index),
                            "target_index": str(pair.target_index),
                            "augmentation_index": str(aug_idx),
                            "protein_path": str(item.protein_path.resolve()),
                            "source_ligand_path": str(pair.source_path.resolve()),
                            "target_ligand_path": str(item.target_ligand_path.resolve()),
                            "source_affinity": prop(pair.source_mol, "Affinity"),
                            "source_affinity_type": prop(pair.source_mol, "Affinity Type"),
                            "source_affinity_unit": prop(pair.source_mol, "Affinity Unit"),
                            "target_affinity": prop(pair.target_mol, "Affinity"),
                            "target_affinity_type": prop(pair.target_mol, "Affinity Type"),
                            "target_affinity_unit": prop(pair.target_mol, "Affinity Unit"),
                            "negative_label": pair.negative_label or "",
                            "negative_index": "" if pair.negative_index is None else str(pair.negative_index),
                            "negative_ligand_path": str(pair.negative_path.resolve()) if pair.negative_path else "",
                            "negative_affinity": prop(pair.negative_mol, "Affinity") if pair.negative_mol else "",
                            "negative_affinity_type": prop(pair.negative_mol, "Affinity Type") if pair.negative_mol else "",
                            "negative_affinity_unit": prop(pair.negative_mol, "Affinity Unit") if pair.negative_mol else "",
                            "negative_similarity_to_target": "" if pair.negative_similarity is None else f"{pair.negative_similarity:.6f}",
                        }
                    )
            except Exception as exc:
                failure_rows.append(
                    {
                        "target_id": item.target_id,
                        "series_id": item.series_id,
                        "source_index": str(pair.source_index),
                        "target_index": str(pair.target_index),
                        "error": repr(exc),
                    }
                )

    manifest = args.outdir / args.manifest_name
    manifest.write_text("\n".join(str(path) for path in manifest_paths) + "\n")

    pairs_path = args.outdir / args.pairs_name
    with pairs_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "record_id",
                "target_id",
                "series_id",
                "pair_index",
                "source_index",
                "target_index",
                "augmentation_index",
                "protein_path",
                "source_ligand_path",
                "target_ligand_path",
                "source_affinity",
                "source_affinity_type",
                "source_affinity_unit",
                "target_affinity",
                "target_affinity_type",
                "target_affinity_unit",
                "negative_label",
                "negative_index",
                "negative_ligand_path",
                "negative_affinity",
                "negative_affinity_type",
                "negative_affinity_unit",
                "negative_similarity_to_target",
            ],
        )
        writer.writeheader()
        writer.writerows(pair_rows)

    failures_path = args.outdir / args.failures_name
    with failures_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["target_id", "series_id", "source_index", "target_index", "error"])
        writer.writeheader()
        writer.writerows(failure_rows)

    print(f"discovered_series: {len(series_list)}")
    print(f"wrote_records: {len(manifest_paths)}")
    print(f"failures: {len(failure_rows)}")
    print(f"manifest: {manifest}")
    print(f"pairs: {pairs_path}")
    print(f"failures_csv: {failures_path}")


if __name__ == "__main__":
    main()
