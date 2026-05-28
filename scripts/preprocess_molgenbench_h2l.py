#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from rdkit import Chem
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
            source = mol_to_record_tensors(source_mol)
            protein = crop_protein_to_ligand(protein_full, source["pos"], args.pocket_radius)
            target_mols = load_sdf_mols(item.target_ligand_path)
        except Exception as exc:
            failure_rows.append(
                {
                    "target_id": item.target_id,
                    "series_id": item.series_id,
                    "target_index": "",
                    "error": repr(exc),
                }
            )
            continue

        for target_index, target_mol in enumerate(target_mols):
            if max_records is not None and len(manifest_paths) >= max_records:
                break
            try:
                target = mol_to_record_tensors(target_mol)
                record_id = f"{item.target_id}_{item.series_id}_{target_index:04d}"
                out_path = args.outdir / item.target_id / f"{record_id}.pt"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                record = {
                    "record_id": record_id,
                    "target_id": item.target_id,
                    "series_id": item.series_id,
                    "target_index": target_index,
                    "protein_atom_type": protein["atom_type"],
                    "protein_pos": protein["pos"],
                    "source_atom_type": source["atom_type"],
                    "source_pos": source["pos"],
                    "ligand_atom_type": target["atom_type"],
                    "ligand_pos": target["pos"],
                    "ligand_bond_edge_index": target["bond_edge_index"],
                    "ligand_bond_type": target["bond_type"],
                    "protein_path": str(item.protein_path.resolve()),
                    "source_ligand_path": str(item.source_ligand_path.resolve()),
                    "target_ligand_path": str(item.target_ligand_path.resolve()),
                    "target_affinity": prop(target_mol, "Affinity"),
                    "target_affinity_type": prop(target_mol, "Affinity Type"),
                    "target_affinity_unit": prop(target_mol, "Affinity Unit"),
                }
                torch.save(record, out_path)
                manifest_paths.append(out_path.resolve())
                pair_rows.append(
                    {
                        "record_id": record_id,
                        "target_id": item.target_id,
                        "series_id": item.series_id,
                        "target_index": str(target_index),
                        "protein_path": str(item.protein_path.resolve()),
                        "source_ligand_path": str(item.source_ligand_path.resolve()),
                        "target_ligand_path": str(item.target_ligand_path.resolve()),
                        "target_affinity": prop(target_mol, "Affinity"),
                        "target_affinity_type": prop(target_mol, "Affinity Type"),
                        "target_affinity_unit": prop(target_mol, "Affinity Unit"),
                    }
                )
            except Exception as exc:
                failure_rows.append(
                    {
                        "target_id": item.target_id,
                        "series_id": item.series_id,
                        "target_index": str(target_index),
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
                "target_index",
                "protein_path",
                "source_ligand_path",
                "target_ligand_path",
                "target_affinity",
                "target_affinity_type",
                "target_affinity_unit",
            ],
        )
        writer.writeheader()
        writer.writerows(pair_rows)

    failures_path = args.outdir / args.failures_name
    with failures_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["target_id", "series_id", "target_index", "error"])
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
