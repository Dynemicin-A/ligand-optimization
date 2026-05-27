#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pco_backbone.chem import (  # noqa: E402
    crop_protein_to_ligand,
    load_first_mol,
    mol_to_record_tensors,
    parse_pdb_atoms,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess H2L records into training .pt files.")
    parser.add_argument("--csv", type=Path, required=True, help="CSV with protein_path, source_ligand_path, target_ligand_path.")
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--manifest-name", type=str, default="manifest.txt")
    parser.add_argument("--pocket-radius", type=float, default=10.0)
    return parser.parse_args()


def resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    base = args.csv.parent
    rows = list(csv.DictReader(args.csv.open()))
    required = {"protein_path", "source_ligand_path", "target_ligand_path"}
    missing = required - set(rows[0].keys() if rows else [])
    if missing:
        raise KeyError(f"CSV missing columns: {sorted(missing)}")

    manifest_paths = []
    for i, row in enumerate(tqdm(rows, desc="preprocess")):
        record_id = row.get("record_id") or row.get("id") or f"record_{i:06d}"
        protein_path = resolve(base, row["protein_path"])
        source_path = resolve(base, row["source_ligand_path"])
        target_path = resolve(base, row["target_ligand_path"])

        protein = parse_pdb_atoms(protein_path)
        source = mol_to_record_tensors(load_first_mol(source_path))
        target = mol_to_record_tensors(load_first_mol(target_path))
        protein = crop_protein_to_ligand(protein, source["pos"], args.pocket_radius)

        record = {
            "record_id": record_id,
            "protein_atom_type": protein["atom_type"],
            "protein_pos": protein["pos"],
            "source_atom_type": source["atom_type"],
            "source_pos": source["pos"],
            "ligand_atom_type": target["atom_type"],
            "ligand_pos": target["pos"],
            "ligand_bond_edge_index": target["bond_edge_index"],
            "ligand_bond_type": target["bond_type"],
            "protein_path": str(protein_path),
            "source_ligand_path": str(source_path),
            "target_ligand_path": str(target_path),
        }
        out_path = args.outdir / f"{record_id}.pt"
        torch.save(record, out_path)
        manifest_paths.append(out_path.resolve())

    manifest = args.outdir / args.manifest_name
    manifest.write_text("\n".join(str(path) for path in manifest_paths) + "\n")
    print(f"wrote {len(manifest_paths)} records")
    print(f"manifest: {manifest}")


if __name__ == "__main__":
    main()
