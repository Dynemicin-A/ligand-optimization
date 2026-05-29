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

from pco_backbone.chem import crop_protein_to_ligand, load_first_mol, mol_to_record_tensors, parse_pdb_atoms  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess generic protein-ligand complexes for denoising pretraining.")
    parser.add_argument("--csv", type=Path, required=True, help="CSV with protein_path and ligand_path columns.")
    parser.add_argument("--outdir", type=Path, default=ROOT / "data/pretrain_complexes/train")
    parser.add_argument("--manifest-name", type=str, default="manifest.txt")
    parser.add_argument("--pairs-name", type=str, default="complexes.csv")
    parser.add_argument("--failures-name", type=str, default="failures.csv")
    parser.add_argument("--pocket-radius", type=float, default=10.0)
    parser.add_argument("--max-records", type=int, default=None)
    return parser.parse_args()


def resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    base = args.csv.parent

    rows = list(csv.DictReader(args.csv.open()))
    required = {"protein_path", "ligand_path"}
    if rows and not required.issubset(rows[0].keys()):
        raise ValueError(f"CSV must contain columns: {sorted(required)}")

    manifest_paths: list[Path] = []
    written_rows: list[dict[str, str]] = []
    failures: list[dict[str, str]] = []

    for i, row in enumerate(tqdm(rows, desc="pretrain-complexes")):
        if args.max_records is not None and len(manifest_paths) >= args.max_records:
            break
        protein_path = resolve(base, row["protein_path"])
        ligand_path = resolve(base, row["ligand_path"])
        record_id = row.get("record_id") or f"complex_{i:07d}"
        try:
            protein_full = parse_pdb_atoms(protein_path)
            ligand_mol = load_first_mol(ligand_path)
            ligand = mol_to_record_tensors(ligand_mol)
            protein = crop_protein_to_ligand(protein_full, ligand["pos"], args.pocket_radius)
            out_path = args.outdir / f"{record_id}.pt"
            record = {
                "record_id": record_id,
                "target_id": row.get("target_id", ""),
                "series_id": row.get("series_id", "pretrain"),
                "protein_atom_type": protein["atom_type"],
                "protein_pos": protein["pos"],
                # Self-conditioning pretraining: source is the known bound ligand.
                # H2L finetuning later replaces this with the low-activity hit.
                "source_atom_type": ligand["atom_type"],
                "source_pos": ligand["pos"],
                "ligand_atom_type": ligand["atom_type"],
                "ligand_pos": ligand["pos"],
                "ligand_bond_edge_index": ligand["bond_edge_index"],
                "ligand_bond_type": ligand["bond_type"],
                "protein_path": str(protein_path.resolve()),
                "source_ligand_path": str(ligand_path.resolve()),
                "target_ligand_path": str(ligand_path.resolve()),
            }
            torch.save(record, out_path)
            manifest_paths.append(out_path.resolve())
            written_rows.append(
                {
                    "record_id": record_id,
                    "protein_path": str(protein_path.resolve()),
                    "ligand_path": str(ligand_path.resolve()),
                }
            )
        except Exception as exc:
            failures.append(
                {
                    "record_id": record_id,
                    "protein_path": str(protein_path),
                    "ligand_path": str(ligand_path),
                    "error": repr(exc),
                }
            )

    manifest = args.outdir / args.manifest_name
    manifest.write_text("\n".join(str(path) for path in manifest_paths) + "\n")

    with (args.outdir / args.pairs_name).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["record_id", "protein_path", "ligand_path"])
        writer.writeheader()
        writer.writerows(written_rows)

    with (args.outdir / args.failures_name).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["record_id", "protein_path", "ligand_path", "error"])
        writer.writeheader()
        writer.writerows(failures)

    print(f"input_rows: {len(rows)}")
    print(f"wrote_records: {len(manifest_paths)}")
    print(f"failures: {len(failures)}")
    print(f"manifest: {manifest}")


if __name__ == "__main__":
    main()
