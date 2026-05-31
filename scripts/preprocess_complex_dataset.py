#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pco_backbone.chem import crop_protein_to_ligand, load_first_mol, mol_to_record_tensors, parse_pdb_atoms  # noqa: E402


PROTEIN_SUFFIXES = {".pdb", ".pdbqt", ".ent", ".pdb.gz", ".pdbqt.gz"}
LIGAND_SUFFIXES = {".sdf", ".mol", ".mol2", ".sdf.gz", ".mol2.gz"}
TEXT_LIGAND_SUFFIXES = {".smi", ".smiles", ".smi.gz"}
ALL_LIGAND_SUFFIXES = LIGAND_SUFFIXES | TEXT_LIGAND_SUFFIXES

PROTEIN_NAME_HINTS = ("pocket", "protein", "receptor", "rec", "prep")
LIGAND_NAME_HINTS = ("ligand", "lig", "pose", "crystal", "docked")

COLUMN_ALIASES = {
    "record_id": ("record_id", "id", "complex_id", "pdb_id", "pdbid", "name"),
    "target_id": ("target_id", "target", "uniprot", "protein_id", "pdb_id", "pdbid"),
    "series_id": ("series_id", "series", "split", "subset", "dataset"),
    "protein_path": ("protein_path", "pocket_path", "receptor_path", "protein", "pocket", "receptor", "rec_path"),
    "ligand_path": ("ligand_path", "target_ligand_path", "ligand", "ligand_file", "sdf_path", "mol_path"),
    "source_ligand_path": ("source_ligand_path", "source_path", "hit_ligand_path", "low_ligand_path"),
    "target_ligand_path": ("target_ligand_path", "target_path", "lead_ligand_path", "high_ligand_path"),
    "negative_ligand_path": ("negative_ligand_path", "negative_path", "decoy_ligand_path", "decoy_path"),
}


@dataclass(frozen=True)
class InputRow:
    record_id: str
    protein_path: Path
    target_ligand_path: Path
    source_ligand_path: Path | None = None
    negative_ligand_path: Path | None = None
    target_id: str = ""
    series_id: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess broad protein-ligand datasets into unified training .pt records."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--csv", type=Path, help="CSV/TSV with protein and ligand path columns.")
    input_group.add_argument("--jsonl", type=Path, help="JSONL with protein and ligand path fields.")
    input_group.add_argument("--root", type=Path, help="Dataset root to auto-discover protein-ligand complexes.")
    parser.add_argument(
        "--preset",
        choices=["auto", "pdbbind", "crossdocked", "moad", "flat"],
        default="auto",
        help="Directory discovery heuristics. auto is usually enough.",
    )
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--manifest-name", type=str, default="manifest.txt")
    parser.add_argument("--records-name", type=str, default="records.csv")
    parser.add_argument("--failures-name", type=str, default="failures.csv")
    parser.add_argument("--pocket-radius", type=float, default=10.0)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--ligands-per-protein", choices=["all", "best"], default="all")
    parser.add_argument("--max-ligands-per-dir", type=int, default=None)
    parser.add_argument(
        "--source-mode",
        choices=["self", "optional", "none"],
        default="self",
        help="For rows without source_ligand_path: self uses target as source; optional keeps explicit source only; none is invalid for current training.",
    )
    parser.add_argument("--sanitize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--limit-dirs", type=int, default=None, help="Directory discovery cap for quick smoke tests.")
    return parser.parse_args()


def has_suffix(path: Path, suffixes: set[str]) -> bool:
    name = path.name.lower()
    return any(name.endswith(suffix) for suffix in suffixes)


def normalize_id(raw: str) -> str:
    raw = raw.strip() or "record"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("_") or "record"


def resolve(base: Path, value: str | None) -> Path | None:
    if value is None or str(value).strip() == "":
        return None
    path = Path(str(value).strip())
    return path if path.is_absolute() else base / path


def read_table(path: Path) -> list[dict[str, str]]:
    sample = path.read_text(errors="replace")[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
    except csv.Error:
        dialect = csv.excel_tab if path.suffix.lower() in {".tsv", ".tab"} else csv.excel
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, dialect=dialect))


def read_jsonl(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)
        if not isinstance(item, dict):
            raise TypeError(f"JSONL row must be an object: {line[:80]}")
        rows.append({str(key): "" if value is None else str(value) for key, value in item.items()})
    return rows


def pick(row: dict[str, str], logical: str) -> str:
    lowered = {key.lower().strip(): value for key, value in row.items() if key is not None}
    for name in COLUMN_ALIASES[logical]:
        if name in lowered and str(lowered[name]).strip():
            return str(lowered[name]).strip()
    return ""


def rows_from_records(records: list[dict[str, str]], base: Path, source_mode: str) -> list[InputRow]:
    out: list[InputRow] = []
    for i, row in enumerate(records):
        protein_path = resolve(base, pick(row, "protein_path"))
        target_path = resolve(base, pick(row, "target_ligand_path") or pick(row, "ligand_path"))
        if protein_path is None or target_path is None:
            raise KeyError("CSV must provide protein_path/receptor_path and ligand_path/target_ligand_path columns")
        source_path = resolve(base, pick(row, "source_ligand_path"))
        if source_path is None and source_mode == "self":
            source_path = target_path
        negative_path = resolve(base, pick(row, "negative_ligand_path"))
        record_id = normalize_id(pick(row, "record_id") or f"complex_{i:08d}")
        out.append(
            InputRow(
                record_id=record_id,
                protein_path=protein_path,
                source_ligand_path=source_path,
                target_ligand_path=target_path,
                negative_ligand_path=negative_path,
                target_id=pick(row, "target_id"),
                series_id=pick(row, "series_id"),
            )
        )
    return out


def rows_from_csv(path: Path, source_mode: str) -> list[InputRow]:
    return rows_from_records(read_table(path), path.parent, source_mode)


def rows_from_jsonl(path: Path, source_mode: str) -> list[InputRow]:
    return rows_from_records(read_jsonl(path), path.parent, source_mode)


def score_protein(path: Path, preset: str) -> tuple[int, str]:
    name = path.name.lower()
    score = 0
    if preset == "pdbbind":
        score += 20 if "pocket" in name else 0
        score += 10 if "protein" in name else 0
    if preset == "crossdocked":
        score += 20 if "rec" in name or "receptor" in name else 0
    for i, hint in enumerate(PROTEIN_NAME_HINTS):
        if hint in name:
            score += 10 - i
    if "lig" in name:
        score -= 20
    return score, path.name


def score_ligand(path: Path, preset: str) -> tuple[int, str]:
    name = path.name.lower()
    score = 0
    if has_suffix(path, LIGAND_SUFFIXES):
        score += 5
    if preset == "pdbbind":
        score += 20 if "ligand" in name or "_lig" in name else 0
    for i, hint in enumerate(LIGAND_NAME_HINTS):
        if hint in name:
            score += 10 - i
    if "pocket" in name or "protein" in name or "receptor" in name:
        score -= 20
    return score, path.name


def iter_candidate_dirs(root: Path, limit_dirs: int | None) -> Iterable[Path]:
    yielded = 0
    for path in root.rglob("*"):
        if not path.is_dir():
            continue
        try:
            files = list(path.iterdir())
        except OSError:
            continue
        has_protein = any(item.is_file() and has_suffix(item, PROTEIN_SUFFIXES) for item in files)
        has_ligand = any(item.is_file() and has_suffix(item, ALL_LIGAND_SUFFIXES) for item in files)
        if has_protein and has_ligand:
            yield path
            yielded += 1
            if limit_dirs is not None and yielded >= limit_dirs:
                return


def rows_from_directory(
    root: Path,
    preset: str,
    source_mode: str,
    limit_dirs: int | None,
    ligands_per_protein: str,
    max_ligands_per_dir: int | None,
) -> list[InputRow]:
    rows: list[InputRow] = []
    for directory in iter_candidate_dirs(root, limit_dirs):
        files = [item for item in directory.iterdir() if item.is_file()]
        proteins = [item for item in files if has_suffix(item, PROTEIN_SUFFIXES)]
        ligands = [item for item in files if has_suffix(item, ALL_LIGAND_SUFFIXES)]
        if not proteins or not ligands:
            continue
        protein_path = sorted(proteins, key=lambda path: score_protein(path, preset), reverse=True)[0]
        ligand_paths = sorted(ligands, key=lambda path: score_ligand(path, preset), reverse=True)
        if ligands_per_protein == "best":
            ligand_paths = ligand_paths[:1]
        if max_ligands_per_dir is not None:
            ligand_paths = ligand_paths[:max_ligands_per_dir]
        rel = directory.relative_to(root)
        target_id = rel.parts[0] if rel.parts else directory.name
        dir_id = "_".join(rel.parts) if rel.parts else directory.name
        for ligand_path in ligand_paths:
            ligand_id = ligand_path.name
            record_id = normalize_id(f"{dir_id}_{ligand_id}" if len(ligand_paths) > 1 else dir_id)
            source_path = ligand_path if source_mode == "self" else None
            rows.append(
                InputRow(
                    record_id=record_id,
                    protein_path=protein_path,
                    source_ligand_path=source_path,
                    target_ligand_path=ligand_path,
                    target_id=target_id,
                    series_id=preset,
                )
            )
    return rows


def deduplicate_record_ids(rows: list[InputRow]) -> list[InputRow]:
    seen: dict[str, int] = {}
    out = []
    for row in rows:
        count = seen.get(row.record_id, 0)
        seen[row.record_id] = count + 1
        if count == 0:
            out.append(row)
        else:
            out.append(
                InputRow(
                    record_id=f"{row.record_id}_{count:04d}",
                    protein_path=row.protein_path,
                    source_ligand_path=row.source_ligand_path,
                    target_ligand_path=row.target_ligand_path,
                    negative_ligand_path=row.negative_ligand_path,
                    target_id=row.target_id,
                    series_id=row.series_id,
                )
            )
    return out


def build_record(row: InputRow, outdir: Path, pocket_radius: float, sanitize: bool, skip_existing: bool):
    out_path = outdir / f"{row.record_id}.pt"
    if skip_existing and out_path.exists():
        return "written", {"record_id": row.record_id, "path": str(out_path.resolve()), "skipped": "1"}, None

    if row.source_ligand_path is None:
        raise ValueError("source_ligand_path is required; use --source-mode self for pretraining datasets")

    protein_full = parse_pdb_atoms(row.protein_path)
    source_mol = load_first_mol(row.source_ligand_path, sanitize=sanitize)
    target_mol = load_first_mol(row.target_ligand_path, sanitize=sanitize)
    source = mol_to_record_tensors(source_mol)
    target = mol_to_record_tensors(target_mol)
    protein = crop_protein_to_ligand(protein_full, source["pos"], pocket_radius)
    record = {
        "record_id": row.record_id,
        "target_id": row.target_id,
        "series_id": row.series_id,
        "protein_atom_type": protein["atom_type"],
        "protein_pos": protein["pos"],
        "source_atom_type": source["atom_type"],
        "source_pos": source["pos"],
        "ligand_atom_type": target["atom_type"],
        "ligand_pos": target["pos"],
        "ligand_bond_edge_index": target["bond_edge_index"],
        "ligand_bond_type": target["bond_type"],
        "protein_path": str(row.protein_path.resolve()),
        "source_ligand_path": str(row.source_ligand_path.resolve()),
        "target_ligand_path": str(row.target_ligand_path.resolve()),
    }
    if row.negative_ligand_path is not None:
        negative_mol = load_first_mol(row.negative_ligand_path, sanitize=sanitize)
        negative = mol_to_record_tensors(negative_mol)
        record.update(
            {
                "negative_ligand_atom_type": negative["atom_type"],
                "negative_ligand_pos": negative["pos"],
                "negative_ligand_bond_edge_index": negative["bond_edge_index"],
                "negative_ligand_bond_type": negative["bond_type"],
                "negative_ligand_path": str(row.negative_ligand_path.resolve()),
            }
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(record, out_path)
    return (
        "written",
        {
            "record_id": row.record_id,
            "target_id": row.target_id,
            "series_id": row.series_id,
            "protein_path": str(row.protein_path.resolve()),
            "source_ligand_path": str(row.source_ligand_path.resolve()),
            "target_ligand_path": str(row.target_ligand_path.resolve()),
            "negative_ligand_path": str(row.negative_ligand_path.resolve()) if row.negative_ligand_path else "",
            "path": str(out_path.resolve()),
            "skipped": "0",
        },
        None,
    )


def worker(payload):
    row, outdir, pocket_radius, sanitize, skip_existing = payload
    try:
        return build_record(row, outdir, pocket_radius, sanitize, skip_existing)
    except Exception as exc:
        return (
            "failure",
            None,
            {
                "record_id": row.record_id,
                "protein_path": str(row.protein_path),
                "source_ligand_path": str(row.source_ligand_path or ""),
                "target_ligand_path": str(row.target_ligand_path),
                "negative_ligand_path": str(row.negative_ligand_path or ""),
                "error": f"{type(exc).__name__}: {exc}",
            },
        )


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    if args.csv is not None:
        rows = rows_from_csv(args.csv, args.source_mode)
    elif args.jsonl is not None:
        rows = rows_from_jsonl(args.jsonl, args.source_mode)
    else:
        rows = rows_from_directory(
            args.root,
            args.preset,
            args.source_mode,
            args.limit_dirs,
            args.ligands_per_protein,
            args.max_ligands_per_dir,
        )
    rows = deduplicate_record_ids(rows)
    if args.max_records is not None:
        rows = rows[: args.max_records]
    if not rows:
        raise ValueError("no input complexes discovered")

    manifest_paths: list[str] = []
    written_rows: list[dict[str, str]] = []
    failure_rows: list[dict[str, str]] = []
    payloads = [(row, args.outdir, args.pocket_radius, args.sanitize, args.skip_existing) for row in rows]

    if args.num_workers <= 1:
        iterator = (worker(payload) for payload in payloads)
        for status, written, failure in tqdm(iterator, total=len(payloads), desc="preprocess-complexes"):
            if status == "written" and written is not None:
                manifest_paths.append(written["path"])
                written_rows.append(written)
            elif failure is not None:
                failure_rows.append(failure)
    else:
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            results = executor.map(worker, payloads, chunksize=max(1, args.chunk_size))
            for status, written, failure in tqdm(results, total=len(payloads), desc="preprocess-complexes"):
                if status == "written" and written is not None:
                    manifest_paths.append(written["path"])
                    written_rows.append(written)
                elif failure is not None:
                    failure_rows.append(failure)

    manifest_paths.sort()
    written_rows.sort(key=lambda row: row["path"])
    failure_rows.sort(key=lambda row: row["record_id"])

    manifest = args.outdir / args.manifest_name
    manifest.write_text("\n".join(manifest_paths) + "\n")

    with (args.outdir / args.records_name).open("w", newline="") as handle:
        fieldnames = [
            "record_id",
            "target_id",
            "series_id",
            "protein_path",
            "source_ligand_path",
            "target_ligand_path",
            "negative_ligand_path",
            "path",
            "skipped",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(written_rows)

    with (args.outdir / args.failures_name).open("w", newline="") as handle:
        fieldnames = ["record_id", "protein_path", "source_ligand_path", "target_ligand_path", "negative_ligand_path", "error"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(failure_rows)

    print(f"input_rows: {len(rows)}")
    print(f"wrote_records: {len(written_rows)}")
    print(f"failures: {len(failure_rows)}")
    print(f"manifest: {manifest}")
    print(f"records_csv: {args.outdir / args.records_name}")
    print(f"failures_csv: {args.outdir / args.failures_name}")


if __name__ == "__main__":
    main()
