#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
from rdkit import Chem
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pco_backbone import BackboneConfig, ComplexDenoiserBackbone, DiffusionConfig, ProteinConditionedDiffusion  # noqa: E402
from pco_backbone.chem import tensors_to_mol, write_mol_sdf  # noqa: E402
from pco_backbone.data import _coerce_record, collate_complex_records, move_batch_to_device  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample H2L ligands from manifest records and write SDF files.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/processed_h2l/train/manifest.txt")
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-steps", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def build_model_from_checkpoint(path: Path, device: torch.device) -> ProteinConditionedDiffusion:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    backbone = ComplexDenoiserBackbone(BackboneConfig(**ckpt["backbone_config"]))
    model = ProteinConditionedDiffusion(backbone, DiffusionConfig(**ckpt["diffusion_config"]))
    model.load_state_dict(ckpt["model_state"])
    return model.to(device)


def read_manifest(path: Path) -> list[Path]:
    base = path.parent
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        item = Path(line)
        out.append(item if item.is_absolute() else base / item)
    return out


def first_mol_smiles(path: str | Path, index: int = 0) -> str:
    try:
        supplier = Chem.SDMolSupplier(str(path), sanitize=True, removeHs=False)
        mol = None
        for i, candidate in enumerate(supplier):
            if i == index:
                mol = candidate
                break
    except Exception:
        return ""
    if mol is None:
        return ""
    return Chem.MolToSmiles(Chem.RemoveHs(mol), canonical=True, isomericSmiles=True)


def mol_smiles_or_empty(mol: Chem.Mol | None) -> str:
    if mol is None:
        return ""
    try:
        copy = Chem.Mol(mol)
        Chem.SanitizeMol(copy)
        return Chem.MolToSmiles(Chem.RemoveHs(copy), canonical=True, isomericSmiles=True)
    except Exception:
        return ""


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    model = build_model_from_checkpoint(args.checkpoint, device)
    model.eval()

    paths = read_manifest(args.manifest)
    selected = paths[args.start_index : args.start_index + args.num_samples]
    if not selected:
        raise ValueError(f"no manifest records selected from {args.manifest}")

    sdf_dir = args.outdir / "sdf"
    tensor_dir = args.outdir / "tensors"
    sdf_dir.mkdir(parents=True, exist_ok=True)
    tensor_dir.mkdir(parents=True, exist_ok=True)

    generated_smiles: list[str] = []
    reference_smiles: list[str] = []
    source_smiles: list[str] = []
    generated_sdfs: list[str] = []
    failures: list[dict[str, str]] = []

    for i, record_path in enumerate(tqdm(selected, desc="sample-h2l")):
        raw = torch.load(record_path, map_location="cpu", weights_only=False)
        rec = _coerce_record(raw)
        rec["ligand_atom_type"] = torch.full_like(rec["ligand_atom_type"], model.atom_mask_token)
        batch = collate_complex_records([rec])
        batch = move_batch_to_device(batch, device)

        sample = model.sample(batch, num_steps=args.num_steps, temperature=args.temperature)
        sample_cpu = {key: value.detach().cpu() for key, value in sample.items()}
        record_id = str(raw.get("record_id", f"record_{args.start_index + i:06d}"))
        torch.save(sample_cpu, tensor_dir / f"{record_id}.pt")

        mol = tensors_to_mol(
            sample_cpu["ligand_atom_type"],
            sample_cpu["ligand_pos"],
            sample_cpu["bond_edge_index"],
            sample_cpu["bond_type"],
            sanitize=False,
        )
        sdf_path = sdf_dir / f"{record_id}.sdf"
        if mol is not None:
            mol.SetProp("_Name", record_id)
            try:
                write_mol_sdf(mol, sdf_path)
                generated_sdfs.append(str(sdf_path.resolve()))
            except Exception as exc:
                failures.append(
                    {
                        "record_id": record_id,
                        "record_path": str(record_path),
                        "stage": "write_sdf",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                if sdf_path.exists():
                    sdf_path.unlink()
        else:
            failures.append(
                {
                    "record_id": record_id,
                    "record_path": str(record_path),
                    "stage": "tensors_to_mol",
                    "error": "tensors_to_mol returned None",
                }
            )
        generated_smiles.append(mol_smiles_or_empty(mol))
        reference_smiles.append(first_mol_smiles(raw["target_ligand_path"], int(raw.get("target_index", 0))))
        source_smiles.append(first_mol_smiles(raw["source_ligand_path"], int(raw.get("source_index", 0))))

    (args.outdir / "generated_sdf_manifest.txt").write_text("\n".join(generated_sdfs) + "\n")
    (args.outdir / "generated.smi").write_text("\n".join(generated_smiles) + "\n")
    (args.outdir / "reference.smi").write_text("\n".join(reference_smiles) + "\n")
    (args.outdir / "source.smi").write_text("\n".join(source_smiles) + "\n")
    with (args.outdir / "failures.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["record_id", "record_path", "stage", "error"])
        writer.writeheader()
        writer.writerows(failures)
    print(f"sampled_records: {len(selected)}")
    print(f"written_sdf: {len(generated_sdfs)}")
    print(f"failed_records: {len(failures)}")
    print(f"sdf_dir: {sdf_dir}")


if __name__ == "__main__":
    main()
