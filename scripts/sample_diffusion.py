#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pco_backbone import BackboneConfig, ComplexDenoiserBackbone, DiffusionConfig, ProteinConditionedDiffusion  # noqa: E402
from pco_backbone.chem import tensors_to_mol, write_mol_sdf  # noqa: E402
from pco_backbone.data import SyntheticH2LConfig, SyntheticH2LDataset, move_batch_to_device  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample from a trained diffusion checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/sample.pt")
    parser.add_argument("--sdf-out", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num-steps", type=int, default=32)
    parser.add_argument("--num-ligand-atoms", type=int, default=12)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def build_model_from_checkpoint(path: Path, device: torch.device) -> ProteinConditionedDiffusion:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    backbone = ComplexDenoiserBackbone(BackboneConfig(**ckpt["backbone_config"]))
    model = ProteinConditionedDiffusion(backbone, DiffusionConfig(**ckpt["diffusion_config"]))
    model.load_state_dict(ckpt["model_state"])
    return model.to(device)


def make_condition_batch(model: ProteinConditionedDiffusion, num_ligand_atoms: int, seed: int) -> dict[str, torch.Tensor]:
    cfg = SyntheticH2LConfig(
        num_samples=1,
        min_protein_atoms=24,
        max_protein_atoms=24,
        min_ligand_atoms=max(2, num_ligand_atoms),
        max_ligand_atoms=max(2, num_ligand_atoms),
        num_ligand_atom_types=model.backbone.config.num_ligand_atom_types,
        num_protein_atom_types=model.backbone.config.num_protein_atom_types,
        num_bond_types=model.backbone.config.num_bond_types,
        seed=seed,
    )
    rec = SyntheticH2LDataset(cfg)[0]
    ligand_pos = torch.randn(num_ligand_atoms, 3, generator=torch.Generator().manual_seed(seed + 999))
    ligand_atom_type = torch.full(
        (num_ligand_atoms,),
        model.atom_mask_token,
        dtype=torch.long,
    )
    return {
        "protein_atom_type": rec["protein_atom_type"],
        "protein_pos": rec["protein_pos"],
        "protein_batch": torch.zeros(rec["protein_atom_type"].shape[0], dtype=torch.long),
        "source_atom_type": rec["source_atom_type"],
        "source_pos": rec["source_pos"],
        "source_batch": torch.zeros(rec["source_atom_type"].shape[0], dtype=torch.long),
        "ligand_atom_type": ligand_atom_type,
        "ligand_pos": ligand_pos,
        "ligand_batch": torch.zeros(num_ligand_atoms, dtype=torch.long),
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model = build_model_from_checkpoint(args.checkpoint, device)
    batch = make_condition_batch(model, args.num_ligand_atoms, args.seed)
    batch = move_batch_to_device(batch, device)
    sample = model.sample(batch, num_steps=args.num_steps)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({key: value.detach().cpu() for key, value in sample.items()}, args.out)
    print(f"saved sample: {args.out}")
    if args.sdf_out is not None:
        mol = tensors_to_mol(
            sample["ligand_atom_type"],
            sample["ligand_pos"],
            sample["bond_edge_index"],
            sample["bond_type"],
            sanitize=False,
        )
        if mol is None:
            raise RuntimeError("failed to convert sample tensors to RDKit molecule")
        write_mol_sdf(mol, args.sdf_out)
        print(f"saved sdf: {args.sdf_out}")


if __name__ == "__main__":
    main()
