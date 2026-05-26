import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pco_backbone import (
    BackboneConfig,
    ComplexDenoiserBackbone,
    DiffusionConfig,
    ProteinConditionedDiffusion,
)


def test_forward_shapes():
    torch.manual_seed(7)
    config = BackboneConfig(
        hidden_dim=64,
        time_dim=32,
        rbf_dim=16,
        num_blocks=2,
        ligand_knn=4,
        protein_knn=6,
        cross_knn=8,
        source_knn=4,
    )
    model = ComplexDenoiserBackbone(config)

    protein_atom_type = torch.randint(0, config.num_protein_atom_types, (30,))
    protein_pos = torch.randn(30, 3)
    protein_batch = torch.tensor([0] * 12 + [1] * 18)

    ligand_atom_type = torch.randint(0, config.num_ligand_atom_types, (11,))
    ligand_pos = torch.randn(11, 3)
    ligand_batch = torch.tensor([0] * 5 + [1] * 6)
    time = torch.tensor([0.2, 0.8])
    source_atom_type = torch.randint(0, config.num_ligand_atom_types, (10,))
    source_pos = torch.randn(10, 3)
    source_batch = torch.tensor([0] * 4 + [1] * 6)

    out = model(
        protein_atom_type=protein_atom_type,
        protein_pos=protein_pos,
        protein_batch=protein_batch,
        ligand_atom_type=ligand_atom_type,
        ligand_pos=ligand_pos,
        ligand_batch=ligand_batch,
        time=time,
        source_atom_type=source_atom_type,
        source_pos=source_pos,
        source_batch=source_batch,
    )

    assert out["pos_update"].shape == (11, 3)
    assert out["atom_logits"].shape == (11, config.num_ligand_atom_types)
    assert out["bond_edge_index"].shape[0] == 2
    assert out["bond_logits"].shape[0] == out["bond_edge_index"].shape[1]
    assert out["bond_logits"].shape[1] == config.num_bond_types
    assert out["complex_score"].shape == (2,)
    assert out["source_h"].shape == (10, config.hidden_dim)


def test_diffusion_training_loss():
    torch.manual_seed(11)
    config = BackboneConfig(
        hidden_dim=48,
        time_dim=32,
        rbf_dim=16,
        num_blocks=1,
        ligand_knn=4,
        protein_knn=6,
        cross_knn=8,
        source_knn=4,
    )
    backbone = ComplexDenoiserBackbone(config)
    diffusion = ProteinConditionedDiffusion(
        backbone,
        DiffusionConfig(num_timesteps=32, atom_mask_token=config.num_ligand_atom_types - 1),
    )

    batch = {
        "protein_atom_type": torch.randint(0, config.num_protein_atom_types, (18,)),
        "protein_pos": torch.randn(18, 3),
        "protein_batch": torch.tensor([0] * 8 + [1] * 10),
        "ligand_atom_type": torch.randint(0, config.num_ligand_atom_types - 1, (9,)),
        "ligand_pos": torch.randn(9, 3),
        "ligand_batch": torch.tensor([0] * 4 + [1] * 5),
        "ligand_bond_edge_index": torch.tensor([[0, 1, 4, 5], [1, 2, 5, 6]]),
        "ligand_bond_type": torch.tensor([1, 1, 2, 1]),
        "source_atom_type": torch.randint(0, config.num_ligand_atom_types - 1, (7,)),
        "source_pos": torch.randn(7, 3),
        "source_batch": torch.tensor([0] * 3 + [1] * 4),
    }

    out = diffusion.training_loss(batch)
    assert out["loss"].dim() == 0
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    assert any(p.grad is not None for p in diffusion.parameters())


if __name__ == "__main__":
    test_forward_shapes()
    test_diffusion_training_loss()
    print("backbone smoke test passed")
