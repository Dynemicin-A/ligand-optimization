import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from pco_backbone import (
    BackboneConfig,
    ComplexDenoiserBackbone,
    DiffusionConfig,
    ProteinConditionedDiffusion,
)
from scripts.train_diffusion import load_model_weights


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
        DiffusionConfig(
            num_timesteps=32,
            atom_mask_token=config.num_ligand_atom_types - 1,
            hard_negative_loss_weight=0.1,
        ),
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
        "negative_ligand_atom_type": torch.randint(0, config.num_ligand_atom_types - 1, (8,)),
        "negative_ligand_pos": torch.randn(8, 3),
        "negative_ligand_batch": torch.tensor([0] * 3 + [1] * 5),
    }

    out = diffusion.training_loss(batch)
    assert out["loss"].dim() == 0
    assert torch.isfinite(out["loss"])
    assert out["hard_negative_loss"].dim() == 0
    assert torch.isfinite(out["hard_negative_loss"])
    out["loss"].backward()
    assert any(p.grad is not None for p in diffusion.parameters())


def test_forward_with_v3_residual_radial_modules():
    torch.manual_seed(17)
    config = BackboneConfig(
        hidden_dim=48,
        time_dim=32,
        rbf_dim=18,
        num_blocks=2,
        ligand_knn=4,
        protein_knn=6,
        cross_knn=8,
        source_knn=4,
        radial_basis="gaussian_cosine",
        radial_envelope="cosine",
        use_layer_norm=True,
        use_residual_ffn=True,
        ffn_multiplier=2,
        edge_gate=True,
        layer_scale_init=0.1,
    )
    model = ComplexDenoiserBackbone(config)

    out = model(
        protein_atom_type=torch.randint(0, config.num_protein_atom_types, (14,)),
        protein_pos=torch.randn(14, 3),
        protein_batch=torch.tensor([0] * 6 + [1] * 8),
        ligand_atom_type=torch.randint(0, config.num_ligand_atom_types, (9,)),
        ligand_pos=torch.randn(9, 3),
        ligand_batch=torch.tensor([0] * 4 + [1] * 5),
        time=torch.tensor([0.1, 0.7]),
        source_atom_type=torch.randint(0, config.num_ligand_atom_types, (8,)),
        source_pos=torch.randn(8, 3),
        source_batch=torch.tensor([0] * 3 + [1] * 5),
    )

    assert out["pos_update"].shape == (9, 3)
    assert out["atom_logits"].shape == (9, config.num_ligand_atom_types)
    assert torch.isfinite(out["pos_update"]).all()
    assert torch.isfinite(out["atom_logits"]).all()


def test_forward_with_v3_pair_trunk_outputs():
    torch.manual_seed(19)
    config = BackboneConfig(
        hidden_dim=48,
        time_dim=32,
        rbf_dim=16,
        num_blocks=1,
        ligand_knn=4,
        protein_knn=5,
        cross_knn=6,
        source_knn=4,
        radial_basis="gaussian_cosine",
        radial_envelope="cosine",
        use_layer_norm=True,
        use_residual_ffn=True,
        edge_gate=True,
        use_pair_trunk=True,
        pair_dim=24,
        pair_num_blocks=2,
        distogram_bins=12,
        use_copy_mutate_gate=True,
        copy_gate_classes=5,
    )
    model = ComplexDenoiserBackbone(config)

    out = model(
        protein_atom_type=torch.randint(0, config.num_protein_atom_types, (16,)),
        protein_pos=torch.randn(16, 3),
        protein_batch=torch.tensor([0] * 7 + [1] * 9),
        ligand_atom_type=torch.randint(0, config.num_ligand_atom_types, (10,)),
        ligand_pos=torch.randn(10, 3),
        ligand_batch=torch.tensor([0] * 4 + [1] * 6),
        time=torch.tensor([0.15, 0.65]),
        source_atom_type=torch.randint(0, config.num_ligand_atom_types, (9,)),
        source_pos=torch.randn(9, 3),
        source_batch=torch.tensor([0] * 4 + [1] * 5),
    )

    n_pl_edges = out["protein_ligand_edge_index"].shape[1]
    assert out["distogram_logits"].shape == (n_pl_edges, config.distogram_bins)
    assert out["contact_logits"].shape == (n_pl_edges,)
    assert out["copy_gate_logits"].shape == (10, config.copy_gate_classes)
    assert torch.isfinite(out["distogram_logits"]).all()
    assert torch.isfinite(out["contact_logits"]).all()
    assert torch.isfinite(out["copy_gate_logits"]).all()

    score_out = model(
        protein_atom_type=torch.randint(0, config.num_protein_atom_types, (16,)),
        protein_pos=torch.randn(16, 3),
        protein_batch=torch.tensor([0] * 7 + [1] * 9),
        ligand_atom_type=torch.randint(0, config.num_ligand_atom_types, (10,)),
        ligand_pos=torch.randn(10, 3),
        ligand_batch=torch.tensor([0] * 4 + [1] * 6),
        time=torch.tensor([0.15, 0.65]),
        source_atom_type=torch.randint(0, config.num_ligand_atom_types, (9,)),
        source_pos=torch.randn(9, 3),
        source_batch=torch.tensor([0] * 4 + [1] * 5),
        score_only=True,
    )
    assert set(score_out) == {"complex_score"}
    assert score_out["complex_score"].shape == (2,)


def test_v3_auxiliary_losses_and_source_negative_ranking():
    torch.manual_seed(29)
    config = BackboneConfig(
        hidden_dim=40,
        time_dim=32,
        rbf_dim=16,
        num_blocks=1,
        ligand_knn=4,
        protein_knn=5,
        cross_knn=6,
        source_knn=4,
        radial_basis="gaussian_cosine",
        radial_envelope="cosine",
        use_layer_norm=True,
        use_residual_ffn=True,
        edge_gate=True,
        use_pair_trunk=True,
        pair_dim=20,
        pair_num_blocks=1,
        distogram_bins=10,
        use_copy_mutate_gate=True,
        copy_gate_classes=5,
    )
    diffusion = ProteinConditionedDiffusion(
        ComplexDenoiserBackbone(config),
        DiffusionConfig(
            num_timesteps=24,
            atom_mask_token=config.num_ligand_atom_types - 1,
            hard_negative_loss_weight=0.1,
            hard_negative_score_only=True,
            hard_negative_grad_side="positive",
            hard_negative_detach_backbone=True,
            hard_negative_score_bound=0.5,
            distogram_loss_weight=0.1,
            contact_loss_weight=0.1,
            copy_gate_loss_weight=0.1,
        ),
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
        "source_atom_type": torch.randint(0, config.num_ligand_atom_types - 1, (8,)),
        "source_pos": torch.randn(8, 3),
        "source_batch": torch.tensor([0] * 3 + [1] * 5),
    }

    out = diffusion.training_loss(batch)
    assert torch.isfinite(out["loss"])
    assert torch.isfinite(out["distogram_loss"])
    assert torch.isfinite(out["contact_loss"])
    assert torch.isfinite(out["copy_gate_loss"])
    assert torch.isfinite(out["hard_negative_loss"])
    assert torch.isfinite(out["score_gap"])
    assert out["hard_negative_count"].item() == 2
    assert out["positive_score"].abs().item() <= 0.5
    assert out["negative_score"].abs().item() <= 0.5
    out["loss"].backward()
    assert any(p.grad is not None for p in diffusion.parameters())


def test_source_anchored_flow_matching_training_loss():
    torch.manual_seed(37)
    config = BackboneConfig(
        hidden_dim=40,
        time_dim=32,
        rbf_dim=16,
        num_blocks=1,
        ligand_knn=4,
        protein_knn=5,
        cross_knn=6,
        source_knn=4,
        radial_basis="gaussian_cosine",
        radial_envelope="cosine",
        use_layer_norm=True,
        use_residual_ffn=True,
        edge_gate=True,
        use_pair_trunk=True,
        pair_dim=20,
        pair_num_blocks=1,
        distogram_bins=10,
        use_copy_mutate_gate=True,
        copy_gate_classes=5,
    )
    diffusion = ProteinConditionedDiffusion(
        ComplexDenoiserBackbone(config),
        DiffusionConfig(
            position_objective="flow_matching",
            flow_matching_base="source",
            flow_matching_noise_scale=0.1,
            atom_mask_token=config.num_ligand_atom_types - 1,
            distogram_loss_weight=0.1,
            contact_loss_weight=0.1,
            copy_gate_loss_weight=0.1,
        ),
    )
    batch = {
        "protein_atom_type": torch.randint(0, config.num_protein_atom_types, (10,)),
        "protein_pos": torch.randn(10, 3),
        "protein_batch": torch.tensor([0] * 5 + [1] * 5),
        "ligand_atom_type": torch.randint(0, config.num_ligand_atom_types - 1, (7,)),
        "ligand_pos": torch.randn(7, 3),
        "ligand_batch": torch.tensor([0] * 3 + [1] * 4),
        "ligand_bond_edge_index": torch.tensor([[0, 1, 3, 4], [1, 2, 4, 5]]),
        "ligand_bond_type": torch.tensor([1, 1, 2, 1]),
        "source_atom_type": torch.randint(0, config.num_ligand_atom_types - 1, (6,)),
        "source_pos": torch.randn(6, 3),
        "source_batch": torch.tensor([0] * 3 + [1] * 3),
        "source_edge_index": torch.tensor([[0, 1, 3, 4], [1, 2, 4, 5]]),
        "ligand_edit_label": torch.tensor([0, 1, 2, 3, 0, 1, 2]),
        "ligand_source_match_index": torch.tensor([0, 1, 2, -1, 3, 4, -1]),
    }

    out = diffusion.training_loss(batch)
    assert torch.isfinite(out["loss"])
    assert torch.isfinite(out["pos_loss"])
    assert torch.isfinite(out["copy_gate_loss"])
    assert out["time_index"].dtype.is_floating_point
    out["loss"].backward()
    assert any(p.grad is not None for p in diffusion.parameters())


def test_partial_init_supports_added_v3_modules(tmp_path):
    torch.manual_seed(23)
    base_config = BackboneConfig(
        hidden_dim=32,
        time_dim=32,
        rbf_dim=16,
        num_blocks=1,
        ligand_knn=4,
        protein_knn=4,
        cross_knn=4,
        source_knn=4,
    )
    base = ProteinConditionedDiffusion(
        ComplexDenoiserBackbone(base_config),
        DiffusionConfig(num_timesteps=16, atom_mask_token=base_config.num_ligand_atom_types - 1),
    )
    ckpt = tmp_path / "base.pt"
    torch.save({"model_state": base.state_dict()}, ckpt)

    v3_config = BackboneConfig(
        hidden_dim=32,
        time_dim=32,
        rbf_dim=16,
        num_blocks=1,
        ligand_knn=4,
        protein_knn=4,
        cross_knn=4,
        source_knn=4,
        radial_basis="gaussian_cosine",
        radial_envelope="cosine",
        use_layer_norm=True,
        use_residual_ffn=True,
        edge_gate=True,
    )
    v3_model = ProteinConditionedDiffusion(
        ComplexDenoiserBackbone(v3_config),
        DiffusionConfig(num_timesteps=16, atom_mask_token=v3_config.num_ligand_atom_types - 1),
    )
    message = load_model_weights(ckpt, v3_model, weights="model")

    assert "model weights" in message
    assert "loaded=" in message


def test_task_specific_training_modes():
    torch.manual_seed(31)
    config = BackboneConfig(
        hidden_dim=32,
        time_dim=32,
        rbf_dim=12,
        num_blocks=1,
        ligand_knn=3,
        protein_knn=4,
        cross_knn=4,
        source_knn=3,
        radial_basis="gaussian_cosine",
        radial_envelope="cosine",
        use_layer_norm=True,
        use_residual_ffn=True,
        edge_gate=True,
        use_pair_trunk=True,
        pair_dim=16,
        pair_num_blocks=1,
        distogram_bins=8,
        use_copy_mutate_gate=True,
        copy_gate_classes=5,
    )
    batch = {
        "protein_atom_type": torch.randint(0, config.num_protein_atom_types, (10,)),
        "protein_pos": torch.randn(10, 3),
        "protein_batch": torch.tensor([0] * 5 + [1] * 5),
        "ligand_atom_type": torch.randint(0, config.num_ligand_atom_types - 1, (7,)),
        "ligand_pos": torch.randn(7, 3),
        "ligand_batch": torch.tensor([0] * 3 + [1] * 4),
        "ligand_bond_edge_index": torch.tensor([[0, 1, 3, 4], [1, 2, 4, 5]]),
        "ligand_bond_type": torch.tensor([1, 1, 2, 1]),
        "source_atom_type": torch.randint(0, config.num_ligand_atom_types - 1, (6,)),
        "source_pos": torch.randn(6, 3),
        "source_batch": torch.tensor([0] * 3 + [1] * 3),
        "negative_ligand_atom_type": torch.randint(0, config.num_ligand_atom_types - 1, (7,)),
        "negative_ligand_pos": torch.randn(7, 3),
        "negative_ligand_batch": torch.tensor([0] * 3 + [1] * 4),
        "ligand_edit_label": torch.tensor([0, 1, 2, 3, 0, 1, 2]),
    }

    for task, kwargs in {
        "edit_policy": {"copy_gate_loss_weight": 1.0},
        "interaction": {"distogram_loss_weight": 1.0, "contact_loss_weight": 1.0},
        "ranking": {"hard_negative_loss_weight": 1.0, "hard_negative_detach_backbone": True, "hard_negative_score_bound": 2.0},
    }.items():
        diffusion = ProteinConditionedDiffusion(
            ComplexDenoiserBackbone(config),
            DiffusionConfig(
                training_task=task,
                num_timesteps=16,
                atom_mask_token=config.num_ligand_atom_types - 1,
                **kwargs,
            ),
        )
        out = diffusion.training_loss(batch)
        assert torch.isfinite(out["loss"])
        if task == "edit_policy":
            assert torch.isfinite(out["copy_gate_accuracy"])
        if task == "interaction":
            assert torch.isfinite(out["distogram_accuracy"])
            assert torch.isfinite(out["contact_accuracy"])
        if task == "ranking":
            assert torch.isfinite(out["ranking_accuracy"])


if __name__ == "__main__":
    test_forward_shapes()
    test_diffusion_training_loss()
    print("backbone smoke test passed")
