import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pco_backbone.data import PTRecordDataset, collate_complex_records
from pco_backbone.records import build_edit_labels, build_model_record, coerce_model_record


def component(n_atoms: int, offset: float = 0.0):
    if n_atoms <= 1:
        edge_index = torch.empty(2, 0, dtype=torch.long)
        bond_type = torch.empty(0, dtype=torch.long)
    else:
        edge_index = torch.stack([torch.arange(n_atoms - 1), torch.arange(1, n_atoms)], dim=0)
        bond_type = torch.ones(n_atoms - 1, dtype=torch.long)
    return {
        "atom_type": torch.arange(n_atoms, dtype=torch.long) % 4,
        "pos": torch.randn(n_atoms, 3) + offset,
        "bond_edge_index": edge_index,
        "bond_type": bond_type,
    }


def test_build_model_record_preserves_single_schema_and_metadata():
    torch.manual_seed(3)
    record = build_model_record(
        protein=component(4),
        source=component(3),
        ligand=component(5),
        negative=component(2),
        metadata={"record_id": "toy", "dataset_name": "unit"},
    )

    assert record["record_id"] == "toy"
    assert record["dataset_name"] == "unit"
    assert record["protein_atom_type"].shape == (4,)
    assert record["source_atom_type"].shape == (3,)
    assert record["ligand_atom_type"].shape == (5,)
    assert record["source_edge_index"].shape == (2, 2)
    assert record["ligand_bond_edge_index"].shape == (2, 4)
    assert record["negative_ligand_bond_edge_index"].shape == (2, 1)


def test_coerce_canonical_record_and_self_source_fallback():
    torch.manual_seed(5)
    canonical = {
        "protein_atom_type": torch.tensor([0, 1]),
        "protein_pos": torch.randn(2, 3),
        "ligand_atom_type": torch.tensor([2, 3, 4]),
        "ligand_pos": torch.randn(3, 3),
        "ligand_bond_edge_index": torch.tensor([[0, 1], [1, 2]]),
        "ligand_bond_type": torch.tensor([1, 1]),
    }

    record = coerce_model_record(canonical, source_mode="self")

    assert torch.equal(record["ligand_atom_type"], canonical["ligand_atom_type"])
    assert torch.equal(record["source_atom_type"], canonical["ligand_atom_type"])
    assert torch.equal(record["source_edge_index"], canonical["ligand_bond_edge_index"])
    assert torch.equal(record["ligand_bond_edge_index"], canonical["ligand_bond_edge_index"])


def test_collate_offsets_source_edges_and_pt_dataset_normalizes(tmp_path):
    torch.manual_seed(7)
    rec_a = build_model_record(
        protein=component(3),
        source=component(3),
        ligand=component(4),
        metadata={"record_id": "a"},
    )
    rec_b = build_model_record(
        protein=component(2, offset=2.0),
        source=component(2, offset=2.0),
        ligand=component(3, offset=2.0),
        metadata={"record_id": "b"},
    )
    path_a = tmp_path / "a.pt"
    path_b = tmp_path / "b.pt"
    torch.save(rec_a, path_a)
    torch.save(rec_b, path_b)
    manifest = tmp_path / "manifest.txt"
    manifest.write_text(f"{path_a.name}\n{path_b.name}\n")

    dataset = PTRecordDataset(manifest)
    batch = collate_complex_records([dataset[0], dataset[1]])

    assert batch["source_atom_type"].shape == (5,)
    assert batch["source_edge_index"].shape == (2, 3)
    assert batch["source_edge_index"][:, :2].max().item() < 3
    assert batch["source_edge_index"][:, 2:].min().item() >= 3
    assert "record_id" not in batch


def test_edit_labels_and_collate_offsets_match_indices():
    labels_a = build_edit_labels(
        ligand_atom_type=torch.tensor([0, 1, 2]),
        ligand_pos=torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [8.0, 0.0, 0.0]]),
        source_atom_type=torch.tensor([0, 3]),
        source_pos=torch.tensor([[0.1, 0.0, 0.0], [3.4, 0.0, 0.0]]),
    )
    assert labels_a["ligand_edit_label"].tolist() == [0, 2, 3]
    rec_a = build_model_record(protein=component(2), source=component(2), ligand=component(3))
    rec_a.update(labels_a)
    labels_b = build_edit_labels(
        ligand_atom_type=torch.tensor([0]),
        ligand_pos=torch.tensor([[0.0, 0.0, 0.0]]),
        source_atom_type=torch.tensor([1]),
        source_pos=torch.tensor([[0.1, 0.0, 0.0]]),
    )
    rec_b = build_model_record(protein=component(2), source=component(1), ligand=component(1))
    rec_b.update(labels_b)

    batch = collate_complex_records([rec_a, rec_b])

    assert batch["ligand_edit_label"].tolist() == [0, 2, 3, 1]
    assert batch["ligand_source_match_index"].tolist() == [0, 1, -1, 2]
    assert batch["source_delete_label"].shape == (3,)
