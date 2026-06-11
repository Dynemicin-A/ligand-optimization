from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from .records import Record, coerce_model_record


class PTRecordDataset(Dataset[Record]):
    """Loads preprocessed records listed in a manifest file.

    Manifest format: one `.pt` path per line. Relative paths are resolved from
    the manifest directory. Each record should contain the same tensor keys as
    the canonical v3 model schema.
    """

    def __init__(self, manifest_path: str | Path):
        self.manifest_path = Path(manifest_path)
        base = self.manifest_path.parent
        paths = []
        for line in self.manifest_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            path = Path(line)
            paths.append(path if path.is_absolute() else base / path)
        if not paths:
            raise ValueError(f"empty manifest: {self.manifest_path}")
        self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Record:
        record = torch.load(self.paths[idx], map_location="cpu", weights_only=False)
        if not isinstance(record, dict):
            raise TypeError(f"record must be a dict: {self.paths[idx]}")
        return coerce_model_record(record)

def collate_complex_records(records: list[Record]) -> Record:
    records = [coerce_model_record(record) for record in records]
    batch: Record = {}
    protein_types, protein_pos, protein_batch = [], [], []
    source_types, source_pos, source_batch = [], [], []
    source_edges = []
    ligand_types, ligand_pos, ligand_batch = [], [], []
    bond_edges, bond_types = [], []
    neg_types, neg_pos, neg_batch = [], [], []
    neg_bond_edges, neg_bond_types = [], []

    ligand_offset = 0
    source_offset = 0
    negative_ligand_offset = 0
    for i, rec in enumerate(records):
        n_protein = rec["protein_atom_type"].shape[0]
        n_source = rec["source_atom_type"].shape[0]
        n_ligand = rec["ligand_atom_type"].shape[0]

        protein_types.append(rec["protein_atom_type"])
        protein_pos.append(rec["protein_pos"])
        protein_batch.append(torch.full((n_protein,), i, dtype=torch.long))

        source_types.append(rec["source_atom_type"])
        source_pos.append(rec["source_pos"])
        source_batch.append(torch.full((n_source,), i, dtype=torch.long))
        source_edge = rec.get("source_edge_index")
        if source_edge is not None and source_edge.numel() > 0:
            source_edges.append(source_edge + source_offset)
        source_offset += n_source

        ligand_types.append(rec["ligand_atom_type"])
        ligand_pos.append(rec["ligand_pos"])
        ligand_batch.append(torch.full((n_ligand,), i, dtype=torch.long))

        edge = rec.get("ligand_bond_edge_index")
        bond_type = rec.get("ligand_bond_type")
        if edge is not None and edge.numel() > 0:
            bond_edges.append(edge + ligand_offset)
            bond_types.append(bond_type)
        ligand_offset += n_ligand

        if "negative_ligand_atom_type" in rec:
            n_negative = rec["negative_ligand_atom_type"].shape[0]
            neg_types.append(rec["negative_ligand_atom_type"])
            neg_pos.append(rec["negative_ligand_pos"])
            neg_batch.append(torch.full((n_negative,), i, dtype=torch.long))
            neg_edge = rec.get("negative_ligand_bond_edge_index")
            neg_bond_type = rec.get("negative_ligand_bond_type")
            if neg_edge is not None and neg_edge.numel() > 0:
                neg_bond_edges.append(neg_edge + negative_ligand_offset)
                neg_bond_types.append(neg_bond_type)
            negative_ligand_offset += n_negative

    batch["protein_atom_type"] = torch.cat(protein_types, dim=0)
    batch["protein_pos"] = torch.cat(protein_pos, dim=0)
    batch["protein_batch"] = torch.cat(protein_batch, dim=0)
    batch["source_atom_type"] = torch.cat(source_types, dim=0)
    batch["source_pos"] = torch.cat(source_pos, dim=0)
    batch["source_batch"] = torch.cat(source_batch, dim=0)
    if source_edges:
        batch["source_edge_index"] = torch.cat(source_edges, dim=1)
    batch["ligand_atom_type"] = torch.cat(ligand_types, dim=0)
    batch["ligand_pos"] = torch.cat(ligand_pos, dim=0)
    batch["ligand_batch"] = torch.cat(ligand_batch, dim=0)

    if bond_edges:
        batch["ligand_bond_edge_index"] = torch.cat(bond_edges, dim=1)
        batch["ligand_bond_type"] = torch.cat(bond_types, dim=0)
    else:
        batch["ligand_bond_edge_index"] = torch.empty(2, 0, dtype=torch.long)
        batch["ligand_bond_type"] = torch.empty(0, dtype=torch.long)
    if neg_types:
        batch["negative_ligand_atom_type"] = torch.cat(neg_types, dim=0)
        batch["negative_ligand_pos"] = torch.cat(neg_pos, dim=0)
        batch["negative_ligand_batch"] = torch.cat(neg_batch, dim=0)
        if neg_bond_edges:
            batch["negative_ligand_bond_edge_index"] = torch.cat(neg_bond_edges, dim=1)
            batch["negative_ligand_bond_type"] = torch.cat(neg_bond_types, dim=0)
        else:
            batch["negative_ligand_bond_edge_index"] = torch.empty(2, 0, dtype=torch.long)
            batch["negative_ligand_bond_type"] = torch.empty(0, dtype=torch.long)
    return batch


def move_batch_to_device(batch: Record, device: torch.device) -> Record:
    return {key: value.to(device) for key, value in batch.items()}
