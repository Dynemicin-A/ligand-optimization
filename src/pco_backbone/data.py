from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


Record = dict[str, torch.Tensor]


@dataclass
class SyntheticH2LConfig:
    num_samples: int = 256
    min_protein_atoms: int = 24
    max_protein_atoms: int = 48
    min_ligand_atoms: int = 8
    max_ligand_atoms: int = 18
    num_ligand_atom_types: int = 16
    num_protein_atom_types: int = 32
    num_bond_types: int = 5
    seed: int = 2024


class SyntheticH2LDataset(Dataset[Record]):
    """Deterministic toy H2L-like records for smoke training.

    The source hit and target ligand share rough spatial support but are not
    fixed to the same scaffold. This is only for end-to-end program validation.
    """

    def __init__(self, config: SyntheticH2LConfig):
        self.config = config

    def __len__(self) -> int:
        return self.config.num_samples

    def _randint(self, g: torch.Generator, low: int, high_inclusive: int) -> int:
        return int(torch.randint(low, high_inclusive + 1, (1,), generator=g).item())

    def __getitem__(self, idx: int) -> Record:
        cfg = self.config
        g = torch.Generator().manual_seed(cfg.seed + idx)

        n_protein = self._randint(g, cfg.min_protein_atoms, cfg.max_protein_atoms)
        n_source = self._randint(g, cfg.min_ligand_atoms, cfg.max_ligand_atoms)
        n_ligand = self._randint(g, cfg.min_ligand_atoms, cfg.max_ligand_atoms)

        protein_atom_type = torch.randint(0, cfg.num_protein_atom_types, (n_protein,), generator=g)
        source_atom_type = torch.randint(0, cfg.num_ligand_atom_types - 1, (n_source,), generator=g)
        ligand_atom_type = torch.randint(0, cfg.num_ligand_atom_types - 1, (n_ligand,), generator=g)

        pocket_center = torch.randn(1, 3, generator=g) * 0.2
        protein_pos = torch.randn(n_protein, 3, generator=g) * 4.0 + pocket_center
        source_pos = torch.randn(n_source, 3, generator=g) * 1.5 + pocket_center

        source_anchor = source_pos[torch.randint(0, n_source, (n_ligand,), generator=g)]
        migration = torch.randn(n_ligand, 3, generator=g) * 0.65
        ligand_pos = source_anchor + migration

        ligand_bond_edge_index, ligand_bond_type = _make_chain_bonds(n_ligand, cfg.num_bond_types)

        return {
            "protein_atom_type": protein_atom_type.long(),
            "protein_pos": protein_pos.float(),
            "source_atom_type": source_atom_type.long(),
            "source_pos": source_pos.float(),
            "ligand_atom_type": ligand_atom_type.long(),
            "ligand_pos": ligand_pos.float(),
            "ligand_bond_edge_index": ligand_bond_edge_index.long(),
            "ligand_bond_type": ligand_bond_type.long(),
        }


class PTRecordDataset(Dataset[Record]):
    """Loads preprocessed records listed in a manifest file.

    Manifest format: one `.pt` path per line. Relative paths are resolved from
    the manifest directory. Each record should contain the same tensor keys as
    `SyntheticH2LDataset`.
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
        return _coerce_record(record)


def _make_chain_bonds(n_atoms: int, num_bond_types: int) -> tuple[torch.Tensor, torch.Tensor]:
    if n_atoms <= 1:
        return torch.empty(2, 0, dtype=torch.long), torch.empty(0, dtype=torch.long)
    src = torch.arange(0, n_atoms - 1, dtype=torch.long)
    dst = torch.arange(1, n_atoms, dtype=torch.long)
    edge_index = torch.stack([src, dst], dim=0)
    bond_type = torch.ones(n_atoms - 1, dtype=torch.long).clamp_max(num_bond_types - 1)
    return edge_index, bond_type


def _coerce_record(record: dict[str, Any]) -> Record:
    required = [
        "protein_atom_type",
        "protein_pos",
        "source_atom_type",
        "source_pos",
        "ligand_atom_type",
        "ligand_pos",
    ]
    missing = [key for key in required if key not in record]
    if missing:
        raise KeyError(f"record missing required keys: {missing}")

    out: Record = {}
    for key in required:
        out[key] = torch.as_tensor(record[key])
    out["protein_atom_type"] = out["protein_atom_type"].long()
    out["source_atom_type"] = out["source_atom_type"].long()
    out["ligand_atom_type"] = out["ligand_atom_type"].long()
    out["protein_pos"] = out["protein_pos"].float()
    out["source_pos"] = out["source_pos"].float()
    out["ligand_pos"] = out["ligand_pos"].float()

    if "ligand_bond_edge_index" in record:
        out["ligand_bond_edge_index"] = torch.as_tensor(record["ligand_bond_edge_index"]).long()
    else:
        out["ligand_bond_edge_index"] = torch.empty(2, 0, dtype=torch.long)
    if "ligand_bond_type" in record:
        out["ligand_bond_type"] = torch.as_tensor(record["ligand_bond_type"]).long()
    else:
        out["ligand_bond_type"] = torch.empty(0, dtype=torch.long)
    if "negative_ligand_atom_type" in record and "negative_ligand_pos" in record:
        out["negative_ligand_atom_type"] = torch.as_tensor(record["negative_ligand_atom_type"]).long()
        out["negative_ligand_pos"] = torch.as_tensor(record["negative_ligand_pos"]).float()
        if "negative_ligand_bond_edge_index" in record:
            out["negative_ligand_bond_edge_index"] = torch.as_tensor(record["negative_ligand_bond_edge_index"]).long()
        else:
            out["negative_ligand_bond_edge_index"] = torch.empty(2, 0, dtype=torch.long)
        if "negative_ligand_bond_type" in record:
            out["negative_ligand_bond_type"] = torch.as_tensor(record["negative_ligand_bond_type"]).long()
        else:
            out["negative_ligand_bond_type"] = torch.empty(0, dtype=torch.long)
    return out


def collate_complex_records(records: list[Record]) -> Record:
    batch: Record = {}
    protein_types, protein_pos, protein_batch = [], [], []
    source_types, source_pos, source_batch = [], [], []
    ligand_types, ligand_pos, ligand_batch = [], [], []
    bond_edges, bond_types = [], []
    neg_types, neg_pos, neg_batch = [], [], []
    neg_bond_edges, neg_bond_types = [], []

    ligand_offset = 0
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
            batch["negative_ligand_edge_index"] = torch.cat(neg_bond_edges, dim=1)
            batch["negative_ligand_bond_type"] = torch.cat(neg_bond_types, dim=0)
        else:
            batch["negative_ligand_edge_index"] = torch.empty(2, 0, dtype=torch.long)
            batch["negative_ligand_bond_type"] = torch.empty(0, dtype=torch.long)
    return batch


def move_batch_to_device(batch: Record, device: torch.device) -> Record:
    return {key: value.to(device) for key, value in batch.items()}
