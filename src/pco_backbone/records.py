from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch


Record = dict[str, torch.Tensor]
SourceMode = Literal["self", "optional", "none"]

REQUIRED_MODEL_KEYS = (
    "protein_atom_type",
    "protein_pos",
    "source_atom_type",
    "source_pos",
    "ligand_atom_type",
    "ligand_pos",
)
OPTIONAL_TENSOR_KEYS = (
    "source_edge_index",
    "source_bond_type",
    "ligand_bond_edge_index",
    "ligand_bond_type",
    "negative_ligand_atom_type",
    "negative_ligand_pos",
    "negative_ligand_bond_edge_index",
    "negative_ligand_bond_type",
)
METADATA_KEYS = (
    "record_id",
    "dataset_name",
    "target_id",
    "series_id",
    "split",
    "protein_path",
    "source_ligand_path",
    "target_ligand_path",
    "negative_ligand_path",
    "pair_index",
    "source_index",
    "target_index",
    "augmentation_index",
    "source_affinity",
    "source_affinity_type",
    "source_affinity_unit",
    "target_affinity",
    "target_affinity_type",
    "target_affinity_unit",
    "negative_label",
    "negative_index",
    "negative_affinity",
    "negative_affinity_type",
    "negative_affinity_unit",
    "negative_similarity_to_target",
)


@dataclass(frozen=True)
class ComplexRecordProcessor:
    """Normalizes dataset records into the canonical v3 model schema.

    Dataset-specific column/file naming belongs in preprocessing scripts. Saved
    `.pt` records and training batches use only these canonical tensor keys.
    """

    source_mode: SourceMode = "self"
    keep_metadata: bool = True

    def build(
        self,
        *,
        protein: dict[str, Any],
        ligand: dict[str, Any],
        source: dict[str, Any] | None = None,
        negative: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if source is None:
            if self.source_mode == "self":
                source = ligand
            elif self.source_mode == "none":
                source = empty_ligand_like(ligand)
            else:
                raise ValueError("source component is required when source_mode='optional'")

        record: dict[str, Any] = {}
        if self.keep_metadata and metadata:
            for key in METADATA_KEYS:
                value = metadata.get(key)
                if value is not None:
                    record[key] = str(value)

        record.update(prefix_component("protein", protein, include_bonds=False))
        record.update(prefix_component("source", source, include_bonds=True))
        record.update(prefix_component("ligand", ligand, include_bonds=True))
        if negative is not None:
            record.update(prefix_component("negative_ligand", negative, include_bonds=True))
        return coerce_model_record(record, source_mode=self.source_mode, keep_metadata=self.keep_metadata)

    def coerce(self, record: dict[str, Any]) -> dict[str, Any]:
        return coerce_model_record(record, source_mode=self.source_mode, keep_metadata=self.keep_metadata)


DEFAULT_RECORD_PROCESSOR = ComplexRecordProcessor()


def build_model_record(
    *,
    protein: dict[str, Any],
    ligand: dict[str, Any],
    source: dict[str, Any] | None = None,
    negative: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    source_mode: SourceMode = "self",
    keep_metadata: bool = True,
) -> dict[str, Any]:
    return ComplexRecordProcessor(source_mode=source_mode, keep_metadata=keep_metadata).build(
        protein=protein,
        source=source,
        ligand=ligand,
        negative=negative,
        metadata=metadata,
    )


def coerce_model_record(
    record: dict[str, Any],
    *,
    source_mode: SourceMode = "self",
    keep_metadata: bool = False,
) -> dict[str, Any]:
    normalized = dict(record)
    if "source_atom_type" not in normalized or "source_pos" not in normalized:
        if source_mode == "self":
            normalized["source_atom_type"] = normalized.get("ligand_atom_type")
            normalized["source_pos"] = normalized.get("ligand_pos")
            normalized["source_edge_index"] = normalized.get("ligand_bond_edge_index")
            normalized["source_bond_type"] = normalized.get("ligand_bond_type")
        elif source_mode == "none":
            normalized["source_atom_type"] = torch.empty(0, dtype=torch.long)
            normalized["source_pos"] = torch.empty(0, 3, dtype=torch.float32)
        else:
            raise KeyError("record missing source_atom_type/source_pos")

    missing = [key for key in REQUIRED_MODEL_KEYS if key not in normalized or normalized[key] is None]
    if missing:
        raise KeyError(f"record missing required model keys: {missing}")

    out: dict[str, Any] = {}
    if keep_metadata:
        for key in METADATA_KEYS:
            if key in normalized:
                out[key] = normalized[key]

    for key in ("protein_atom_type", "source_atom_type", "ligand_atom_type"):
        out[key] = as_atom_type(normalized[key], key)
    for key in ("protein_pos", "source_pos", "ligand_pos"):
        out[key] = as_pos(normalized[key], key)

    out["source_edge_index"] = as_edge_index(normalized.get("source_edge_index"), "source_edge_index")
    out["source_bond_type"] = as_bond_type(normalized.get("source_bond_type"), "source_bond_type")
    out["ligand_bond_edge_index"] = as_edge_index(
        normalized.get("ligand_bond_edge_index"),
        "ligand_bond_edge_index",
    )
    out["ligand_bond_type"] = as_bond_type(normalized.get("ligand_bond_type"), "ligand_bond_type")

    has_negative = "negative_ligand_atom_type" in normalized or "negative_ligand_pos" in normalized
    if has_negative:
        if "negative_ligand_atom_type" not in normalized or "negative_ligand_pos" not in normalized:
            raise KeyError("negative ligand requires both atom_type and pos")
        out["negative_ligand_atom_type"] = as_atom_type(
            normalized["negative_ligand_atom_type"],
            "negative_ligand_atom_type",
        )
        out["negative_ligand_pos"] = as_pos(normalized["negative_ligand_pos"], "negative_ligand_pos")
        out["negative_ligand_bond_edge_index"] = as_edge_index(
            normalized.get("negative_ligand_bond_edge_index"),
            "negative_ligand_bond_edge_index",
        )
        out["negative_ligand_bond_type"] = as_bond_type(
            normalized.get("negative_ligand_bond_type"),
            "negative_ligand_bond_type",
        )
    return out

def prefix_component(prefix: str, component: dict[str, Any], *, include_bonds: bool) -> dict[str, Any]:
    if "atom_type" not in component or "pos" not in component:
        raise KeyError(f"{prefix} component requires atom_type and pos")
    out = {
        f"{prefix}_atom_type": component["atom_type"],
        f"{prefix}_pos": component["pos"],
    }
    if include_bonds:
        edge_key = "source_edge_index" if prefix == "source" else f"{prefix}_bond_edge_index"
        bond_key = "source_bond_type" if prefix == "source" else f"{prefix}_bond_type"
        out[edge_key] = component.get("bond_edge_index")
        out[bond_key] = component.get("bond_type")
    return out


def empty_ligand_like(component: dict[str, Any]) -> dict[str, torch.Tensor]:
    pos = torch.as_tensor(component["pos"])
    return {
        "atom_type": torch.empty(0, dtype=torch.long),
        "pos": pos.new_zeros((0, 3), dtype=torch.float32),
        "bond_edge_index": torch.empty(2, 0, dtype=torch.long),
        "bond_type": torch.empty(0, dtype=torch.long),
    }


def as_atom_type(value: Any, key: str) -> torch.Tensor:
    tensor = torch.as_tensor(value).long().view(-1)
    return tensor


def as_pos(value: Any, key: str) -> torch.Tensor:
    tensor = torch.as_tensor(value).float()
    if tensor.dim() != 2 or tensor.shape[-1] != 3:
        raise ValueError(f"{key} must have shape [N, 3], got {tuple(tensor.shape)}")
    return tensor


def as_edge_index(value: Any, key: str) -> torch.Tensor:
    if value is None:
        return torch.empty(2, 0, dtype=torch.long)
    tensor = torch.as_tensor(value).long()
    if tensor.numel() == 0:
        return torch.empty(2, 0, dtype=torch.long)
    if tensor.dim() != 2:
        raise ValueError(f"{key} must have shape [2, E], got {tuple(tensor.shape)}")
    if tensor.shape[0] != 2 and tensor.shape[1] == 2:
        tensor = tensor.t().contiguous()
    if tensor.shape[0] != 2:
        raise ValueError(f"{key} must have shape [2, E], got {tuple(tensor.shape)}")
    return tensor.contiguous()


def as_bond_type(value: Any, key: str) -> torch.Tensor:
    if value is None:
        return torch.empty(0, dtype=torch.long)
    return torch.as_tensor(value).long().view(-1)
