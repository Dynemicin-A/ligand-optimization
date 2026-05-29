from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .layers import (
    LigandUpdateBlock,
    RadialBasis,
    ScalarMessageBlock,
    SinusoidalTimeEmbedding,
    build_knn_edges,
    make_mlp,
)


@dataclass
class BackboneConfig:
    num_ligand_atom_types: int = 16
    num_protein_atom_types: int = 32
    num_bond_types: int = 5
    hidden_dim: int = 192
    time_dim: int = 64
    rbf_dim: int = 32
    num_blocks: int = 6
    ligand_knn: int = 16
    protein_knn: int = 24
    cross_knn: int = 32
    source_knn: int = 16
    cutoff: float = 12.0
    dropout: float = 0.0


class ComplexDenoiserBackbone(nn.Module):
    """Diffusion-first protein-conditioned molecule optimization trunk.

    The first wrapper trains this as a denoiser. The trunk is kept general enough
    for later flow/BFN ablations, but those are not first-version requirements.
    It predicts ligand coordinate noise/updates, atom logits, and bond logits
    while keeping protein coordinates fixed.
    """

    def __init__(self, config: BackboneConfig):
        super().__init__()
        self.config = config
        h = config.hidden_dim

        self.ligand_atom_embed = nn.Embedding(config.num_ligand_atom_types, h)
        self.protein_atom_embed = nn.Embedding(config.num_protein_atom_types, h)
        self.time_embed = SinusoidalTimeEmbedding(config.time_dim)
        self.time_to_hidden = make_mlp(config.time_dim, h, h, num_layers=2)
        self.rbf = RadialBasis(config.rbf_dim, config.cutoff)

        self.protein_blocks = nn.ModuleList(
            [
                ScalarMessageBlock(h, config.rbf_dim, config.time_dim, dropout=config.dropout)
                for _ in range(config.num_blocks)
            ]
        )
        self.ligand_blocks = nn.ModuleList(
            [
                LigandUpdateBlock(h, config.rbf_dim, config.time_dim, dropout=config.dropout)
                for _ in range(config.num_blocks)
            ]
        )
        self.source_blocks = nn.ModuleList(
            [
                ScalarMessageBlock(h, config.rbf_dim, config.time_dim, dropout=config.dropout)
                for _ in range(config.num_blocks)
            ]
        )
        self.source_pool_proj = make_mlp(h, h, h, num_layers=2, dropout=config.dropout)

        self.atom_head = make_mlp(h, h, config.num_ligand_atom_types, num_layers=3, dropout=config.dropout)
        self.pos_head = make_mlp(h, h, 3, num_layers=3, dropout=config.dropout)
        self.bond_head = make_mlp(
            h * 2 + config.rbf_dim,
            h,
            config.num_bond_types,
            num_layers=3,
            dropout=config.dropout,
        )
        self.global_head = make_mlp(h * 2, h, 1, num_layers=3, dropout=config.dropout)

    def _edge_rbf(
        self,
        src_pos: torch.Tensor,
        dst_pos: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        if edge_index.numel() == 0:
            return src_pos.new_zeros((0, self.config.rbf_dim))
        src, dst = edge_index
        dist = torch.linalg.norm(dst_pos[dst] - src_pos[src], dim=-1)
        return self.rbf(dist)

    @staticmethod
    def _graph_mean(h: torch.Tensor, batch: torch.Tensor, n_graphs: int) -> torch.Tensor:
        out = h.new_zeros(n_graphs, h.shape[-1])
        count = h.new_zeros(n_graphs, 1)
        if h.numel() == 0:
            return out
        out.index_add_(0, batch, h)
        count.index_add_(0, batch, torch.ones(h.shape[0], 1, device=h.device, dtype=h.dtype))
        return out / count.clamp_min(1.0)

    @staticmethod
    def _infer_n_graphs(*batches: torch.Tensor | None) -> int:
        max_graph = 0
        for batch in batches:
            if batch is not None and batch.numel() > 0:
                max_graph = max(max_graph, int(batch.max().item()))
        return max_graph + 1

    def forward(
        self,
        *,
        protein_atom_type: torch.Tensor,
        protein_pos: torch.Tensor,
        protein_batch: torch.Tensor,
        ligand_atom_type: torch.Tensor,
        ligand_pos: torch.Tensor,
        ligand_batch: torch.Tensor,
        time: torch.Tensor,
        ligand_edge_index: torch.Tensor | None = None,
        source_atom_type: torch.Tensor | None = None,
        source_pos: torch.Tensor | None = None,
        source_batch: torch.Tensor | None = None,
        source_edge_index: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if ligand_pos.shape[-1] != 3 or protein_pos.shape[-1] != 3:
            raise ValueError("positions must have shape [N, 3]")
        has_source = source_atom_type is not None
        if has_source:
            if source_pos is None or source_batch is None:
                raise ValueError("source_pos and source_batch are required with source_atom_type")
            if source_pos.shape[-1] != 3:
                raise ValueError("source positions must have shape [N, 3]")

        n_graphs = self._infer_n_graphs(
            protein_batch,
            ligand_batch,
            source_batch if has_source else None,
        )
        time_emb_graph = self.time_embed(time.to(ligand_pos.device))
        if time_emb_graph.shape[0] == 1 and n_graphs > 1:
            time_emb_graph = time_emb_graph.expand(n_graphs, -1)
        if time_emb_graph.shape[0] != n_graphs:
            raise ValueError(f"time must have shape [1] or [{n_graphs}]")

        time_ligand = time_emb_graph[ligand_batch]
        time_protein = time_emb_graph[protein_batch]
        ligand_h = self.ligand_atom_embed(ligand_atom_type) + self.time_to_hidden(time_ligand)
        protein_h = self.protein_atom_embed(protein_atom_type) + self.time_to_hidden(time_protein)
        source_h = None
        source_time = None
        if has_source:
            source_h = self.ligand_atom_embed(source_atom_type)
            source_time = source_h.new_zeros(source_h.shape[0], self.config.time_dim)

        pp_edge_index = build_knn_edges(
            protein_pos,
            protein_pos,
            protein_batch,
            protein_batch,
            self.config.protein_knn,
            exclude_self=True,
            radius=self.config.cutoff,
        )
        if ligand_edge_index is None:
            ll_edge_index = build_knn_edges(
                ligand_pos,
                ligand_pos,
                ligand_batch,
                ligand_batch,
                self.config.ligand_knn,
                exclude_self=True,
                radius=self.config.cutoff,
            )
        else:
            ll_edge_index = ligand_edge_index.to(ligand_pos.device)
        pl_edge_index = build_knn_edges(
            ligand_pos,
            protein_pos,
            ligand_batch,
            protein_batch,
            self.config.cross_knn,
            radius=self.config.cutoff,
        )
        ss_edge_index = None
        sl_edge_index = None
        if has_source:
            if source_edge_index is None:
                ss_edge_index = build_knn_edges(
                    source_pos,
                    source_pos,
                    source_batch,
                    source_batch,
                    self.config.source_knn,
                    exclude_self=True,
                    radius=self.config.cutoff,
                )
            else:
                ss_edge_index = source_edge_index.to(source_pos.device)
            sl_edge_index = build_knn_edges(
                ligand_pos,
                source_pos,
                ligand_batch,
                source_batch,
                self.config.source_knn,
                radius=self.config.cutoff,
            )

        for protein_block, ligand_block, source_block in zip(
            self.protein_blocks,
            self.ligand_blocks,
            self.source_blocks,
        ):
            pp_rbf = self._edge_rbf(protein_pos, protein_pos, pp_edge_index)
            protein_h = protein_block(
                protein_h,
                protein_pos,
                pp_edge_index,
                pp_rbf,
                time_protein,
            )
            if has_source:
                ss_rbf = self._edge_rbf(source_pos, source_pos, ss_edge_index)
                source_h = source_block(
                    source_h,
                    source_pos,
                    ss_edge_index,
                    ss_rbf,
                    source_time,
                )

            ll_rbf = self._edge_rbf(ligand_pos, ligand_pos, ll_edge_index)
            pl_rbf = self._edge_rbf(protein_pos, ligand_pos, pl_edge_index)
            sl_rbf = self._edge_rbf(source_pos, ligand_pos, sl_edge_index) if has_source else None
            ligand_h, ligand_pos = ligand_block(
                ligand_h,
                ligand_pos,
                protein_h,
                protein_pos,
                ll_edge_index,
                ll_rbf,
                pl_edge_index,
                pl_rbf,
                time_ligand,
                source_h=source_h if has_source else None,
                source_pos=source_pos if has_source else None,
                sl_edge_index=sl_edge_index if has_source else None,
                sl_rbf=sl_rbf,
            )
            if has_source:
                source_pool = self._graph_mean(source_h, source_batch, n_graphs)
                ligand_h = ligand_h + self.source_pool_proj(source_pool[ligand_batch])

        pos_update = self.pos_head(ligand_h)
        atom_logits = self.atom_head(ligand_h)

        ll_rbf = self._edge_rbf(ligand_pos, ligand_pos, ll_edge_index)
        src, dst = ll_edge_index
        bond_input = torch.cat([ligand_h[src], ligand_h[dst], ll_rbf], dim=-1)
        bond_logits = self.bond_head(bond_input)

        ligand_pool = self._graph_mean(ligand_h, ligand_batch, n_graphs)
        protein_pool = self._graph_mean(protein_h, protein_batch, n_graphs)
        complex_score = self.global_head(torch.cat([ligand_pool, protein_pool], dim=-1)).squeeze(-1)

        return {
            "pos_update": pos_update,
            "atom_logits": atom_logits,
            "bond_edge_index": ll_edge_index,
            "bond_logits": bond_logits,
            "complex_score": complex_score,
            "ligand_h": ligand_h,
            "protein_h": protein_h,
            **({"source_h": source_h} if has_source else {}),
        }
