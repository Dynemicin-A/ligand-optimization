from __future__ import annotations

import math

import torch
from torch import nn


def make_mlp(
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    num_layers: int = 2,
    activation: type[nn.Module] = nn.SiLU,
    final_activation: nn.Module | None = None,
    dropout: float = 0.0,
) -> nn.Sequential:
    if num_layers < 1:
        raise ValueError("num_layers must be >= 1")

    layers: list[nn.Module] = []
    last_dim = in_dim
    for _ in range(num_layers - 1):
        layers.extend([nn.Linear(last_dim, hidden_dim), activation()])
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        last_dim = hidden_dim
    layers.append(nn.Linear(last_dim, out_dim))
    if final_activation is not None:
        layers.append(final_activation)
    return nn.Sequential(*layers)


class SinusoidalTimeEmbedding(nn.Module):
    """Maps scalar diffusion/flow time to a learned embedding."""

    def __init__(self, dim: int, max_period: int = 10_000):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("time embedding dim must be even")
        self.dim = dim
        self.max_period = max_period
        self.proj = make_mlp(dim, dim * 2, dim, num_layers=2)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 0:
            t = t[None]
        t = t.float().view(-1)
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half, device=t.device, dtype=t.dtype)
            / max(half - 1, 1)
        )
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.proj(emb)


class RadialBasis(nn.Module):
    """Gaussian radial basis for pair distances."""

    def __init__(self, num_basis: int, cutoff: float):
        super().__init__()
        centers = torch.linspace(0.0, cutoff, num_basis)
        self.register_buffer("centers", centers)
        self.gamma = 1.0 / max((cutoff / num_basis) ** 2, 1e-6)

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        return torch.exp(-self.gamma * (distances[..., None] - self.centers) ** 2)


def segment_sum(values: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    out = values.new_zeros((dim_size, *values.shape[1:]))
    if index.numel() == 0:
        return out
    out.index_add_(0, index, values)
    return out


def build_knn_edges(
    query_pos: torch.Tensor,
    context_pos: torch.Tensor,
    query_batch: torch.Tensor,
    context_batch: torch.Tensor,
    k: int,
    *,
    exclude_self: bool = False,
    radius: float | None = None,
) -> torch.Tensor:
    """Returns edges as [src_context_index, dst_query_index].

    The implementation is intentionally dependency-free for preliminary work.
    It loops over graph ids, which is acceptable for small prototype batches.
    """

    if query_pos.numel() == 0 or context_pos.numel() == 0 or k <= 0:
        return torch.empty(2, 0, dtype=torch.long, device=query_pos.device)

    edges: list[torch.Tensor] = []
    graph_ids = torch.unique(query_batch)
    for graph_id in graph_ids.tolist():
        q_idx = torch.nonzero(query_batch == graph_id, as_tuple=False).flatten()
        c_idx = torch.nonzero(context_batch == graph_id, as_tuple=False).flatten()
        if q_idx.numel() == 0 or c_idx.numel() == 0:
            continue

        dist = torch.cdist(query_pos[q_idx], context_pos[c_idx])
        if exclude_self and query_pos.data_ptr() == context_pos.data_ptr():
            same = q_idx[:, None] == c_idx[None, :]
            dist = dist.masked_fill(same, float("inf"))
        if radius is not None:
            dist = dist.masked_fill(dist > radius, float("inf"))

        k_eff = min(k, c_idx.numel() - int(exclude_self))
        if k_eff <= 0:
            continue
        nn_dist, nn_local = torch.topk(dist, k=k_eff, largest=False, dim=-1)
        valid = torch.isfinite(nn_dist)
        if not valid.any():
            continue

        dst = q_idx[:, None].expand_as(nn_local)[valid]
        src = c_idx[nn_local[valid]]
        edges.append(torch.stack([src, dst], dim=0))

    if not edges:
        return torch.empty(2, 0, dtype=torch.long, device=query_pos.device)
    return torch.cat(edges, dim=1)


class ScalarMessageBlock(nn.Module):
    """Updates scalar node states from geometric neighbor messages."""

    def __init__(self, hidden_dim: int, rbf_dim: int, time_dim: int, dropout: float = 0.0):
        super().__init__()
        self.edge_mlp = make_mlp(
            hidden_dim * 2 + rbf_dim + time_dim,
            hidden_dim,
            hidden_dim,
            num_layers=3,
            dropout=dropout,
        )
        self.node_mlp = make_mlp(hidden_dim * 2, hidden_dim, hidden_dim, num_layers=2, dropout=dropout)

    def forward(
        self,
        h: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        edge_rbf: torch.Tensor,
        time_per_node: torch.Tensor,
    ) -> torch.Tensor:
        if edge_index.numel() == 0:
            return h

        src, dst = edge_index
        edge_input = torch.cat(
            [h[src], h[dst], edge_rbf, time_per_node[dst]],
            dim=-1,
        )
        msg = self.edge_mlp(edge_input)
        agg = segment_sum(msg, dst, h.shape[0])
        return h + self.node_mlp(torch.cat([h, agg], dim=-1))


class LigandUpdateBlock(nn.Module):
    """Ligand self/cross message passing with equivariant coordinate updates."""

    def __init__(self, hidden_dim: int, rbf_dim: int, time_dim: int, dropout: float = 0.0):
        super().__init__()
        edge_in = hidden_dim * 2 + rbf_dim + time_dim
        self.ll_msg = make_mlp(edge_in, hidden_dim, hidden_dim, num_layers=3, dropout=dropout)
        self.pl_msg = make_mlp(edge_in, hidden_dim, hidden_dim, num_layers=3, dropout=dropout)
        self.sl_msg = make_mlp(edge_in, hidden_dim, hidden_dim, num_layers=3, dropout=dropout)
        self.ll_coord = make_mlp(edge_in, hidden_dim, 1, num_layers=3, dropout=dropout)
        self.pl_coord = make_mlp(edge_in, hidden_dim, 1, num_layers=3, dropout=dropout)
        self.sl_coord = make_mlp(edge_in, hidden_dim, 1, num_layers=3, dropout=dropout)
        self.node_mlp = make_mlp(hidden_dim * 4, hidden_dim, hidden_dim, num_layers=2, dropout=dropout)
        self.coord_scale = nn.Parameter(torch.tensor(0.05))

    def _edge_updates(
        self,
        src_h: torch.Tensor,
        dst_h: torch.Tensor,
        src_pos: torch.Tensor,
        dst_pos: torch.Tensor,
        edge_index: torch.Tensor,
        edge_rbf: torch.Tensor,
        time_per_dst: torch.Tensor,
        msg_mlp: nn.Module,
        coord_mlp: nn.Module,
        dst_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if edge_index.numel() == 0:
            return (
                dst_h.new_zeros(dst_size, dst_h.shape[-1]),
                dst_pos.new_zeros(dst_size, 3),
            )

        src, dst = edge_index
        rel = dst_pos[dst] - src_pos[src]
        dist = torch.linalg.norm(rel, dim=-1, keepdim=True).clamp_min(1e-6)
        direction = rel / dist
        edge_input = torch.cat(
            [src_h[src], dst_h[dst], edge_rbf, time_per_dst[dst]],
            dim=-1,
        )
        msg = msg_mlp(edge_input)
        coord_weight = coord_mlp(edge_input).tanh()
        coord_msg = direction * coord_weight * self.coord_scale
        return (
            segment_sum(msg, dst, dst_size),
            segment_sum(coord_msg, dst, dst_size),
        )

    def forward(
        self,
        ligand_h: torch.Tensor,
        ligand_pos: torch.Tensor,
        protein_h: torch.Tensor,
        protein_pos: torch.Tensor,
        ll_edge_index: torch.Tensor,
        ll_rbf: torch.Tensor,
        pl_edge_index: torch.Tensor,
        pl_rbf: torch.Tensor,
        time_per_ligand: torch.Tensor,
        source_h: torch.Tensor | None = None,
        source_pos: torch.Tensor | None = None,
        sl_edge_index: torch.Tensor | None = None,
        sl_rbf: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ll_msg, ll_delta = self._edge_updates(
            ligand_h,
            ligand_h,
            ligand_pos,
            ligand_pos,
            ll_edge_index,
            ll_rbf,
            time_per_ligand,
            self.ll_msg,
            self.ll_coord,
            ligand_h.shape[0],
        )
        pl_msg, pl_delta = self._edge_updates(
            protein_h,
            ligand_h,
            protein_pos,
            ligand_pos,
            pl_edge_index,
            pl_rbf,
            time_per_ligand,
            self.pl_msg,
            self.pl_coord,
            ligand_h.shape[0],
        )
        if source_h is not None and source_pos is not None and sl_edge_index is not None and sl_rbf is not None:
            sl_msg, sl_delta = self._edge_updates(
                source_h,
                ligand_h,
                source_pos,
                ligand_pos,
                sl_edge_index,
                sl_rbf,
                time_per_ligand,
                self.sl_msg,
                self.sl_coord,
                ligand_h.shape[0],
            )
        else:
            sl_msg = ligand_h.new_zeros(ligand_h.shape)
            sl_delta = ligand_pos.new_zeros(ligand_pos.shape)

        next_h = ligand_h + self.node_mlp(torch.cat([ligand_h, ll_msg, pl_msg, sl_msg], dim=-1))
        next_pos = ligand_pos + ll_delta + pl_delta + sl_delta
        return next_h, next_pos
