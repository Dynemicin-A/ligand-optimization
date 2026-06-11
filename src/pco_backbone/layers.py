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
    """Distance features for local geometric message passing.

    ``basis="gaussian"`` preserves the original implementation. The cosine
    options add a low-cost periodic distance code plus an optional smooth cutoff
    envelope, which is useful for ablations without adding new dependencies.
    """

    def __init__(self, num_basis: int, cutoff: float, basis: str = "gaussian", envelope: str = "none"):
        super().__init__()
        if num_basis < 1:
            raise ValueError("num_basis must be >= 1")
        if cutoff <= 0:
            raise ValueError("cutoff must be positive")
        if basis not in {"gaussian", "cosine", "gaussian_cosine"}:
            raise ValueError(f"unknown radial basis: {basis}")
        if envelope not in {"none", "cosine"}:
            raise ValueError(f"unknown radial envelope: {envelope}")

        self.num_basis = num_basis
        self.cutoff = float(cutoff)
        self.basis = basis
        self.envelope = envelope

        gaussian_dim = num_basis if basis == "gaussian" else 0
        if basis == "gaussian_cosine":
            gaussian_dim = num_basis // 2
        cosine_dim = num_basis - gaussian_dim

        centers = torch.linspace(0.0, cutoff, max(gaussian_dim, 1))
        self.register_buffer("centers", centers)
        freqs = torch.arange(1, max(cosine_dim, 1) + 1, dtype=torch.float)
        self.register_buffer("freqs", freqs, persistent=False)
        self.gaussian_dim = gaussian_dim
        self.cosine_dim = cosine_dim
        self.gamma = 1.0 / max((cutoff / max(gaussian_dim, 1)) ** 2, 1e-6)

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        features: list[torch.Tensor] = []
        if self.gaussian_dim > 0:
            centers = self.centers[: self.gaussian_dim].to(dtype=distances.dtype)
            features.append(torch.exp(-self.gamma * (distances[..., None] - centers) ** 2))
        if self.cosine_dim > 0:
            scaled = (distances / self.cutoff).clamp(min=0.0, max=1.0)
            freqs = self.freqs[: self.cosine_dim].to(device=distances.device, dtype=distances.dtype)
            features.append(torch.cos(math.pi * scaled[..., None] * freqs))
        out = torch.cat(features, dim=-1)
        if self.envelope == "cosine":
            scaled = (distances / self.cutoff).clamp(min=0.0, max=1.0)
            envelope = 0.5 * (torch.cos(math.pi * scaled) + 1.0)
            envelope = envelope * (distances <= self.cutoff).to(distances.dtype)
            out = out * envelope[..., None]
        return out


class ResidualFFN(nn.Module):
    """Pre-norm residual feed-forward block for scalar node states."""

    def __init__(
        self,
        hidden_dim: int,
        dropout: float = 0.0,
        multiplier: int = 2,
        *,
        use_layer_norm: bool = True,
        layer_scale_init: float = 1.0,
    ):
        super().__init__()
        inner_dim = max(hidden_dim, int(hidden_dim * multiplier))
        self.norm = nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity()
        self.ffn = make_mlp(hidden_dim, inner_dim, hidden_dim, num_layers=2, dropout=dropout)
        if layer_scale_init > 0:
            self.layer_scale = nn.Parameter(torch.full((hidden_dim,), float(layer_scale_init)))
        else:
            self.register_parameter("layer_scale", None)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        delta = self.ffn(self.norm(h))
        if self.layer_scale is not None:
            delta = delta * self.layer_scale
        return h + delta


class EdgePairUpdateBlock(nn.Module):
    """Updates edge-level pair states and returns messages for destination nodes.

    This is a memory-bounded Pairformer-lite primitive. Instead of materializing
    a dense N x N pair tensor, it keeps pair states only on selected local/cross
    edges. That preserves the useful single/pair separation from AF3/Boltz-style
    trunks while staying practical for protein pockets with hundreds to
    thousands of atoms on 4090-class GPUs.
    """

    def __init__(
        self,
        hidden_dim: int,
        pair_dim: int,
        rbf_dim: int,
        time_dim: int,
        dropout: float = 0.0,
        *,
        use_layer_norm: bool = True,
        edge_gate: bool = True,
        ffn_multiplier: int = 2,
        layer_scale_init: float = 0.1,
    ):
        super().__init__()
        edge_in = hidden_dim * 2 + pair_dim + rbf_dim + time_dim
        self.src_norm = nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity()
        self.dst_norm = nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity()
        self.pair_norm = nn.LayerNorm(pair_dim) if use_layer_norm else nn.Identity()
        self.edge_update = make_mlp(edge_in, pair_dim * ffn_multiplier, pair_dim, num_layers=3, dropout=dropout)
        self.edge_gate = (
            make_mlp(edge_in, pair_dim, pair_dim, num_layers=2, final_activation=nn.Sigmoid(), dropout=dropout)
            if edge_gate
            else None
        )
        self.pair_ffn = ResidualFFN(
            pair_dim,
            dropout=dropout,
            multiplier=ffn_multiplier,
            use_layer_norm=use_layer_norm,
            layer_scale_init=layer_scale_init,
        )
        self.node_proj = make_mlp(pair_dim, hidden_dim, hidden_dim, num_layers=2, dropout=dropout)

    def forward(
        self,
        src_h: torch.Tensor,
        dst_h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_rbf: torch.Tensor,
        time_per_dst: torch.Tensor,
        pair_h: torch.Tensor,
        dst_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if edge_index.numel() == 0:
            return pair_h, dst_h.new_zeros(dst_size, dst_h.shape[-1])

        src, dst = edge_index
        src_state = self.src_norm(src_h)
        dst_state = self.dst_norm(dst_h)
        pair_state = self.pair_norm(pair_h)
        edge_input = torch.cat(
            [src_state[src], dst_state[dst], pair_state, edge_rbf, time_per_dst[dst]],
            dim=-1,
        )
        delta = self.edge_update(edge_input)
        if self.edge_gate is not None:
            delta = delta * self.edge_gate(edge_input)
        next_pair = self.pair_ffn(pair_h + delta)
        node_msg = segment_sum(self.node_proj(next_pair), dst, dst_size)
        return next_pair, node_msg


class PairTrunkBlock(nn.Module):
    """Edge-sparse single/pair trunk for ligand-centered complex reasoning."""

    def __init__(
        self,
        hidden_dim: int,
        pair_dim: int,
        rbf_dim: int,
        time_dim: int,
        dropout: float = 0.0,
        *,
        use_layer_norm: bool = True,
        edge_gate: bool = True,
        ffn_multiplier: int = 2,
        layer_scale_init: float = 0.1,
    ):
        super().__init__()
        self.ll_update = EdgePairUpdateBlock(
            hidden_dim,
            pair_dim,
            rbf_dim,
            time_dim,
            dropout=dropout,
            use_layer_norm=use_layer_norm,
            edge_gate=edge_gate,
            ffn_multiplier=ffn_multiplier,
            layer_scale_init=layer_scale_init,
        )
        self.pl_update = EdgePairUpdateBlock(
            hidden_dim,
            pair_dim,
            rbf_dim,
            time_dim,
            dropout=dropout,
            use_layer_norm=use_layer_norm,
            edge_gate=edge_gate,
            ffn_multiplier=ffn_multiplier,
            layer_scale_init=layer_scale_init,
        )
        self.sl_update = EdgePairUpdateBlock(
            hidden_dim,
            pair_dim,
            rbf_dim,
            time_dim,
            dropout=dropout,
            use_layer_norm=use_layer_norm,
            edge_gate=edge_gate,
            ffn_multiplier=ffn_multiplier,
            layer_scale_init=layer_scale_init,
        )
        self.node_update = make_mlp(hidden_dim * 4, hidden_dim * ffn_multiplier, hidden_dim, num_layers=2, dropout=dropout)
        self.post_ffn = ResidualFFN(
            hidden_dim,
            dropout=dropout,
            multiplier=ffn_multiplier,
            use_layer_norm=use_layer_norm,
            layer_scale_init=layer_scale_init,
        )

    def forward(
        self,
        *,
        ligand_h: torch.Tensor,
        protein_h: torch.Tensor,
        source_h: torch.Tensor | None,
        ll_edge_index: torch.Tensor,
        pl_edge_index: torch.Tensor,
        sl_edge_index: torch.Tensor | None,
        ll_rbf: torch.Tensor,
        pl_rbf: torch.Tensor,
        sl_rbf: torch.Tensor | None,
        time_ligand: torch.Tensor,
        ll_pair_h: torch.Tensor,
        pl_pair_h: torch.Tensor,
        sl_pair_h: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        ll_pair_h, ll_msg = self.ll_update(
            ligand_h,
            ligand_h,
            ll_edge_index,
            ll_rbf,
            time_ligand,
            ll_pair_h,
            ligand_h.shape[0],
        )
        pl_pair_h, pl_msg = self.pl_update(
            protein_h,
            ligand_h,
            pl_edge_index,
            pl_rbf,
            time_ligand,
            pl_pair_h,
            ligand_h.shape[0],
        )
        if source_h is not None and sl_edge_index is not None and sl_rbf is not None and sl_pair_h is not None:
            sl_pair_h, sl_msg = self.sl_update(
                source_h,
                ligand_h,
                sl_edge_index,
                sl_rbf,
                time_ligand,
                sl_pair_h,
                ligand_h.shape[0],
            )
        else:
            sl_msg = ligand_h.new_zeros(ligand_h.shape)

        ligand_h = ligand_h + self.node_update(torch.cat([ligand_h, ll_msg, pl_msg, sl_msg], dim=-1))
        return self.post_ffn(ligand_h), ll_pair_h, pl_pair_h, sl_pair_h


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

    def __init__(
        self,
        hidden_dim: int,
        rbf_dim: int,
        time_dim: int,
        dropout: float = 0.0,
        *,
        use_layer_norm: bool = False,
        use_residual_ffn: bool = False,
        ffn_multiplier: int = 2,
        edge_gate: bool = False,
        layer_scale_init: float = 1.0,
    ):
        super().__init__()
        edge_in = hidden_dim * 2 + rbf_dim + time_dim
        self.use_layer_norm = use_layer_norm
        self.use_residual_ffn = use_residual_ffn
        self.edge_gate = edge_gate
        self.input_norm = nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity()
        self.edge_mlp = make_mlp(
            edge_in,
            hidden_dim,
            hidden_dim,
            num_layers=3,
            dropout=dropout,
        )
        self.gate_mlp = (
            make_mlp(edge_in, hidden_dim, hidden_dim, num_layers=2, final_activation=nn.Sigmoid(), dropout=dropout)
            if edge_gate
            else None
        )
        self.node_mlp = make_mlp(hidden_dim * 2, hidden_dim, hidden_dim, num_layers=2, dropout=dropout)
        self.post_ffn = (
            ResidualFFN(
                hidden_dim,
                dropout=dropout,
                multiplier=ffn_multiplier,
                use_layer_norm=use_layer_norm,
                layer_scale_init=layer_scale_init,
            )
            if use_residual_ffn
            else nn.Identity()
        )

    def forward(
        self,
        h: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        edge_rbf: torch.Tensor,
        time_per_node: torch.Tensor,
    ) -> torch.Tensor:
        if edge_index.numel() == 0:
            return self.post_ffn(h)

        src, dst = edge_index
        h_msg = self.input_norm(h)
        edge_input = torch.cat(
            [h_msg[src], h_msg[dst], edge_rbf, time_per_node[dst]],
            dim=-1,
        )
        msg = self.edge_mlp(edge_input)
        if self.gate_mlp is not None:
            msg = msg * self.gate_mlp(edge_input)
        agg = segment_sum(msg, dst, h.shape[0])
        next_h = h + self.node_mlp(torch.cat([h, agg], dim=-1))
        return self.post_ffn(next_h)


class LigandUpdateBlock(nn.Module):
    """Ligand self/cross message passing with equivariant coordinate updates."""

    def __init__(
        self,
        hidden_dim: int,
        rbf_dim: int,
        time_dim: int,
        dropout: float = 0.0,
        *,
        use_layer_norm: bool = False,
        use_residual_ffn: bool = False,
        ffn_multiplier: int = 2,
        edge_gate: bool = False,
        layer_scale_init: float = 1.0,
    ):
        super().__init__()
        edge_in = hidden_dim * 2 + rbf_dim + time_dim
        self.input_norm = nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity()
        self.edge_gate = edge_gate
        self.ll_msg = make_mlp(edge_in, hidden_dim, hidden_dim, num_layers=3, dropout=dropout)
        self.pl_msg = make_mlp(edge_in, hidden_dim, hidden_dim, num_layers=3, dropout=dropout)
        self.sl_msg = make_mlp(edge_in, hidden_dim, hidden_dim, num_layers=3, dropout=dropout)
        self.ll_gate = (
            make_mlp(edge_in, hidden_dim, hidden_dim, num_layers=2, final_activation=nn.Sigmoid(), dropout=dropout)
            if edge_gate
            else None
        )
        self.pl_gate = (
            make_mlp(edge_in, hidden_dim, hidden_dim, num_layers=2, final_activation=nn.Sigmoid(), dropout=dropout)
            if edge_gate
            else None
        )
        self.sl_gate = (
            make_mlp(edge_in, hidden_dim, hidden_dim, num_layers=2, final_activation=nn.Sigmoid(), dropout=dropout)
            if edge_gate
            else None
        )
        self.ll_coord = make_mlp(edge_in, hidden_dim, 1, num_layers=3, dropout=dropout)
        self.pl_coord = make_mlp(edge_in, hidden_dim, 1, num_layers=3, dropout=dropout)
        self.sl_coord = make_mlp(edge_in, hidden_dim, 1, num_layers=3, dropout=dropout)
        self.node_mlp = make_mlp(hidden_dim * 4, hidden_dim, hidden_dim, num_layers=2, dropout=dropout)
        self.post_ffn = (
            ResidualFFN(
                hidden_dim,
                dropout=dropout,
                multiplier=ffn_multiplier,
                use_layer_norm=use_layer_norm,
                layer_scale_init=layer_scale_init,
            )
            if use_residual_ffn
            else nn.Identity()
        )
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
        gate_mlp: nn.Module | None,
        dst_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if edge_index.numel() == 0:
            return (
                dst_h.new_zeros(dst_size, dst_h.shape[-1]),
                dst_pos.new_zeros(dst_size, 3),
            )

        src, dst = edge_index
        src_h = self.input_norm(src_h)
        dst_h = self.input_norm(dst_h)
        rel = dst_pos[dst] - src_pos[src]
        dist = torch.linalg.norm(rel, dim=-1, keepdim=True).clamp_min(1e-6)
        direction = rel / dist
        edge_input = torch.cat(
            [src_h[src], dst_h[dst], edge_rbf, time_per_dst[dst]],
            dim=-1,
        )
        msg = msg_mlp(edge_input)
        if gate_mlp is not None:
            msg = msg * gate_mlp(edge_input)
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
            self.ll_gate,
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
            self.pl_gate,
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
                self.sl_gate,
                ligand_h.shape[0],
            )
        else:
            sl_msg = ligand_h.new_zeros(ligand_h.shape)
            sl_delta = ligand_pos.new_zeros(ligand_pos.shape)

        next_h = ligand_h + self.node_mlp(torch.cat([ligand_h, ll_msg, pl_msg, sl_msg], dim=-1))
        next_pos = ligand_pos + ll_delta + pl_delta + sl_delta
        return self.post_ffn(next_h), next_pos
