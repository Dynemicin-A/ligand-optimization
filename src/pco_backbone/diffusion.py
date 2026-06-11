from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .model import ComplexDenoiserBackbone


@dataclass
class DiffusionConfig:
    num_timesteps: int = 1_000
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    atom_mask_token: int | None = None
    pos_loss_weight: float = 1.0
    atom_loss_weight: float = 1.0
    bond_loss_weight: float = 0.2
    hard_negative_loss_weight: float = 0.0
    hard_negative_margin: float = 0.2
    hard_negative_score_only: bool = True
    hard_negative_grad_side: str = "positive"
    distogram_loss_weight: float = 0.0
    contact_loss_weight: float = 0.0
    copy_gate_loss_weight: float = 0.0
    copy_gate_copy_threshold: float = 1.25
    copy_gate_move_threshold: float = 3.0


class ProteinConditionedDiffusion(nn.Module):
    """Diffusion-first training wrapper around the shared backbone."""

    def __init__(self, backbone: ComplexDenoiserBackbone, config: DiffusionConfig):
        super().__init__()
        self.backbone = backbone
        self.config = config

        betas = torch.linspace(config.beta_start, config.beta_end, config.num_timesteps)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)

    @property
    def atom_mask_token(self) -> int:
        if self.config.atom_mask_token is not None:
            return self.config.atom_mask_token
        return self.backbone.config.num_ligand_atom_types - 1

    def sample_timesteps(self, n_graphs: int, device: torch.device) -> torch.Tensor:
        return torch.randint(0, self.config.num_timesteps, (n_graphs,), device=device)

    def q_sample_pos(
        self,
        clean_pos: torch.Tensor,
        graph_t: torch.Tensor,
        ligand_batch: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(clean_pos)
        alpha_bar = self.alpha_bars[graph_t][ligand_batch].view(-1, 1)
        noisy_pos = alpha_bar.sqrt() * clean_pos + (1.0 - alpha_bar).sqrt() * noise
        return noisy_pos, noise

    def q_sample_atom_type(
        self,
        clean_atom_type: torch.Tensor,
        graph_t: torch.Tensor,
        ligand_batch: torch.Tensor,
    ) -> torch.Tensor:
        mask_prob = (graph_t.float() / max(self.config.num_timesteps - 1, 1))[ligand_batch]
        should_mask = torch.rand_like(mask_prob) < mask_prob
        noisy = clean_atom_type.clone()
        noisy[should_mask] = self.atom_mask_token
        return noisy

    @staticmethod
    def _num_graphs(protein_batch: torch.Tensor, ligand_batch: torch.Tensor) -> int:
        return int(max(protein_batch.max(), ligand_batch.max()).item()) + 1

    @staticmethod
    def _bond_targets_for_edges(
        edge_index: torch.Tensor,
        num_nodes: int,
        target_edge_index: torch.Tensor | None,
        target_bond_type: torch.Tensor | None,
    ) -> torch.Tensor:
        target = torch.zeros(edge_index.shape[1], dtype=torch.long, device=edge_index.device)
        if (
            target_edge_index is None
            or target_bond_type is None
            or target_edge_index.numel() == 0
            or edge_index.numel() == 0
        ):
            return target

        dense = torch.zeros((num_nodes, num_nodes), dtype=torch.long, device=edge_index.device)
        src, dst = target_edge_index.to(edge_index.device)
        bond_type = target_bond_type.to(edge_index.device).long()
        dense[src, dst] = bond_type
        dense[dst, src] = bond_type
        pred_src, pred_dst = edge_index
        return dense[pred_src, pred_dst]

    def _hard_negative_loss(
        self,
        *,
        batch: dict[str, torch.Tensor],
        protein_atom_type: torch.Tensor,
        protein_pos: torch.Tensor,
        protein_batch: torch.Tensor,
        ligand_atom_type: torch.Tensor,
        ligand_pos: torch.Tensor,
        ligand_batch: torch.Tensor,
        n_graphs: int,
        optional_source: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        zero = ligand_pos.new_zeros(())
        if self.config.hard_negative_loss_weight <= 0:
            return zero, zero, zero, zero, zero

        if (
            "negative_ligand_atom_type" in batch
            and "negative_ligand_pos" in batch
            and "negative_ligand_batch" in batch
            and batch["negative_ligand_atom_type"].numel() > 0
        ):
            negative_atom_type = batch["negative_ligand_atom_type"]
            negative_pos = batch["negative_ligand_pos"]
            negative_batch = batch["negative_ligand_batch"]
            negative_edge_index = batch.get("negative_ligand_bond_edge_index")
        elif (
            "source_atom_type" in optional_source
            and "source_pos" in optional_source
            and "source_batch" in optional_source
            and optional_source["source_atom_type"].numel() > 0
        ):
            negative_atom_type = optional_source["source_atom_type"]
            negative_pos = optional_source["source_pos"]
            negative_batch = optional_source["source_batch"]
            negative_edge_index = optional_source.get("source_edge_index")
        else:
            return zero, zero, zero, zero, zero

        grad_side = self.config.hard_negative_grad_side
        if grad_side not in {"both", "positive", "negative"}:
            raise ValueError("hard_negative_grad_side must be one of: both, positive, negative")

        time_zero = ligand_pos.new_zeros(n_graphs)

        def score_forward(
            atom_type: torch.Tensor,
            pos: torch.Tensor,
            graph_batch: torch.Tensor,
            edge_index: torch.Tensor | None,
        ) -> torch.Tensor:
            out = self.backbone(
                protein_atom_type=protein_atom_type,
                protein_pos=protein_pos,
                protein_batch=protein_batch,
                ligand_atom_type=atom_type,
                ligand_pos=pos,
                ligand_batch=graph_batch,
                time=time_zero,
                ligand_edge_index=edge_index,
                score_only=self.config.hard_negative_score_only,
                **optional_source,
            )
            return out["complex_score"]

        if grad_side == "negative":
            with torch.no_grad():
                pos_score_all = score_forward(
                    ligand_atom_type,
                    ligand_pos,
                    ligand_batch,
                    batch.get("ligand_edge_index"),
                )
            neg_score_all = score_forward(
                negative_atom_type,
                negative_pos,
                negative_batch,
                negative_edge_index,
            )
        else:
            pos_score_all = score_forward(
                ligand_atom_type,
                ligand_pos,
                ligand_batch,
                batch.get("ligand_edge_index"),
            )
            if grad_side == "positive":
                with torch.no_grad():
                    neg_score_all = score_forward(
                        negative_atom_type,
                        negative_pos,
                        negative_batch,
                        negative_edge_index,
                    )
            else:
                neg_score_all = score_forward(
                    negative_atom_type,
                    negative_pos,
                    negative_batch,
                    negative_edge_index,
                )

        neg_graphs = torch.unique(negative_batch)
        pos_score = pos_score_all[neg_graphs]
        neg_score = neg_score_all[neg_graphs]
        ranking = F.relu(self.config.hard_negative_margin - pos_score + neg_score).mean()
        score_gap = (pos_score - neg_score).mean()
        hard_negative_count = ligand_pos.new_tensor(float(neg_graphs.numel()))
        return ranking, pos_score.mean(), neg_score.mean(), score_gap, hard_negative_count

    def _distogram_loss(
        self,
        out: dict[str, torch.Tensor],
        protein_pos: torch.Tensor,
        ligand_pos: torch.Tensor,
    ) -> torch.Tensor:
        logits = out.get("distogram_logits")
        edge_index = out.get("protein_ligand_edge_index")
        if self.config.distogram_loss_weight <= 0 or logits is None or edge_index is None or edge_index.numel() == 0:
            return ligand_pos.new_zeros(())

        src, dst = edge_index
        dist = torch.linalg.norm(ligand_pos[dst] - protein_pos[src], dim=-1)
        num_bins = logits.shape[-1]
        boundaries = torch.linspace(
            self.backbone.config.distogram_min,
            self.backbone.config.distogram_max,
            max(num_bins - 1, 1),
            device=dist.device,
            dtype=dist.dtype,
        )
        target = torch.bucketize(dist, boundaries).clamp(max=num_bins - 1).long()
        return F.cross_entropy(logits, target)

    def _contact_loss(
        self,
        out: dict[str, torch.Tensor],
        protein_pos: torch.Tensor,
        ligand_pos: torch.Tensor,
    ) -> torch.Tensor:
        logits = out.get("contact_logits")
        edge_index = out.get("protein_ligand_edge_index")
        if self.config.contact_loss_weight <= 0 or logits is None or edge_index is None or edge_index.numel() == 0:
            return ligand_pos.new_zeros(())

        src, dst = edge_index
        dist = torch.linalg.norm(ligand_pos[dst] - protein_pos[src], dim=-1)
        target = (dist <= self.backbone.config.contact_cutoff).to(logits.dtype)
        return F.binary_cross_entropy_with_logits(logits, target)

    def _copy_gate_targets(
        self,
        ligand_atom_type: torch.Tensor,
        ligand_pos: torch.Tensor,
        ligand_batch: torch.Tensor,
        source_atom_type: torch.Tensor,
        source_pos: torch.Tensor,
        source_batch: torch.Tensor,
        num_classes: int,
    ) -> torch.Tensor:
        copy_class = 0
        mutate_class = min(1, num_classes - 1)
        move_class = min(2, num_classes - 1)
        grow_class = min(3, num_classes - 1)
        targets = torch.full(
            (ligand_atom_type.shape[0],),
            grow_class,
            dtype=torch.long,
            device=ligand_atom_type.device,
        )
        if source_atom_type.numel() == 0 or ligand_atom_type.numel() == 0:
            return targets

        for graph_id in torch.unique(ligand_batch).tolist():
            ligand_idx = torch.nonzero(ligand_batch == graph_id, as_tuple=False).flatten()
            source_idx = torch.nonzero(source_batch == graph_id, as_tuple=False).flatten()
            if ligand_idx.numel() == 0 or source_idx.numel() == 0:
                continue
            dist = torch.cdist(ligand_pos[ligand_idx], source_pos[source_idx])
            nearest_dist, nearest_local = dist.min(dim=1)
            nearest_source = source_idx[nearest_local]
            same_type = ligand_atom_type[ligand_idx] == source_atom_type[nearest_source]
            near_copy = nearest_dist <= self.config.copy_gate_copy_threshold
            near_move = (
                (nearest_dist > self.config.copy_gate_copy_threshold)
                & (nearest_dist <= self.config.copy_gate_move_threshold)
            )
            targets[ligand_idx[near_copy & same_type]] = copy_class
            targets[ligand_idx[near_copy & ~same_type]] = mutate_class
            targets[ligand_idx[near_move]] = move_class
        return targets

    def _copy_gate_loss(
        self,
        out: dict[str, torch.Tensor],
        ligand_atom_type: torch.Tensor,
        ligand_pos: torch.Tensor,
        ligand_batch: torch.Tensor,
        optional_source: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        logits = out.get("copy_gate_logits")
        if (
            self.config.copy_gate_loss_weight <= 0
            or logits is None
            or "source_atom_type" not in optional_source
            or "source_pos" not in optional_source
            or "source_batch" not in optional_source
        ):
            return ligand_pos.new_zeros(())

        targets = self._copy_gate_targets(
            ligand_atom_type=ligand_atom_type,
            ligand_pos=ligand_pos,
            ligand_batch=ligand_batch,
            source_atom_type=optional_source["source_atom_type"],
            source_pos=optional_source["source_pos"],
            source_batch=optional_source["source_batch"],
            num_classes=logits.shape[-1],
        )
        return F.cross_entropy(logits, targets)

    def training_loss(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        protein_atom_type = batch["protein_atom_type"]
        protein_pos = batch["protein_pos"]
        protein_batch = batch["protein_batch"]
        ligand_atom_type = batch["ligand_atom_type"]
        ligand_pos = batch["ligand_pos"]
        ligand_batch = batch["ligand_batch"]

        n_graphs = self._num_graphs(protein_batch, ligand_batch)
        graph_t = batch.get("time_index")
        if graph_t is None:
            graph_t = self.sample_timesteps(n_graphs, ligand_pos.device)
        time = graph_t.float() / max(self.config.num_timesteps - 1, 1)

        noisy_pos, pos_noise = self.q_sample_pos(ligand_pos, graph_t, ligand_batch)
        noisy_atom_type = self.q_sample_atom_type(ligand_atom_type, graph_t, ligand_batch)

        optional_source = {}
        for key in ["source_atom_type", "source_pos", "source_batch", "source_edge_index"]:
            if key in batch:
                optional_source[key] = batch[key]

        out = self.backbone(
            protein_atom_type=protein_atom_type,
            protein_pos=protein_pos,
            protein_batch=protein_batch,
            ligand_atom_type=noisy_atom_type,
            ligand_pos=noisy_pos,
            ligand_batch=ligand_batch,
            time=time,
            ligand_edge_index=batch.get("ligand_edge_index"),
            **optional_source,
        )

        pos_loss = F.mse_loss(out["pos_update"], pos_noise)
        atom_loss = F.cross_entropy(out["atom_logits"], ligand_atom_type)

        bond_target = self._bond_targets_for_edges(
            out["bond_edge_index"],
            ligand_atom_type.shape[0],
            batch.get("ligand_bond_edge_index"),
            batch.get("ligand_bond_type"),
        )
        if out["bond_logits"].numel() == 0:
            bond_loss = out["atom_logits"].sum() * 0.0
        else:
            bond_loss = F.cross_entropy(out["bond_logits"], bond_target)

        distogram_loss = self._distogram_loss(out, protein_pos, ligand_pos)
        contact_loss = self._contact_loss(out, protein_pos, ligand_pos)
        copy_gate_loss = self._copy_gate_loss(out, ligand_atom_type, ligand_pos, ligand_batch, optional_source)

        (
            hard_negative_loss,
            pos_score_mean,
            neg_score_mean,
            score_gap,
            hard_negative_count,
        ) = self._hard_negative_loss(
            batch=batch,
            protein_atom_type=protein_atom_type,
            protein_pos=protein_pos,
            protein_batch=protein_batch,
            ligand_atom_type=ligand_atom_type,
            ligand_pos=ligand_pos,
            ligand_batch=ligand_batch,
            n_graphs=n_graphs,
            optional_source=optional_source,
        )

        total_loss = (
            self.config.pos_loss_weight * pos_loss
            + self.config.atom_loss_weight * atom_loss
            + self.config.bond_loss_weight * bond_loss
            + self.config.hard_negative_loss_weight * hard_negative_loss
            + self.config.distogram_loss_weight * distogram_loss
            + self.config.contact_loss_weight * contact_loss
            + self.config.copy_gate_loss_weight * copy_gate_loss
        )
        return {
            "loss": total_loss,
            "pos_loss": pos_loss.detach(),
            "atom_loss": atom_loss.detach(),
            "bond_loss": bond_loss.detach(),
            "hard_negative_loss": hard_negative_loss.detach(),
            "distogram_loss": distogram_loss.detach(),
            "contact_loss": contact_loss.detach(),
            "copy_gate_loss": copy_gate_loss.detach(),
            "positive_score": pos_score_mean.detach(),
            "negative_score": neg_score_mean.detach(),
            "score_gap": score_gap.detach(),
            "hard_negative_count": hard_negative_count.detach(),
            "time_index": graph_t.detach(),
        }

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return self.training_loss(batch)

    @torch.no_grad()
    def sample(
        self,
        batch: dict[str, torch.Tensor],
        *,
        num_steps: int | None = None,
        temperature: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        """Minimal DDPM-style sampler for fixed target atom count.

        This is intended for program-level smoke tests and early qualitative
        inspection. Production sampling should add validity constraints,
        variable-size proposals, and molecule reconstruction.
        """

        self.eval()
        protein_batch = batch["protein_batch"]
        ligand_batch = batch["ligand_batch"]
        ligand_pos = batch["ligand_pos"]
        ligand_atom_type = batch.get("ligand_atom_type")
        if ligand_atom_type is None:
            ligand_atom_type = torch.full(
                (ligand_pos.shape[0],),
                self.atom_mask_token,
                dtype=torch.long,
                device=ligand_pos.device,
            )

        n_graphs = self._num_graphs(protein_batch, ligand_batch)
        steps = num_steps or self.config.num_timesteps
        step_indices = torch.linspace(
            self.config.num_timesteps - 1,
            0,
            steps,
            device=ligand_pos.device,
        ).long()
        x_t = torch.randn_like(ligand_pos) * temperature
        atom_t = ligand_atom_type.clone()

        optional_source = {}
        for key in ["source_atom_type", "source_pos", "source_batch", "source_edge_index"]:
            if key in batch:
                optional_source[key] = batch[key]

        last_out = None
        for t_scalar in step_indices:
            graph_t = torch.full((n_graphs,), int(t_scalar.item()), device=ligand_pos.device, dtype=torch.long)
            time = graph_t.float() / max(self.config.num_timesteps - 1, 1)
            last_out = self.backbone(
                protein_atom_type=batch["protein_atom_type"],
                protein_pos=batch["protein_pos"],
                protein_batch=protein_batch,
                ligand_atom_type=atom_t,
                ligand_pos=x_t,
                ligand_batch=ligand_batch,
                time=time,
                ligand_edge_index=batch.get("ligand_edge_index"),
                **optional_source,
            )

            beta_t = self.betas[t_scalar].view(1, 1)
            alpha_t = self.alphas[t_scalar].view(1, 1)
            alpha_bar_t = self.alpha_bars[t_scalar].view(1, 1)
            eps = last_out["pos_update"]
            mean = (x_t - beta_t / (1.0 - alpha_bar_t).sqrt().clamp_min(1e-6) * eps) / alpha_t.sqrt()
            if int(t_scalar.item()) > 0:
                noise = torch.randn_like(x_t) * temperature
                x_t = mean + beta_t.sqrt() * noise
            else:
                x_t = mean
            atom_t = last_out["atom_logits"].argmax(dim=-1)

        assert last_out is not None
        return {
            "ligand_pos": x_t,
            "ligand_atom_type": atom_t,
            "bond_edge_index": last_out["bond_edge_index"],
            "bond_type": last_out["bond_logits"].argmax(dim=-1),
            "atom_logits": last_out["atom_logits"],
            "bond_logits": last_out["bond_logits"],
        }
