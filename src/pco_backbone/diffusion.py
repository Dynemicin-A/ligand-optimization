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

        total_loss = (
            self.config.pos_loss_weight * pos_loss
            + self.config.atom_loss_weight * atom_loss
            + self.config.bond_loss_weight * bond_loss
        )
        return {
            "loss": total_loss,
            "pos_loss": pos_loss.detach(),
            "atom_loss": atom_loss.detach(),
            "bond_loss": bond_loss.detach(),
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
