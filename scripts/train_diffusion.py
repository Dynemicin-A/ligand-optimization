#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path
from time import time

import torch
import yaml
from torch.utils.data import DataLoader, random_split
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pco_backbone import (  # noqa: E402
    BackboneConfig,
    ComplexDenoiserBackbone,
    DiffusionConfig,
    ProteinConditionedDiffusion,
)
from pco_backbone.data import (  # noqa: E402
    PTRecordDataset,
    SyntheticH2LConfig,
    SyntheticH2LDataset,
    collate_complex_records,
    move_batch_to_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train diffusion-first protein-conditioned optimizer.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/train_synthetic_tiny.yaml")
    parser.add_argument("--outdir", type=Path, default=ROOT / "outputs/diffusion_tiny")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    with path.open("r") as f:
        return yaml.safe_load(f)


def seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_arg)


def build_dataset(cfg: dict):
    data_cfg = cfg.get("data", {})
    kind = data_cfg.get("kind", "synthetic")
    if kind == "synthetic":
        synth = SyntheticH2LConfig(
            num_samples=data_cfg.get("num_samples", 256),
            min_protein_atoms=data_cfg.get("min_protein_atoms", 24),
            max_protein_atoms=data_cfg.get("max_protein_atoms", 48),
            min_ligand_atoms=data_cfg.get("min_ligand_atoms", 8),
            max_ligand_atoms=data_cfg.get("max_ligand_atoms", 18),
            num_ligand_atom_types=cfg["model"].get("num_ligand_atom_types", 16),
            num_protein_atom_types=cfg["model"].get("num_protein_atom_types", 32),
            num_bond_types=cfg["model"].get("num_bond_types", 5),
            seed=cfg.get("seed", 2024),
        )
        return SyntheticH2LDataset(synth)
    if kind == "pt_manifest":
        return PTRecordDataset(data_cfg["manifest"])
    raise ValueError(f"unknown data.kind: {kind}")


def build_model(cfg: dict) -> ProteinConditionedDiffusion:
    model_cfg = BackboneConfig(**cfg["model"])
    diffusion_cfg = DiffusionConfig(**cfg["diffusion"])
    backbone = ComplexDenoiserBackbone(model_cfg)
    return ProteinConditionedDiffusion(backbone, diffusion_cfg)


def split_dataset(dataset, valid_fraction: float, seed: int):
    if valid_fraction <= 0:
        return dataset, None
    n_valid = max(1, int(round(len(dataset) * valid_fraction)))
    n_valid = min(n_valid, len(dataset) - 1)
    n_train = len(dataset) - n_valid
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [n_train, n_valid], generator=generator)


def make_loader(dataset, train_cfg: dict, shuffle: bool, num_workers_override: int | None):
    num_workers = train_cfg.get("num_workers", 0) if num_workers_override is None else num_workers_override
    return DataLoader(
        dataset,
        batch_size=train_cfg.get("batch_size", 4),
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_complex_records,
        pin_memory=train_cfg.get("pin_memory", False),
    )


@torch.no_grad()
def evaluate(model, loader, device: torch.device, max_batches: int) -> dict[str, float]:
    model.eval()
    sums: dict[str, float] = {}
    count = 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = move_batch_to_device(batch, device)
        out = model.training_loss(batch)
        for key, value in out.items():
            if key.endswith("loss") or key == "loss":
                sums[key] = sums.get(key, 0.0) + float(value.detach().cpu())
        count += 1
    return {key: value / max(count, 1) for key, value in sums.items()}


def save_checkpoint(
    path: Path,
    model: ProteinConditionedDiffusion,
    optimizer: torch.optim.Optimizer,
    step: int,
    epoch: int,
    cfg: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": cfg,
            "backbone_config": asdict(model.backbone.config),
            "diffusion_config": asdict(model.config),
        },
        path,
    )


def load_checkpoint(path: Path, model: ProteinConditionedDiffusion, optimizer: torch.optim.Optimizer):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    return int(ckpt.get("step", 0)), int(ckpt.get("epoch", 0))


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    train_cfg = cfg.get("train", {})
    seed_all(cfg.get("seed", 2024))

    device = resolve_device(args.device)
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    dataset = build_dataset(cfg)
    train_set, valid_set = split_dataset(dataset, train_cfg.get("valid_fraction", 0.1), cfg.get("seed", 2024))
    train_loader = make_loader(train_set, train_cfg, shuffle=True, num_workers_override=args.num_workers)
    valid_loader = None
    if valid_set is not None:
        valid_loader = make_loader(valid_set, train_cfg, shuffle=False, num_workers_override=args.num_workers)

    model = build_model(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.get("lr", 1e-4),
        weight_decay=train_cfg.get("weight_decay", 1e-4),
    )

    step = 0
    start_epoch = 0
    if args.resume is not None:
        step, start_epoch = load_checkpoint(args.resume, model, optimizer)

    max_steps = args.max_steps or train_cfg.get("max_steps", 1000)
    grad_clip = train_cfg.get("grad_clip", 1.0)
    log_every = train_cfg.get("log_every", 10)
    valid_every = train_cfg.get("valid_every", 100)
    save_every = train_cfg.get("save_every", 200)
    valid_batches = train_cfg.get("valid_batches", 8)

    metrics_path = args.outdir / "metrics.jsonl"
    start_time = time()
    pbar = tqdm(total=max_steps, initial=step, desc="training", dynamic_ncols=True)
    epoch = start_epoch
    while step < max_steps:
        model.train()
        for batch in train_loader:
            batch = move_batch_to_device(batch, device)
            out = model.training_loss(batch)
            loss = out["loss"]

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            step += 1
            pbar.update(1)

            if step % log_every == 0 or step == 1:
                log = {
                    "step": step,
                    "epoch": epoch,
                    "elapsed_sec": round(time() - start_time, 3),
                    "train_loss": float(out["loss"].detach().cpu()),
                    "train_pos_loss": float(out["pos_loss"].detach().cpu()),
                    "train_atom_loss": float(out["atom_loss"].detach().cpu()),
                    "train_bond_loss": float(out["bond_loss"].detach().cpu()),
                    "lr": optimizer.param_groups[0]["lr"],
                }
                pbar.set_postfix(loss=f"{log['train_loss']:.4f}")
                with metrics_path.open("a") as f:
                    f.write(json.dumps(log) + "\n")

            if valid_loader is not None and step % valid_every == 0:
                valid_metrics = evaluate(model, valid_loader, device, valid_batches)
                log = {"step": step, **{f"valid_{k}": v for k, v in valid_metrics.items()}}
                with metrics_path.open("a") as f:
                    f.write(json.dumps(log) + "\n")

            if step % save_every == 0:
                save_checkpoint(args.outdir / f"checkpoint_step_{step}.pt", model, optimizer, step, epoch, cfg)

            if step >= max_steps:
                break
        epoch += 1

    pbar.close()
    save_checkpoint(args.outdir / "checkpoint_last.pt", model, optimizer, step, epoch, cfg)
    print(f"done: step={step}, checkpoint={args.outdir / 'checkpoint_last.pt'}")


if __name__ == "__main__":
    main()
