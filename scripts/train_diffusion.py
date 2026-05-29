#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import signal
import sys
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from time import time

import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
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
    parser.add_argument(
        "--data-manifest",
        type=Path,
        default=None,
        help="Override data.manifest from the config for alternate processed datasets.",
    )
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--init-model",
        type=Path,
        default=None,
        help="Load only model weights from a pretraining checkpoint; optimizer and step start fresh.",
    )
    parser.add_argument(
        "--init-weights",
        choices=["auto", "model", "ema"],
        default="auto",
        help="Which checkpoint weights to load for --init-model. auto prefers EMA when present.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    with path.open("r") as f:
        return yaml.safe_load(f)


def seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_distributed() -> tuple[bool, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
    return distributed, rank, local_rank, world_size


def is_main_process(rank: int) -> bool:
    return rank == 0


def unwrap_model(model: torch.nn.Module) -> ProteinConditionedDiffusion:
    if isinstance(model, DistributedDataParallel):
        return model.module
    return model


def resolve_device(device_arg: str, local_rank: int = 0, distributed: bool = False) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            if distributed:
                torch.cuda.set_device(local_rank)
                return torch.device("cuda", local_rank)
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if device_arg == "cuda" and distributed:
        torch.cuda.set_device(local_rank)
        return torch.device("cuda", local_rank)
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


class ModelEMA:
    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {}
        self.reset(model)

    def reset(self, model: torch.nn.Module) -> None:
        self.shadow = {
            key: value.detach().clone()
            for key, value in unwrap_model(model).state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for key, value in unwrap_model(model).state_dict().items():
            value = value.detach()
            if key not in self.shadow:
                self.shadow[key] = value.clone()
            elif value.is_floating_point():
                self.shadow[key].mul_(self.decay).add_(value, alpha=1.0 - self.decay)
            else:
                self.shadow[key].copy_(value)

    def load_state_dict(self, state: dict[str, torch.Tensor], model: torch.nn.Module) -> None:
        current = unwrap_model(model).state_dict()
        self.shadow = {
            key: value.detach().clone().to(current[key].device) if key in current else value.detach().clone()
            for key, value in state.items()
        }

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {key: value.detach().cpu().clone() for key, value in self.shadow.items()}

    @contextmanager
    def apply_to(self, model: torch.nn.Module):
        raw_model = unwrap_model(model)
        current = {
            key: value.detach().clone()
            for key, value in raw_model.state_dict().items()
        }
        raw_model.load_state_dict(self.shadow, strict=False)
        try:
            yield
        finally:
            raw_model.load_state_dict(current, strict=False)


def split_dataset(dataset, valid_fraction: float, seed: int):
    if valid_fraction <= 0:
        return dataset, None
    n_valid = max(1, int(round(len(dataset) * valid_fraction)))
    n_valid = min(n_valid, len(dataset) - 1)
    n_train = len(dataset) - n_valid
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [n_train, n_valid], generator=generator)


def make_loader(
    dataset,
    train_cfg: dict,
    shuffle: bool,
    num_workers_override: int | None,
    sampler=None,
):
    num_workers = train_cfg.get("num_workers", 0) if num_workers_override is None else num_workers_override
    return DataLoader(
        dataset,
        batch_size=train_cfg.get("batch_size", 4),
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_complex_records,
        pin_memory=train_cfg.get("pin_memory", False),
        drop_last=train_cfg.get("drop_last", False),
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
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    epoch: int,
    cfg: dict,
    ema: ModelEMA | None = None,
) -> None:
    raw_model = unwrap_model(model)
    payload = {
        "step": step,
        "epoch": epoch,
        "model_state": raw_model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": cfg,
        "backbone_config": asdict(raw_model.backbone.config),
        "diffusion_config": asdict(raw_model.config),
    }
    if ema is not None:
        payload["ema_model_state"] = ema.state_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(
    path: Path,
    model: ProteinConditionedDiffusion,
    optimizer: torch.optim.Optimizer,
    ema: ModelEMA | None = None,
):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    loaded_ema = False
    if ema is not None and "ema_model_state" in ckpt:
        ema.load_state_dict(ckpt["ema_model_state"], model)
        loaded_ema = True
    return int(ckpt.get("step", 0)), int(ckpt.get("epoch", 0)), loaded_ema


def load_model_weights(path: Path, model: ProteinConditionedDiffusion, weights: str = "auto") -> str:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    use_ema = weights in {"auto", "ema"} and "ema_model_state" in ckpt
    if weights == "ema" and not use_ema:
        raise KeyError(f"checkpoint has no ema_model_state: {path}")
    model.load_state_dict(ckpt["ema_model_state"] if use_ema else ckpt["model_state"])
    return "ema" if use_ema else "model"


def metric_improved(value: float, best: float | None, mode: str, min_delta: float) -> bool:
    if best is None:
        return True
    if mode == "min":
        return value < best - min_delta
    if mode == "max":
        return value > best + min_delta
    raise ValueError(f"early_stop.mode must be 'min' or 'max', got {mode!r}")


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    if args.data_manifest is not None:
        cfg.setdefault("data", {})["manifest"] = str(args.data_manifest)
    train_cfg = cfg.get("train", {})
    distributed, rank, local_rank, world_size = setup_distributed()
    main_process = is_main_process(rank)
    stop_requested = {"value": False}

    def request_stop(signum, _frame) -> None:
        stop_requested["value"] = True
        if main_process:
            print(f"received signal {signum}; saving checkpoint after current step", flush=True)

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    seed_all(cfg.get("seed", 2024))

    device = resolve_device(args.device, local_rank=local_rank, distributed=distributed)
    args.outdir.mkdir(parents=True, exist_ok=True)
    if main_process:
        (args.outdir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    if distributed:
        dist.barrier()

    dataset = build_dataset(cfg)
    train_set, valid_set = split_dataset(dataset, train_cfg.get("valid_fraction", 0.1), cfg.get("seed", 2024))
    train_sampler = None
    if distributed:
        train_sampler = DistributedSampler(
            train_set,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=cfg.get("seed", 2024),
            drop_last=train_cfg.get("drop_last", False),
        )
    train_loader = make_loader(
        train_set,
        train_cfg,
        shuffle=True,
        num_workers_override=args.num_workers,
        sampler=train_sampler,
    )
    valid_loader = None
    if valid_set is not None and main_process:
        valid_loader = make_loader(valid_set, train_cfg, shuffle=False, num_workers_override=args.num_workers)

    model = build_model(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.get("lr", 1e-4),
        weight_decay=train_cfg.get("weight_decay", 1e-4),
    )
    ema_cfg = train_cfg.get("ema", {}) or {}
    ema_enabled = bool(ema_cfg.get("enabled", False))
    ema = ModelEMA(model, decay=float(ema_cfg.get("decay", 0.999))) if ema_enabled else None
    ema_validate = bool(ema_cfg.get("validate", True))

    step = 0
    start_epoch = 0
    if args.resume is not None and args.init_model is not None:
        raise ValueError("--resume and --init-model are mutually exclusive")
    if args.resume is not None:
        step, start_epoch, loaded_ema = load_checkpoint(args.resume, model, optimizer, ema=ema)
        if ema is not None and not loaded_ema:
            ema.reset(model)
    elif args.init_model is not None:
        loaded_weights = load_model_weights(args.init_model, model, weights=args.init_weights)
        if ema is not None:
            ema.reset(model)
        if main_process:
            print(f"initialized {loaded_weights} weights from {args.init_model}", flush=True)
    if distributed:
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
            find_unused_parameters=True,
        )

    max_steps = args.max_steps or train_cfg.get("max_steps", 1000)
    grad_clip = train_cfg.get("grad_clip", 1.0)
    log_every = train_cfg.get("log_every", 10)
    valid_every = train_cfg.get("valid_every", 100)
    save_every = train_cfg.get("save_every", 200)
    valid_batches = train_cfg.get("valid_batches", 8)
    early_cfg = train_cfg.get("early_stopping", {}) or {}
    early_enabled = bool(early_cfg.get("enabled", False)) and valid_loader is not None
    early_monitor = early_cfg.get("monitor", "valid_loss")
    early_mode = early_cfg.get("mode", "min")
    early_min_delta = float(early_cfg.get("min_delta", 0.0))
    early_patience = int(early_cfg.get("patience", 3))
    early_start_after = int(early_cfg.get("start_after", 0))
    early_save_best = bool(early_cfg.get("save_best", True))
    best_metric: float | None = None
    best_step = step
    bad_validations = 0

    metrics_path = args.outdir / "metrics.jsonl"
    start_time = time()
    pbar = tqdm(
        total=max_steps,
        initial=step,
        desc="training",
        dynamic_ncols=True,
        disable=not main_process,
    )
    epoch = start_epoch
    while step < max_steps and not stop_requested["value"]:
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        for batch in train_loader:
            if stop_requested["value"]:
                break
            batch = move_batch_to_device(batch, device)
            out = model(batch)
            loss = out["loss"]

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            if ema is not None:
                ema.update(model)

            step += 1
            pbar.update(1)

            if main_process and (step % log_every == 0 or step == 1):
                log = {
                    "step": step,
                    "epoch": epoch,
                    "elapsed_sec": round(time() - start_time, 3),
                    "train_loss": float(out["loss"].detach().cpu()),
                    "train_pos_loss": float(out["pos_loss"].detach().cpu()),
                    "train_atom_loss": float(out["atom_loss"].detach().cpu()),
                    "train_bond_loss": float(out["bond_loss"].detach().cpu()),
                    "train_hard_negative_loss": float(out["hard_negative_loss"].detach().cpu()),
                    "train_positive_score": float(out["positive_score"].detach().cpu()),
                    "train_negative_score": float(out["negative_score"].detach().cpu()),
                    "lr": optimizer.param_groups[0]["lr"],
                }
                pbar.set_postfix(loss=f"{log['train_loss']:.4f}")
                with metrics_path.open("a") as f:
                    f.write(json.dumps(log) + "\n")

            if valid_loader is not None and step % valid_every == 0:
                eval_model = unwrap_model(model)
                if ema is not None and ema_validate:
                    with ema.apply_to(eval_model):
                        valid_metrics = evaluate(eval_model, valid_loader, device, valid_batches)
                else:
                    valid_metrics = evaluate(eval_model, valid_loader, device, valid_batches)
                if main_process:
                    log = {
                        "step": step,
                        "valid_weights": "ema" if ema is not None and ema_validate else "model",
                        **{f"valid_{k}": v for k, v in valid_metrics.items()},
                    }
                    if early_enabled:
                        monitored = log.get(early_monitor)
                        if monitored is None:
                            raise KeyError(f"early stopping monitor {early_monitor!r} not found in validation log")
                        monitored = float(monitored)
                        improved = metric_improved(monitored, best_metric, early_mode, early_min_delta)
                        if improved:
                            best_metric = monitored
                            best_step = step
                            bad_validations = 0
                            if early_save_best:
                                save_checkpoint(args.outdir / "checkpoint_best.pt", model, optimizer, step, epoch, cfg, ema=ema)
                        elif step >= early_start_after:
                            bad_validations += 1
                        log.update(
                            {
                                "early_stop_monitor": early_monitor,
                                "early_stop_value": monitored,
                                "early_stop_best": best_metric,
                                "early_stop_best_step": best_step,
                                "early_stop_bad_validations": bad_validations,
                            }
                        )
                        if step >= early_start_after and bad_validations >= early_patience:
                            log["early_stop_triggered"] = True
                            stop_requested["value"] = True
                            print(
                                "early stopping: "
                                f"{early_monitor}={monitored:.6f}, "
                                f"best={best_metric:.6f} at step={best_step}",
                                flush=True,
                            )
                    with metrics_path.open("a") as f:
                        f.write(json.dumps(log) + "\n")
            if distributed and step % valid_every == 0:
                dist.barrier()

            if main_process and step % save_every == 0:
                save_checkpoint(args.outdir / f"checkpoint_step_{step}.pt", model, optimizer, step, epoch, cfg, ema=ema)
            if distributed and step % save_every == 0:
                dist.barrier()

            if step >= max_steps or stop_requested["value"]:
                break
        epoch += 1

    pbar.close()
    if main_process:
        save_checkpoint(args.outdir / "checkpoint_last.pt", model, optimizer, step, epoch, cfg, ema=ema)
        print(f"done: step={step}, checkpoint={args.outdir / 'checkpoint_last.pt'}")
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
