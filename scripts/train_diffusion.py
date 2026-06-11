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
    collate_complex_records,
    move_batch_to_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train diffusion-first protein-conditioned optimizer.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/train_h2l_chembl_backbone_v3_4090.yaml")
    parser.add_argument("--outdir", type=Path, default=ROOT / "outputs/v3_h2l")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None, help="Override train.lr from the config.")
    parser.add_argument("--dropout", type=float, default=None, help="Override model.dropout from the config.")
    parser.add_argument("--disable-early-stopping", action="store_true")
    parser.add_argument("--disable-weak-learning-stop", action="store_true")
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=None,
        help="Accumulate this many micro-batches per optimizer step.",
    )
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


def require_v3_config(cfg: dict, path: Path) -> None:
    model_cfg = cfg.get("model", {}) or {}
    data_cfg = cfg.get("data", {}) or {}
    required_model = {
        "radial_basis": "gaussian_cosine",
        "radial_envelope": "cosine",
        "use_layer_norm": True,
        "use_residual_ffn": True,
        "edge_gate": True,
        "use_pair_trunk": True,
    }
    mismatched = [
        f"model.{key}={model_cfg.get(key)!r}"
        for key, expected in required_model.items()
        if model_cfg.get(key) != expected
    ]
    if mismatched:
        details = ", ".join(mismatched)
        raise ValueError(f"{path} is not a v3 config; expected v3 backbone fields, got {details}")
    if data_cfg.get("kind") != "pt_manifest":
        raise ValueError(f"{path} is not a v3 training config; data.kind must be 'pt_manifest'")


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
    kind = data_cfg.get("kind")
    if kind == "pt_manifest":
        return PTRecordDataset(data_cfg["manifest"])
    raise ValueError(f"unknown data.kind: {kind}")


def build_train_valid_datasets(cfg: dict, train_cfg: dict):
    data_cfg = cfg.get("data", {})
    train_set = build_dataset(cfg)
    heldout_manifest = data_cfg.get("heldout_manifest")
    if heldout_manifest:
        return train_set, PTRecordDataset(heldout_manifest)
    return split_dataset(train_set, train_cfg.get("valid_fraction", 0.1), cfg.get("seed", 2024))


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
    scalar_metrics = {
        "positive_score",
        "negative_score",
        "score_gap",
        "hard_negative_count",
        "copy_gate_accuracy",
        "distogram_accuracy",
        "contact_accuracy",
        "ranking_accuracy",
    }
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = move_batch_to_device(batch, device)
        out = model.training_loss(batch)
        for key, value in out.items():
            if value.dim() == 0 and (key.endswith("loss") or key == "loss" or key in scalar_metrics):
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
    state = ckpt["ema_model_state"] if use_ema else ckpt["model_state"]
    current = model.state_dict()
    compatible = {}
    skipped = []
    for key, value in state.items():
        if key not in current:
            skipped.append(key)
            continue
        if current[key].shape != value.shape:
            skipped.append(key)
            continue
        compatible[key] = value
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    weight_name = "ema" if use_ema else "model"
    return (
        f"{weight_name} weights "
        f"(loaded={len(compatible)}/{len(current)}, "
        f"missing={len(missing)}, skipped={len(skipped)}, unexpected={len(unexpected)})"
    )


def metric_improved(value: float, best: float | None, mode: str, min_delta: float) -> bool:
    if best is None:
        return True
    if mode == "min":
        return value < best - min_delta
    if mode == "max":
        return value > best + min_delta
    raise ValueError(f"early_stop.mode must be 'min' or 'max', got {mode!r}")


def optional_float(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class WeakLearningStopper:
    """Stop runs where validation improves on easy auxiliaries but core learning stalls."""

    def __init__(self, cfg: dict | None):
        cfg = cfg or {}
        self.enabled = bool(cfg.get("enabled", False))
        self.monitor = str(cfg.get("monitor", "valid_loss"))
        self.mode = str(cfg.get("mode", "min"))
        self.core_monitor = str(cfg.get("core_monitor", "valid_pos_loss"))
        self.core_mode = str(cfg.get("core_mode", "min"))
        self.core_min_delta = float(cfg.get("core_min_delta", 0.0))
        self.core_max = optional_float(cfg.get("core_max"))
        self.baseline_valid_loss = optional_float(cfg.get("baseline_valid_loss"))
        self.max_relative_gap = optional_float(cfg.get("max_relative_gap"))
        self.start_after = int(cfg.get("start_after", 0))
        self.min_validations = int(cfg.get("min_validations", 1))
        self.patience = int(cfg.get("patience", 1))
        self.inactive_metrics = list(cfg.get("inactive_metrics", []) or [])
        self.inactive_threshold = float(cfg.get("inactive_threshold", 1e-8))

        self.validation_count = 0
        self.best_monitor: float | None = None
        self.best_monitor_step = 0
        self.best_core: float | None = None
        self.best_core_step = 0
        self.bad_core_validations = 0

    def update(self, log: dict) -> dict:
        if not self.enabled:
            return {}

        self.validation_count += 1
        step = int(log.get("step", 0))
        monitor_value = optional_float(log.get(self.monitor))
        core_value = optional_float(log.get(self.core_monitor))
        if monitor_value is None:
            raise KeyError(f"weak_learning_stop monitor {self.monitor!r} not found in validation log")
        if core_value is None:
            raise KeyError(f"weak_learning_stop core_monitor {self.core_monitor!r} not found in validation log")

        if metric_improved(monitor_value, self.best_monitor, self.mode, 0.0):
            self.best_monitor = monitor_value
            self.best_monitor_step = step

        core_improved = metric_improved(core_value, self.best_core, self.core_mode, self.core_min_delta)
        if core_improved:
            self.best_core = core_value
            self.best_core_step = step
            if step >= self.start_after:
                self.bad_core_validations = 0
        elif step >= self.start_after:
            self.bad_core_validations += 1

        baseline_gap = None
        if self.baseline_valid_loss is not None and self.max_relative_gap is not None:
            if self.mode != "min":
                raise ValueError("weak_learning_stop baseline gate currently supports mode='min' only")
            allowed_loss = self.baseline_valid_loss * (1.0 + self.max_relative_gap)
            baseline_gap = self.best_monitor is not None and self.best_monitor > allowed_loss

        core_stuck = True
        if self.core_max is not None:
            core_stuck = core_value >= self.core_max if self.core_mode == "min" else core_value <= self.core_max

        inactive = {
            key: optional_float(log.get(key))
            for key in self.inactive_metrics
            if key in log
        }
        inactive_keys = [
            key
            for key, value in inactive.items()
            if value is not None and abs(value) <= self.inactive_threshold
        ]

        ready = step >= self.start_after and self.validation_count >= self.min_validations
        triggered = (
            ready
            and self.bad_core_validations >= self.patience
            and core_stuck
            and (baseline_gap is not False)
        )

        reason_parts: list[str] = []
        if triggered:
            reason_parts.append(
                f"{self.core_monitor} stalled: current={core_value:.6f}, "
                f"best={self.best_core:.6f}@{self.best_core_step}, "
                f"bad_validations={self.bad_core_validations}"
            )
            if baseline_gap:
                reason_parts.append(
                    f"best {self.monitor}={self.best_monitor:.6f}@{self.best_monitor_step} "
                    f"remains > baseline {self.baseline_valid_loss:.6f} "
                    f"by max_relative_gap={self.max_relative_gap:.4f}"
                )
            if inactive_keys:
                reason_parts.append(f"inactive metrics: {', '.join(inactive_keys)}")

        update = {
            "weak_stop_monitor": self.monitor,
            "weak_stop_best": self.best_monitor,
            "weak_stop_best_step": self.best_monitor_step,
            "weak_stop_core_monitor": self.core_monitor,
            "weak_stop_core_value": core_value,
            "weak_stop_core_best": self.best_core,
            "weak_stop_core_best_step": self.best_core_step,
            "weak_stop_bad_core_validations": self.bad_core_validations,
            "weak_stop_validation_count": self.validation_count,
        }
        if baseline_gap is not None:
            update["weak_stop_baseline_gap"] = baseline_gap
        if inactive_keys:
            update["weak_stop_inactive_metrics"] = inactive_keys
        if triggered:
            update["weak_stop_triggered"] = True
            update["weak_stop_reason"] = "; ".join(reason_parts)
        return update


def scalar_metric(out: dict[str, torch.Tensor], key: str) -> float:
    value = out.get(key)
    if value is None:
        return 0.0
    return float(value.detach().cpu())


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    require_v3_config(cfg, args.config)
    if args.data_manifest is not None:
        cfg.setdefault("data", {})["manifest"] = str(args.data_manifest)
    if args.dropout is not None:
        cfg.setdefault("model", {})["dropout"] = args.dropout
    train_cfg = cfg.get("train", {})
    if args.batch_size is not None:
        train_cfg["batch_size"] = args.batch_size
    if args.lr is not None:
        train_cfg["lr"] = args.lr
    if args.grad_accum_steps is not None:
        train_cfg["grad_accum_steps"] = args.grad_accum_steps
    if args.disable_early_stopping:
        train_cfg.setdefault("early_stopping", {})["enabled"] = False
    if args.disable_weak_learning_stop:
        train_cfg.setdefault("weak_learning_stop", {})["enabled"] = False
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

    train_set, valid_set = build_train_valid_datasets(cfg, train_cfg)
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
    grad_accum_steps = max(1, int(train_cfg.get("grad_accum_steps", 1) or 1))
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
    weak_stopper = WeakLearningStopper(train_cfg.get("weak_learning_stop", {}) or {})
    weak_enabled = weak_stopper.enabled and valid_loader is not None
    stop_summary: dict | None = None

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
    accum_counter = 0
    optimizer.zero_grad(set_to_none=True)
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

            (loss / grad_accum_steps).backward()
            accum_counter += 1
            if accum_counter < grad_accum_steps:
                continue
            accum_counter = 0
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
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
                    "train_distogram_loss": scalar_metric(out, "distogram_loss"),
                    "train_contact_loss": scalar_metric(out, "contact_loss"),
                    "train_distogram_accuracy": scalar_metric(out, "distogram_accuracy"),
                    "train_contact_accuracy": scalar_metric(out, "contact_accuracy"),
                    "train_copy_gate_loss": scalar_metric(out, "copy_gate_loss"),
                    "train_copy_gate_accuracy": scalar_metric(out, "copy_gate_accuracy"),
                    "train_positive_score": scalar_metric(out, "positive_score"),
                    "train_negative_score": scalar_metric(out, "negative_score"),
                    "train_score_gap": scalar_metric(out, "score_gap"),
                    "train_hard_negative_count": scalar_metric(out, "hard_negative_count"),
                    "train_ranking_accuracy": scalar_metric(out, "ranking_accuracy"),
                    "lr": optimizer.param_groups[0]["lr"],
                    "grad_accum_steps": grad_accum_steps,
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
                            stop_summary = {
                                "stop_type": "early_stopping",
                                "step": step,
                                "epoch": epoch,
                                "monitor": early_monitor,
                                "value": monitored,
                                "best": best_metric,
                                "best_step": best_step,
                                "bad_validations": bad_validations,
                            }
                            print(
                                "early stopping: "
                                f"{early_monitor}={monitored:.6f}, "
                                f"best={best_metric:.6f} at step={best_step}",
                                flush=True,
                            )
                    if weak_enabled:
                        weak_update = weak_stopper.update(log)
                        log.update(weak_update)
                        if weak_update.get("weak_stop_triggered"):
                            stop_requested["value"] = True
                            stop_summary = {
                                "stop_type": "weak_learning_stop",
                                "step": step,
                                "epoch": epoch,
                                "reason": weak_update.get("weak_stop_reason"),
                                "monitor": weak_update.get("weak_stop_monitor"),
                                "best": weak_update.get("weak_stop_best"),
                                "best_step": weak_update.get("weak_stop_best_step"),
                                "core_monitor": weak_update.get("weak_stop_core_monitor"),
                                "core_value": weak_update.get("weak_stop_core_value"),
                                "core_best": weak_update.get("weak_stop_core_best"),
                                "core_best_step": weak_update.get("weak_stop_core_best_step"),
                                "bad_core_validations": weak_update.get("weak_stop_bad_core_validations"),
                            }
                            print(
                                "weak learning stop: "
                                f"{weak_update.get('weak_stop_reason')}",
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
        if stop_summary is not None:
            stop_summary = {
                **stop_summary,
                "created_at": time(),
                "checkpoint_last": str(args.outdir / "checkpoint_last.pt"),
                "checkpoint_best": str(args.outdir / "checkpoint_best.pt"),
            }
            (args.outdir / "run_stop_summary.json").write_text(
                json.dumps(stop_summary, indent=2) + "\n"
            )
        print(f"done: step={step}, checkpoint={args.outdir / 'checkpoint_last.pt'}")
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
