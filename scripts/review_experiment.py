#!/usr/bin/env python3
"""Write post-run quality and improvement reviews for a training experiment."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import yaml
except Exception:  # pragma: no cover - PyYAML is available in training envs.
    yaml = None  # type: ignore[assignment]


ERROR_PATTERNS = (
    "OutOfMemoryError",
    "CUDA out of memory",
    "Traceback",
    "RuntimeError",
    "KekulizeException",
    "segmentation fault",
    "Segmentation fault",
    "nan",
    "NaN",
)


@dataclass
class MetricSummary:
    last_train: dict[str, Any] | None
    last_valid: dict[str, Any] | None
    best_valid: dict[str, Any] | None
    valid_count: int
    train_count: int
    nonfinite_train_steps: list[int]
    nonfinite_valid_steps: list[int]


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def metric_value(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if value is None:
        return None
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value_f):
        return None
    return value_f


def summarize_metrics(rows: list[dict[str, Any]]) -> MetricSummary:
    train_rows = [r for r in rows if "train_loss" in r]
    valid_rows = [r for r in rows if "valid_loss" in r]
    best_valid = min(valid_rows, key=lambda r: float(r.get("valid_loss", math.inf)), default=None)

    nonfinite_train = []
    for row in train_rows:
        value = row.get("train_loss")
        try:
            finite = math.isfinite(float(value))
        except (TypeError, ValueError):
            finite = False
        if not finite:
            nonfinite_train.append(int(row.get("step", -1)))

    nonfinite_valid = []
    for row in valid_rows:
        value = row.get("valid_loss")
        try:
            finite = math.isfinite(float(value))
        except (TypeError, ValueError):
            finite = False
        if not finite:
            nonfinite_valid.append(int(row.get("step", -1)))

    return MetricSummary(
        last_train=train_rows[-1] if train_rows else None,
        last_valid=valid_rows[-1] if valid_rows else None,
        best_valid=best_valid,
        valid_count=len(valid_rows),
        train_count=len(train_rows),
        nonfinite_train_steps=nonfinite_train,
        nonfinite_valid_steps=nonfinite_valid,
    )


def read_tail(path: Path, max_bytes: int = 300_000) -> str:
    if not path.exists():
        return ""
    with path.open("rb") as f:
        f.seek(0, os.SEEK_END)
        end = f.tell()
        f.seek(max(0, end - max_bytes))
        return f.read().decode("utf-8", "replace")


def scan_log(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"path": None, "exists": False, "errors": [], "tail": ""}
    text = read_tail(path)
    errors = [pattern for pattern in ERROR_PATTERNS if pattern in text]
    tail_lines = [line.strip() for line in text.splitlines()[-8:] if line.strip()]
    return {
        "path": str(path),
        "exists": path.exists(),
        "errors": errors,
        "tail": tail_lines,
    }


def load_yaml(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists() or yaml is None:
        return {}
    with path.open() as f:
        data = yaml.safe_load(f) or {}
    return data if isinstance(data, dict) else {}


def checkpoint_steps(run_dir: Path) -> dict[int, Path]:
    found: dict[int, Path] = {}
    for path in run_dir.glob("checkpoint_step_*.pt"):
        match = re.search(r"checkpoint_step_(\d+)\.pt$", path.name)
        if match:
            found[int(match.group(1))] = path
    return found


def select_saved_best(run_dir: Path, rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    steps = checkpoint_steps(run_dir)
    valid_rows = [r for r in rows if "valid_loss" in r and int(r.get("step", -1)) in steps]
    if not valid_rows:
        return None
    row = min(valid_rows, key=lambda r: float(r.get("valid_loss", math.inf)))
    step = int(row.get("step"))
    return {
        "step": step,
        "valid_loss": row.get("valid_loss"),
        "checkpoint": str(steps[step]),
    }


def infer_status(run_dir: Path, pid: int | None, log_info: dict[str, Any]) -> str:
    if pid is not None and Path(f"/proc/{pid}").exists():
        return "running"
    if log_info["errors"]:
        return "failed_or_error_in_log"
    if (run_dir / "early_stop_summary.json").exists() or (run_dir / "run_stop_summary.json").exists():
        return "stopped_reviewed"
    if (run_dir / "checkpoint_last.pt").exists():
        return "completed_or_stopped"
    return "unknown"


def analyze_quality(
    run_dir: Path,
    log_path: Path | None,
    cfg_path: Path | None,
    pid: int | None,
    overfit_delta: float,
    overfit_steps: int,
) -> tuple[dict[str, Any], list[str]]:
    rows = load_jsonl(run_dir / "metrics.jsonl")
    summary = summarize_metrics(rows)
    cfg = load_yaml(cfg_path)
    log_info = scan_log(log_path)
    status = infer_status(run_dir, pid, log_info)
    saved_best = select_saved_best(run_dir, rows)

    issues: list[str] = []
    warnings: list[str] = []

    if not rows:
        issues.append("metrics.jsonl is missing or empty")
    if log_info["errors"]:
        issues.append("log contains error patterns: " + ", ".join(log_info["errors"]))
    if summary.nonfinite_train_steps or summary.nonfinite_valid_steps:
        issues.append("non-finite losses were detected")
    if summary.valid_count == 0:
        warnings.append("no validation metrics found")
    if not (run_dir / "checkpoint_best.pt").exists() and summary.valid_count > 0:
        warnings.append("checkpoint_best.pt is missing despite validation metrics")
    if not (run_dir / "checkpoint_last.pt").exists() and status not in {"running"}:
        warnings.append("checkpoint_last.pt is missing for a non-running experiment")

    last_valid_loss = metric_value(summary.last_valid or {}, "valid_loss")
    best_valid_loss = metric_value(summary.best_valid or {}, "valid_loss")
    last_valid_step = int((summary.last_valid or {}).get("step", -1))
    best_valid_step = int((summary.best_valid or {}).get("step", -1))
    overfit = False
    if last_valid_loss is not None and best_valid_loss is not None:
        overfit = (
            last_valid_step - best_valid_step >= overfit_steps
            and last_valid_loss - best_valid_loss >= overfit_delta
        )
        if overfit:
            warnings.append(
                f"validation regression: best={best_valid_loss:.4f}@{best_valid_step}, "
                f"last={last_valid_loss:.4f}@{last_valid_step}"
            )

    manifest = (cfg.get("data") or {}).get("manifest") if isinstance(cfg.get("data"), dict) else None
    manifest_count = None
    if manifest:
        manifest_path = Path(manifest)
        if not manifest_path.is_absolute():
            manifest_path = Path.cwd() / manifest_path
        if manifest_path.exists():
            with manifest_path.open() as f:
                manifest_count = sum(1 for _ in f)
        else:
            warnings.append(f"manifest not found: {manifest}")

    review = {
        "review_type": "quality",
        "created_at_utc": utc_now(),
        "run_dir": str(run_dir),
        "status": status,
        "pid": pid,
        "config": str(cfg_path) if cfg_path else None,
        "manifest": manifest,
        "manifest_count": manifest_count,
        "metrics": {
            "train_count": summary.train_count,
            "valid_count": summary.valid_count,
            "last_train": summary.last_train,
            "last_valid": summary.last_valid,
            "best_valid": summary.best_valid,
            "saved_best": saved_best,
            "overfit_like_regression": overfit,
            "nonfinite_train_steps": summary.nonfinite_train_steps[:20],
            "nonfinite_valid_steps": summary.nonfinite_valid_steps[:20],
        },
        "checkpoints": {
            "checkpoint_best": str(run_dir / "checkpoint_best.pt") if (run_dir / "checkpoint_best.pt").exists() else None,
            "checkpoint_last": str(run_dir / "checkpoint_last.pt") if (run_dir / "checkpoint_last.pt").exists() else None,
            "num_step_checkpoints": len(checkpoint_steps(run_dir)),
        },
        "log": log_info,
        "issues": issues,
        "warnings": warnings,
    }
    return review, issues + warnings


def analyze_improvement(
    quality: dict[str, Any],
    baseline_valid_loss: float | None,
    baseline_name: str | None,
) -> dict[str, Any]:
    metrics = quality.get("metrics", {})
    last_train = metrics.get("last_train") or {}
    last_valid = metrics.get("last_valid") or {}
    best_valid = metrics.get("best_valid") or {}
    best_loss = metric_value(best_valid, "valid_loss")
    last_loss = metric_value(last_valid, "valid_loss")
    train_loss = metric_value(last_train, "train_loss")
    atom_loss = metric_value(last_valid, "valid_atom_loss")
    pos_loss = metric_value(last_valid, "valid_pos_loss")

    diagnoses: list[str] = []
    next_actions: list[str] = []

    if quality["issues"]:
        diagnoses.append("run has correctness or stability issues; fix these before interpreting metrics")
        next_actions.append("inspect log tail and rerun only after the failure mode is removed")

    if best_loss is None:
        diagnoses.append("no validation signal is available")
        next_actions.append("run a short validation-enabled job before launching longer training")
    elif baseline_valid_loss is not None:
        delta = best_loss - baseline_valid_loss
        if delta <= -0.005:
            diagnoses.append(f"improves over {baseline_name or 'baseline'} by {-delta:.4f} valid_loss")
            next_actions.append("promote checkpoint_best.pt to candidate finetune/evaluation pool")
        elif delta <= 0.01:
            diagnoses.append(f"roughly ties {baseline_name or 'baseline'}; differences are small")
            next_actions.append("evaluate generation quality before spending more pretrain time")
        else:
            diagnoses.append(f"underperforms {baseline_name or 'baseline'} by {delta:.4f} valid_loss")
            next_actions.append("do not continue this direction unless it answers a specific ablation question")

    if metrics.get("overfit_like_regression"):
        diagnoses.append("validation has regressed after the best step")
        next_actions.append("stop or keep only checkpoint_best.pt/checkpoint_best_saved.pt; avoid extending the same run")

    if train_loss is not None and last_loss is not None and train_loss + 0.1 < last_loss:
        diagnoses.append("train loss is materially below validation loss, suggesting overfit or validation mismatch")
        next_actions.append("increase regularization, reduce LR, or move to pair-level H2L finetune instead of more source=target pretrain")

    if pos_loss is not None and pos_loss > 0.98:
        diagnoses.append("coordinate/pose loss remains high; geometry may be the main bottleneck")
        next_actions.append("test stronger pocket conditioning, contact/clash auxiliary loss, or coordinate noise schedule changes")

    if atom_loss is not None and atom_loss > 0.54:
        diagnoses.append("atom-type loss is high; chemical identity modeling may be weak")
        next_actions.append("inspect atom vocabulary coverage and generated validity before scaling the backbone")

    if not diagnoses:
        diagnoses.append("no obvious failure mode from metrics/logs")
        next_actions.append("keep as candidate and compare by downstream H2L generation/evaluation")

    return {
        "review_type": "improvement",
        "created_at_utc": utc_now(),
        "run_dir": quality["run_dir"],
        "baseline": {
            "name": baseline_name,
            "valid_loss": baseline_valid_loss,
        },
        "best_valid_loss": best_loss,
        "last_valid_loss": last_loss,
        "diagnoses": diagnoses,
        "next_actions": list(dict.fromkeys(next_actions)),
    }


def write_markdown(path: Path, title: str, payload: dict[str, Any]) -> None:
    lines = [f"# {title}", "", f"- Created: `{payload.get('created_at_utc')}`", f"- Run: `{payload.get('run_dir')}`"]
    if payload.get("review_type") == "quality":
        lines.extend(
            [
                f"- Status: `{payload.get('status')}`",
                f"- PID: `{payload.get('pid')}`",
                "",
                "## Metrics",
            ]
        )
        metrics = payload.get("metrics", {})
        best = metrics.get("best_valid") or {}
        last = metrics.get("last_valid") or {}
        lines.append(f"- Best valid: step `{best.get('step')}`, loss `{best.get('valid_loss')}`")
        lines.append(f"- Last valid: step `{last.get('step')}`, loss `{last.get('valid_loss')}`")
        lines.append(f"- Saved best: `{metrics.get('saved_best')}`")
        lines.append("")
        lines.append("## Issues")
        for item in payload.get("issues", []) or ["none"]:
            lines.append(f"- {item}")
        lines.append("")
        lines.append("## Warnings")
        for item in payload.get("warnings", []) or ["none"]:
            lines.append(f"- {item}")
    else:
        baseline = payload.get("baseline", {})
        lines.extend(
            [
                f"- Baseline: `{baseline.get('name')}` loss `{baseline.get('valid_loss')}`",
                f"- Best valid loss: `{payload.get('best_valid_loss')}`",
                f"- Last valid loss: `{payload.get('last_valid_loss')}`",
                "",
                "## Diagnoses",
            ]
        )
        for item in payload.get("diagnoses", []):
            lines.append(f"- {item}")
        lines.append("")
        lines.append("## Next Actions")
        for item in payload.get("next_actions", []):
            lines.append(f"- {item}")
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--log", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--pid", type=int)
    parser.add_argument("--baseline-name")
    parser.add_argument("--baseline-valid-loss", type=float)
    parser.add_argument("--overfit-delta", type=float, default=0.04)
    parser.add_argument("--overfit-steps", type=int, default=20_000)
    parser.add_argument("--stdout", action="store_true", help="Print JSON summary to stdout.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    quality, _ = analyze_quality(
        run_dir=run_dir,
        log_path=args.log,
        cfg_path=args.config,
        pid=args.pid,
        overfit_delta=args.overfit_delta,
        overfit_steps=args.overfit_steps,
    )
    improvement = analyze_improvement(
        quality=quality,
        baseline_valid_loss=args.baseline_valid_loss,
        baseline_name=args.baseline_name,
    )

    quality_json = run_dir / "quality_review.json"
    improvement_json = run_dir / "improvement_review.json"
    quality_json.write_text(json.dumps(quality, indent=2, sort_keys=True) + "\n")
    improvement_json.write_text(json.dumps(improvement, indent=2, sort_keys=True) + "\n")
    write_markdown(run_dir / "quality_review.md", "Quality Review", quality)
    write_markdown(run_dir / "improvement_review.md", "Improvement Review", improvement)

    if args.stdout:
        print(
            json.dumps(
                {
                    "run_dir": str(run_dir),
                    "quality_review": str(quality_json),
                    "improvement_review": str(improvement_json),
                    "status": quality["status"],
                    "best_valid_loss": improvement["best_valid_loss"],
                    "next_actions": improvement["next_actions"],
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
