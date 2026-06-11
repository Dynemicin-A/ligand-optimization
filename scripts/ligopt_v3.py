#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def passthrough_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified v3 command wrapper for ligand optimization preprocessing, training, sampling, and review."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, help_text in [
        ("preprocess", "Preprocess generic protein-ligand CSV/JSONL/root datasets into canonical v3 records."),
        ("preprocess-molgenbench-h2l", "Preprocess MolGenBench v3 H2L series into canonical v3 records."),
        ("train-pretrain", "Train v3 PDBbind/SBDD pretraining config."),
        ("train-h2l", "Train v3 ChEMBL/H2L finetuning config."),
        ("eval-loss", "Evaluate a v3 checkpoint on the configured validation split."),
        ("sample-h2l", "Sample H2L ligands from a v3 checkpoint and manifest."),
        ("review", "Run post-run quality and improvement review."),
    ]:
        sub = subparsers.add_parser(name, help=help_text)
        sub.add_argument("args", nargs=argparse.REMAINDER)
    return parser


def strip_separator(args: list[str]) -> list[str]:
    return args[1:] if args and args[0] == "--" else args


def run_script(script: str, args: list[str]) -> int:
    cmd = [PYTHON, str(ROOT / "scripts" / script), *strip_separator(args)]
    print(" ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=ROOT, check=False).returncode


def run_train(config: str, outdir: str, args: list[str]) -> int:
    passthrough = strip_separator(args)
    cmd = [
        PYTHON,
        str(ROOT / "scripts" / "train_diffusion.py"),
        "--config",
        config,
        "--outdir",
        outdir,
        *passthrough,
    ]
    print(" ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=ROOT, check=False).returncode


def main() -> None:
    parsed = passthrough_parser().parse_args()
    if parsed.command == "preprocess":
        raise SystemExit(run_script("preprocess_complex_dataset.py", parsed.args))
    if parsed.command == "preprocess-molgenbench-h2l":
        raise SystemExit(run_script("preprocess_molgenbench_h2l.py", parsed.args))
    if parsed.command == "train-pretrain":
        raise SystemExit(
            run_train(
                "configs/pretrain_complexes_backbone_v3_4090.yaml",
                "outputs/v3_pretrain",
                parsed.args,
            )
        )
    if parsed.command == "train-h2l":
        raise SystemExit(
            run_train(
                "configs/train_h2l_chembl_backbone_v3_4090.yaml",
                "outputs/v3_h2l",
                parsed.args,
            )
        )
    if parsed.command == "eval-loss":
        raise SystemExit(run_script("evaluate_checkpoint_loss.py", parsed.args))
    if parsed.command == "sample-h2l":
        raise SystemExit(run_script("sample_h2l_manifest.py", parsed.args))
    if parsed.command == "review":
        raise SystemExit(run_script("review_experiment.py", parsed.args))
    raise AssertionError(f"unhandled command: {parsed.command}")


if __name__ == "__main__":
    main()
