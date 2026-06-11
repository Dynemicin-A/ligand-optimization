#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/zhangxuanhao/miniconda3/envs/ligopt-v3/bin/python")
INIT_MODEL = Path("outputs/v3_baseline_pretrain_20260608_072616_bs6/checkpoint_best.pt")


@dataclass
class Experiment:
    name: str
    config: str
    max_steps: int
    extra_args: list[str]
    require_edit_labels: bool = False


EXPERIMENTS = [
    # Foundation first: this is the most important run and should exceed 10h
    # on the current shared workstation based on the 2200-step gradaccum smoke.
    Experiment(
        name="denoise_gradaccum3_60k",
        config="configs/train_h2l_chembl_backbone_v3_denoise_ablation_4090.yaml",
        max_steps=60_000,
        extra_args=["--batch-size", "2", "--grad-accum-steps", "3"],
    ),
    Experiment(
        name="edit_policy_60k",
        config="configs/train_h2l_chembl_backbone_v3_edit_policy_4090.yaml",
        max_steps=60_000,
        extra_args=["--batch-size", "2"],
        require_edit_labels=True,
    ),
    Experiment(
        name="interaction_40k",
        config="configs/train_h2l_chembl_backbone_v3_interaction_4090.yaml",
        max_steps=40_000,
        extra_args=["--batch-size", "2"],
    ),
    Experiment(
        name="ranking_eval_40k",
        config="configs/train_h2l_chembl_backbone_v3_ranking_4090.yaml",
        max_steps=40_000,
        extra_args=["--batch-size", "2"],
    ),
]


@dataclass
class GpuApp:
    pid: int
    process_name: str
    used_memory_mib: int


@dataclass
class GpuState:
    index: int
    uuid: str
    memory_used_mib: int
    utilization_pct: int
    compute_apps: list[GpuApp]

    @property
    def has_large_job(self) -> bool:
        return any(app.used_memory_mib >= 4_000 for app in self.compute_apps)

    @property
    def has_gmx_only(self) -> bool:
        return bool(self.compute_apps) and all("gmx" in app.process_name for app in self.compute_apps)


def log_line(path: Path, message: str) -> None:
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} {message}"
    print(line, flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(line + "\n")


def run_smi(args: list[str]) -> str:
    return subprocess.check_output(
        ["timeout", "20s", "nvidia-smi", *args],
        text=True,
        stderr=subprocess.STDOUT,
    )


def query_gpus(log_path: Path) -> list[GpuState]:
    try:
        gpu_csv = run_smi(
            [
                "--query-gpu=index,uuid,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ]
        )
    except Exception as exc:
        log_line(log_path, f"nvidia_smi_gpu_query_failed={type(exc).__name__}: {exc}")
        return []

    states: dict[str, GpuState] = {}
    for row in csv.reader(gpu_csv.splitlines()):
        if len(row) < 4:
            continue
        index = int(row[0].strip())
        uuid = row[1].strip()
        states[uuid] = GpuState(
            index=index,
            uuid=uuid,
            memory_used_mib=int(row[2].strip()),
            utilization_pct=int(row[3].strip()),
            compute_apps=[],
        )

    try:
        apps_csv = run_smi(
            [
                "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ]
        )
    except Exception:
        apps_csv = ""

    for row in csv.reader(apps_csv.splitlines()):
        if len(row) < 4:
            continue
        uuid = row[0].strip()
        if uuid not in states:
            continue
        try:
            states[uuid].compute_apps.append(
                GpuApp(
                    pid=int(row[1].strip()),
                    process_name=row[2].strip(),
                    used_memory_mib=int(row[3].strip()),
                )
            )
        except ValueError:
            pass
    return sorted(states.values(), key=lambda state: state.index)


def choose_gpu(states: list[GpuState], candidates: list[int], allow_gmx_share: bool) -> int | None:
    by_index = {state.index: state for state in states}
    for index in candidates:
        state = by_index.get(index)
        if state is None or state.has_large_job:
            continue
        if not state.compute_apps and state.memory_used_mib <= 900 and state.utilization_pct <= 10:
            return index
    if allow_gmx_share:
        for index in candidates:
            state = by_index.get(index)
            if state is None or state.has_large_job:
                continue
            if state.has_gmx_only and state.memory_used_mib <= 1_200:
                return index
    return None


def check_required_inputs(exp: Experiment) -> None:
    config = ROOT / exp.config
    if not config.exists():
        raise FileNotFoundError(config)
    if not (ROOT / INIT_MODEL).exists():
        raise FileNotFoundError(ROOT / INIT_MODEL)
    if exp.require_edit_labels:
        for manifest in [
            ROOT / "data/processed_chembl_h2l_round6_delta05/train_edit_labels/manifest.txt",
            ROOT / "data/processed_chembl_h2l_round6_delta05/val_edit_labels/manifest.txt",
        ]:
            if not manifest.exists():
                raise FileNotFoundError(manifest)


def run_experiment(exp: Experiment, gpu: int, group: str, log_path: Path) -> int:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = f"{group}_{exp.name}_gpu{gpu}_{stamp}"
    outdir = ROOT / "outputs" / run_name
    train_log = ROOT / "logs" / f"{run_name}.log"
    cmd = [
        str(PYTHON),
        "scripts/train_diffusion.py",
        "--config",
        exp.config,
        "--outdir",
        str(outdir),
        "--device",
        "cuda",
        "--max-steps",
        str(exp.max_steps),
        "--init-model",
        str(INIT_MODEL),
        "--init-weights",
        "auto",
        *exp.extra_args,
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    log_line(log_path, "launch " + json.dumps({"run": run_name, "gpu": gpu, "cmd": cmd}))
    with train_log.open("w") as handle:
        proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
    log_line(log_path, f"finished run={run_name} returncode={proc.returncode}")
    return proc.returncode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submit a long H2L decomposition experiment queue.")
    parser.add_argument("--group", default=f"long_h2l_decomp_{time.strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--candidate-gpus", default="1,3,5,6,2,4,7")
    parser.add_argument("--poll-sec", type=int, default=120)
    parser.add_argument("--allow-gmx-share", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidates = [int(part) for part in args.candidate_gpus.split(",") if part.strip()]
    log_path = ROOT / "logs" / f"{args.group}_launcher.log"
    state_path = ROOT / "outputs" / f"{args.group}_queue_state.json"
    log_line(log_path, f"queue_start group={args.group} candidates={candidates} allow_gmx_share={args.allow_gmx_share}")
    for exp in EXPERIMENTS:
        check_required_inputs(exp)

    results = []
    for exp in EXPERIMENTS:
        gpu = None
        while gpu is None:
            states = query_gpus(log_path)
            log_line(
                log_path,
                "gpu_status="
                + json.dumps(
                    [
                        {
                            "index": state.index,
                            "memory_used_mib": state.memory_used_mib,
                            "utilization_pct": state.utilization_pct,
                            "compute_apps": [asdict(app) for app in state.compute_apps],
                        }
                        for state in states
                    ],
                    sort_keys=True,
                ),
            )
            gpu = choose_gpu(states, candidates, args.allow_gmx_share)
            if gpu is None:
                time.sleep(args.poll_sec)

        code = run_experiment(exp, gpu, args.group, log_path)
        results.append({"experiment": exp.name, "returncode": code})
        state_path.write_text(json.dumps({"group": args.group, "results": results}, indent=2) + "\n")
    log_line(log_path, f"queue_done group={args.group}")


if __name__ == "__main__":
    main()
