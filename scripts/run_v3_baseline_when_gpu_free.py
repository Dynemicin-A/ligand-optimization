#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


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
    def has_compute_process(self) -> bool:
        return bool(self.compute_apps)

    @property
    def compute_pids(self) -> list[int]:
        return [app.pid for app in self.compute_apps]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wait for a free non-reserved GPU and launch the v3 baseline.")
    parser.add_argument("--mode", choices=["pretrain", "h2l"], default="pretrain")
    parser.add_argument("--candidate-gpus", default="1,2,3,5,6,4,0,7")
    parser.add_argument("--exclude-gpus", default="")
    parser.add_argument("--poll-sec", type=int, default=120)
    parser.add_argument("--max-wait-sec", type=int, default=0, help="0 means wait indefinitely.")
    parser.add_argument("--max-memory-used-mib", type=int, default=900)
    parser.add_argument("--max-utilization-pct", type=int, default=5)
    parser.add_argument(
        "--allow-shared-process-regex",
        default="",
        help="Allow sharing a GPU when every existing compute process name matches this regex.",
    )
    parser.add_argument(
        "--max-shared-memory-used-mib",
        type=int,
        default=1200,
        help="Maximum total GPU memory used before launch when sharing with allowed processes.",
    )
    parser.add_argument(
        "--max-shared-app-memory-mib",
        type=int,
        default=900,
        help="Maximum memory for each allowed existing process when sharing.",
    )
    parser.add_argument(
        "--max-shared-utilization-pct",
        type=int,
        default=100,
        help="Maximum GPU utilization allowed when sharing with allowed processes.",
    )
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--log-path", type=Path, default=None)
    parser.add_argument("--init-model", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=None, help="Override train.batch_size for the launched run.")
    parser.add_argument("--extra-train-arg", action="append", default=[], help="Extra argument forwarded to train_diffusion.py.")
    return parser.parse_args()


def parse_gpu_list(value: str) -> list[int]:
    return [int(part) for part in value.split(",") if part.strip()]


def run_nvidia_smi(args: list[str]) -> str:
    return subprocess.check_output(["nvidia-smi", *args], text=True, stderr=subprocess.STDOUT)


def query_gpus() -> list[GpuState]:
    gpu_csv = run_nvidia_smi(
        [
            "--query-gpu=index,uuid,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
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
        apps_csv = run_nvidia_smi(
            [
                "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ]
        )
    except subprocess.CalledProcessError:
        apps_csv = ""

    for row in csv.reader(apps_csv.splitlines()):
        if len(row) < 4:
            continue
        uuid = row[0].strip()
        if uuid in states:
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


def choose_gpu(
    states: list[GpuState],
    candidates: list[int],
    excluded: set[int],
    max_memory_used_mib: int,
    max_utilization_pct: int,
    allow_shared_process_regex: str,
    max_shared_memory_used_mib: int,
    max_shared_app_memory_mib: int,
    max_shared_utilization_pct: int,
) -> GpuState | None:
    by_index = {state.index: state for state in states}
    allowed_process = re.compile(allow_shared_process_regex) if allow_shared_process_regex else None

    # Prefer a genuinely free GPU over a shareable lightweight job, even if the
    # shareable GPU appears earlier in the candidate order.
    for gpu in candidates:
        if gpu in excluded:
            continue
        state = by_index.get(gpu)
        if state is None:
            continue
        if not state.has_compute_process:
            if state.memory_used_mib <= max_memory_used_mib and state.utilization_pct <= max_utilization_pct:
                return state

    if allowed_process is None:
        return None

    for gpu in candidates:
        if gpu in excluded:
            continue
        state = by_index.get(gpu)
        if state is None or not state.has_compute_process:
            continue
        if state.memory_used_mib > max_shared_memory_used_mib:
            continue
        if state.utilization_pct > max_shared_utilization_pct:
            continue
        if any(app.used_memory_mib > max_shared_app_memory_mib for app in state.compute_apps):
            continue
        if all(allowed_process.search(app.process_name) for app in state.compute_apps):
            return state
    return None


def default_outdir(mode: str) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    if mode == "pretrain":
        return ROOT / "outputs" / f"v3_baseline_pretrain_{stamp}_bs6"
    return ROOT / "outputs" / f"v3_baseline_h2l_{stamp}_bs6"


def build_command(args: argparse.Namespace, selected_gpu: int, outdir: Path) -> list[str]:
    command = [sys.executable, "scripts/ligopt_v3.py"]
    if args.mode == "pretrain":
        command.extend(["train-pretrain", "--", "--outdir", str(outdir), "--device", "cuda"])
    else:
        command.extend(["train-h2l", "--", "--outdir", str(outdir), "--device", "cuda"])
        if args.init_model is not None:
            command.extend(["--init-model", str(args.init_model), "--init-weights", "auto"])
    if args.batch_size is not None:
        command.extend(["--batch-size", str(args.batch_size)])
    command.extend(args.extra_train_arg)
    return command


def log_line(path: Path | None, message: str) -> None:
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} {message}"
    print(line, flush=True)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as handle:
            handle.write(line + "\n")


def main() -> None:
    args = parse_args()
    candidates = parse_gpu_list(args.candidate_gpus)
    excluded = set(parse_gpu_list(args.exclude_gpus))
    outdir = args.outdir or default_outdir(args.mode)
    log_path = args.log_path
    start = time.monotonic()

    log_line(
        log_path,
        "waiting "
        f"mode={args.mode} candidates={candidates} excluded={sorted(excluded)} "
        f"max_memory={args.max_memory_used_mib}MiB max_util={args.max_utilization_pct}% "
        f"allow_shared_process_regex={args.allow_shared_process_regex!r} "
        f"max_shared_memory={args.max_shared_memory_used_mib}MiB "
        f"max_shared_app_memory={args.max_shared_app_memory_mib}MiB "
        f"max_shared_util={args.max_shared_utilization_pct}%",
    )

    selected: GpuState | None = None
    while selected is None:
        states = query_gpus()
        status = [
            {
                "index": state.index,
                "memory_used_mib": state.memory_used_mib,
                "utilization_pct": state.utilization_pct,
                "compute_pids": state.compute_pids,
                "compute_apps": [asdict(app) for app in state.compute_apps],
            }
            for state in states
        ]
        log_line(log_path, "gpu_status=" + json.dumps(status, sort_keys=True))
        selected = choose_gpu(
            states,
            candidates=candidates,
            excluded=excluded,
            max_memory_used_mib=args.max_memory_used_mib,
            max_utilization_pct=args.max_utilization_pct,
            allow_shared_process_regex=args.allow_shared_process_regex,
            max_shared_memory_used_mib=args.max_shared_memory_used_mib,
            max_shared_app_memory_mib=args.max_shared_app_memory_mib,
            max_shared_utilization_pct=args.max_shared_utilization_pct,
        )
        if selected is not None:
            break
        if args.max_wait_sec and time.monotonic() - start >= args.max_wait_sec:
            raise SystemExit("no free eligible GPU before max wait")
        time.sleep(max(args.poll_sec, 1))

    outdir.mkdir(parents=True, exist_ok=True)
    command = build_command(args, selected.index, outdir)
    launch_info = {
        "selected_gpu": asdict(selected),
        "mode": args.mode,
        "outdir": str(outdir),
        "command": command,
        "exclude_gpus": sorted(excluded),
        "candidate_gpus": candidates,
        "launched_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (outdir / "launch_info.json").write_text(json.dumps(launch_info, indent=2, sort_keys=True) + "\n")
    log_line(log_path, "launch=" + json.dumps(launch_info, sort_keys=True))

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(selected.index)
    env["PYTHONPATH"] = "src" + os.pathsep + env.get("PYTHONPATH", "")
    raise SystemExit(subprocess.run(command, cwd=ROOT, env=env, check=False).returncode)


if __name__ == "__main__":
    main()
