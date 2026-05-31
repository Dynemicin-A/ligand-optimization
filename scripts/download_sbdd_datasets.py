#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import subprocess
import sys
import time
import urllib.request
from urllib.parse import quote
from pathlib import Path
from typing import Any


DATASETS: dict[str, dict[str, Any]] = {
    "pdbbind_v2020": {
        "kind": "zenodo",
        "record_id": "7014096",
        "dirname": "PDBbind_v2020",
        "source": "https://zenodo.org/records/7014096",
    },
    "bindingmoad_prepared": {
        "kind": "zenodo",
        "record_id": "11191555",
        "dirname": "BindingMOAD_prepared",
        "source": "https://zenodo.org/records/11191555",
    },
    "crossdocked2020": {
        "kind": "hf",
        "repo_id": "kohbanye/crossdocked2020",
        "dirname": "CrossDocked2020",
        "source": "https://huggingface.co/datasets/kohbanye/crossdocked2020",
        "include": ["manifest.parquet", "receptors/*", "ligands/*"],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download public SBDD complex datasets with resumable retries."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/home/zhangxuanhao/zxh/datasets"),
        help="Dataset root. Each source is written into a named subdirectory.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["all"],
        choices=["all", *DATASETS.keys()],
        help="Datasets to download.",
    )
    parser.add_argument("--max-attempts", type=int, default=50)
    parser.add_argument("--retry-sleep", type=float, default=60.0)
    parser.add_argument("--wget-tries", type=int, default=20)
    parser.add_argument("--wget-timeout", type=int, default=60)
    parser.add_argument(
        "--hf-endpoint",
        default=os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"),
        help="HF endpoint for CrossDocked. Use a reachable mirror on restricted networks.",
    )
    parser.add_argument("--hf-command", default=os.environ.get("HF_COMMAND", "hf"))
    parser.add_argument("--force", action="store_true", help="Ignore download.done and verify/resume files.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def selected_datasets(names: list[str]) -> list[str]:
    if "all" in names:
        return list(DATASETS)
    seen: set[str] = set()
    ordered: list[str] = []
    for name in names:
        if name not in seen:
            seen.add(name)
            ordered.append(name)
    return ordered


def log(message: str) -> None:
    print(f"[{time.ctime()}] {message}", flush=True)


def fetch_json(url: str, attempts: int, retry_sleep: float) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            log(f"fetch metadata attempt {attempt}/{attempts}: {url}")
            with urllib.request.urlopen(url, timeout=120) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001 - network retries should catch all transport errors.
            last_error = exc
            log(f"metadata failed: {type(exc).__name__}: {exc}")
            if attempt < attempts:
                time.sleep(retry_sleep)
    raise RuntimeError(f"failed to fetch metadata from {url}") from last_error


def expected_size(file_entry: dict[str, Any]) -> int | None:
    raw = file_entry.get("size")
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def zenodo_file_url(record_id: str, file_entry: dict[str, Any]) -> tuple[str, str]:
    key = file_entry.get("key") or file_entry.get("filename")
    if not key:
        raise KeyError(f"Zenodo file entry has no key: {file_entry}")
    links = file_entry.get("links") or {}
    url = links.get("self") or links.get("download")
    if not url:
        url = f"https://zenodo.org/api/records/{record_id}/files/{key}/content"
    return Path(str(key)).name, str(url)


def is_complete(path: Path, size: int | None) -> bool:
    return path.exists() and size is not None and path.stat().st_size == size


def run_wget(
    url: str,
    target: Path,
    wget_tries: int,
    wget_timeout: int,
    dry_run: bool,
) -> int:
    cmd = [
        "wget",
        "-c",
        f"--tries={wget_tries}",
        f"--timeout={wget_timeout}",
        "--waitretry=10",
        "-O",
        str(target),
        url,
    ]
    log("run: " + " ".join(cmd))
    if dry_run:
        return 0
    return subprocess.run(cmd).returncode


def download_with_retries(
    name: str,
    filename: str,
    url: str,
    target: Path,
    size: int | None,
    args: argparse.Namespace,
) -> None:
    if is_complete(target, size):
        log(f"{name}: skip complete {filename} ({size} bytes)")
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, args.max_attempts + 1):
        existing = target.stat().st_size if target.exists() else 0
        log(f"{name}: download {filename} attempt {attempt}/{args.max_attempts}; existing={existing}")
        rc = run_wget(url, target, args.wget_tries, args.wget_timeout, args.dry_run)
        if args.dry_run and rc == 0:
            log(f"{name}: dry-run complete {filename}")
            return
        if rc == 0 and (size is None or target.stat().st_size == size):
            log(f"{name}: complete {filename}")
            return
        got = target.stat().st_size if target.exists() else 0
        log(f"{name}: wget rc={rc}; got={got}; expected={size}; retrying after {args.retry_sleep:g}s")
        if attempt == args.max_attempts:
            raise RuntimeError(f"{name}: failed to download {filename}")
        time.sleep(args.retry_sleep)


def download_zenodo(name: str, cfg: dict[str, Any], args: argparse.Namespace) -> None:
    outdir = args.root / str(cfg["dirname"])
    archives = outdir / "archives"
    archives.mkdir(parents=True, exist_ok=True)
    done = outdir / "download.done"
    if done.exists() and not args.force:
        log(f"{name}: skip because {done} exists")
        return

    record_id = str(cfg["record_id"])
    meta = fetch_json(f"https://zenodo.org/api/records/{record_id}", args.max_attempts, args.retry_sleep)
    (outdir / f"zenodo_{record_id}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    files = meta.get("files") or []
    if not files:
        raise RuntimeError(f"{name}: no files found in Zenodo record {record_id}")

    for file_entry in files:
        filename, url = zenodo_file_url(record_id, file_entry)
        target = archives / filename
        size = expected_size(file_entry)
        if is_complete(target, size):
            log(f"{name}: skip complete {filename} ({size} bytes)")
            continue

        download_with_retries(name, filename, url, target, size, args)

    manifest = outdir / "archive_manifest.txt"
    with manifest.open("w", encoding="utf-8") as handle:
        handle.write("filename\tsize_bytes\n")
        for file_entry in files:
            filename, _ = zenodo_file_url(record_id, file_entry)
            size = expected_size(file_entry)
            handle.write(f"{filename}\t{'' if size is None else size}\n")
    if not args.dry_run:
        done.write_text(time.ctime() + "\n", encoding="utf-8")
    log(f"{name}: finished -> {outdir}")


def download_hf(name: str, cfg: dict[str, Any], args: argparse.Namespace) -> None:
    outdir = args.root / str(cfg["dirname"])
    raw = outdir / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    done = outdir / "download.done"
    if done.exists() and not args.force:
        log(f"{name}: skip because {done} exists")
        return

    repo_id = str(cfg["repo_id"])
    endpoint = args.hf_endpoint.rstrip("/")
    tree_url = f"{endpoint}/api/datasets/{repo_id}/tree/main?recursive=1&expand=1"
    manifest = outdir / "hf_manifest.tsv"
    metadata_attempts = 1 if manifest.exists() else args.max_attempts
    try:
        tree = fetch_json(tree_url, metadata_attempts, args.retry_sleep)
    except RuntimeError:
        if not manifest.exists():
            raise
        log(f"{name}: metadata unavailable; using existing {manifest}")
        tree = []
        with manifest.open(encoding="utf-8") as handle:
            next(handle, None)
            for line in handle:
                path, _, raw_size = line.rstrip("\n").partition("\t")
                if not path:
                    continue
                tree.append({"type": "file", "path": path, "size": int(raw_size) if raw_size else None})
    include = [str(pattern) for pattern in cfg.get("include", [])]
    files = [
        item
        for item in tree
        if item.get("type") == "file"
        and any(fnmatch.fnmatch(str(item.get("path", "")), pattern) for pattern in include)
    ]
    files.sort(key=lambda item: str(item.get("path", "")))
    if not files:
        raise RuntimeError(f"{name}: no HF files matched include patterns {include}")

    with manifest.open("w", encoding="utf-8") as handle:
        handle.write("path\tsize_bytes\n")
        for item in files:
            path = str(item["path"])
            size = expected_size(item)
            handle.write(f"{path}\t{'' if size is None else size}\n")

    for item in files:
        path = str(item["path"])
        size = expected_size(item)
        quoted_path = "/".join(quote(part) for part in path.split("/"))
        url = f"{endpoint}/datasets/{repo_id}/resolve/main/{quoted_path}"
        download_with_retries(name, path, url, raw / path, size, args)

    if not args.dry_run:
        incomplete: list[str] = []
        for item in files:
            path = str(item["path"])
            size = expected_size(item)
            if not is_complete(raw / path, size):
                incomplete.append(path)
        if incomplete:
            raise RuntimeError(f"{name}: incomplete files after download: {incomplete[:10]}")

    if not args.dry_run:
        done.write_text(time.ctime() + "\n", encoding="utf-8")
    log(f"{name}: finished -> {outdir}")


def main() -> None:
    args = parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    for name in selected_datasets(args.datasets):
        cfg = DATASETS[name]
        log(f"dataset={name} source={cfg['source']}")
        if cfg["kind"] == "zenodo":
            download_zenodo(name, cfg, args)
        elif cfg["kind"] == "hf":
            download_hf(name, cfg, args)
        else:
            raise ValueError(f"unknown dataset kind: {cfg['kind']}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
