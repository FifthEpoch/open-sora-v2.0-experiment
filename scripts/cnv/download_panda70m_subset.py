#!/usr/bin/env python3
"""
Download a small subset of Panda-70M videos and build metadata.csv.

Expected input metadata file (csv or jsonl) with at least:
  - url
  - caption (or text)
Optional columns:
  - duration (seconds)
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path
from typing import Iterable

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta-path", type=str, required=True, help="Metadata CSV/JSONL path")
    parser.add_argument("--out-dir", type=str, required=True, help="Output dataset directory")
    parser.add_argument("--num-videos", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-duration", type=int, default=4)
    parser.add_argument("--max-duration", type=int, default=60)
    return parser.parse_args()


def _open_text(path: Path):
    with open(path, "rb") as f:
        head = f.read(2)
    if head == b"\x1f\x8b":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "rt", encoding="utf-8", errors="replace")


def read_rows(path: Path) -> list[dict]:
    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".gz"}:
        rows = []
        with _open_text(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows
    if suffix == ".csv":
        with open(path, newline="") as f:
            return list(csv.DictReader(f))
    raise ValueError(f"Unsupported meta file: {path}")


def download_file(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        return
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    videos_dir = out_dir / "videos"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(Path(args.meta_path))
    # Simple filtering on duration if present
    filtered = []
    for r in rows:
        duration = r.get("duration")
        if duration is not None:
            try:
                duration = float(duration)
                if duration < args.min_duration or duration > args.max_duration:
                    continue
            except Exception:
                pass
        filtered.append(r)

    import random

    random.seed(args.seed)
    random.shuffle(filtered)
    subset = filtered[: args.num_videos]

    out_rows = []
    for i, item in enumerate(subset):
        url = item.get("url") or item.get("video") or item.get("video_url")
        caption = item.get("caption") or item.get("text") or "panda video"
        if not url:
            continue
        filename = f"panda_{i:04d}.mp4"
        video_path = videos_dir / filename
        download_file(url, video_path)
        out_rows.append(
            {
                "path": str(video_path),
                "caption": caption,
                "class": "panda",
                "video_name": video_path.stem,
            }
        )

    metadata_path = out_dir / "metadata.csv"
    with open(metadata_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "caption", "class", "video_name"])
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"Wrote metadata: {metadata_path}")


if __name__ == "__main__":
    main()
