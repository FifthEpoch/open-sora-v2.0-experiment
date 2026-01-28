#!/usr/bin/env python3
"""
Download a small subset of Pexels-400k videos and build metadata.csv.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, required=True, help="Output dataset directory")
    parser.add_argument("--num-videos", type=int, default=100, help="Number of videos to download")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-duration", type=int, default=60, help="Max duration in seconds")
    parser.add_argument("--min-duration", type=int, default=4, help="Min duration in seconds")
    parser.add_argument("--dataset", type=str, default="jovianzm/Pexels-400k")
    return parser.parse_args()


def extract_id(url: str) -> str:
    m = re.search(r"/video/(\d+)", url)
    return m.group(1) if m else "unknown"


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

    from datasets import load_dataset
    ds = load_dataset(args.dataset, split="train")

    # Filter by duration + SFW
    ds = ds.filter(
        lambda x: x.get("sfw", True)
        and args.min_duration <= int(x.get("duration", 0)) <= args.max_duration
    )

    ds = ds.shuffle(seed=args.seed)
    ds = ds.select(range(min(args.num_videos, len(ds))))

    rows = []
    for item in ds:
        url = item["video"]
        title = item.get("title", "pexels video")
        vid_id = extract_id(url)
        filename = f"pexels_{vid_id}.mp4"
        video_path = videos_dir / filename
        download_file(url, video_path)
        rows.append(
            {
                "path": str(video_path),
                "caption": title,
                "class": "pexels",
                "video_name": video_path.stem,
            }
        )

    metadata_path = out_dir / "metadata.csv"
    with open(metadata_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "caption", "class", "video_name"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote metadata: {metadata_path}")


if __name__ == "__main__":
    main()
