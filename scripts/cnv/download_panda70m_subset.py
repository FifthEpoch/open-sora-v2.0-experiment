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
import zipfile
from pathlib import Path
from typing import Iterable

import requests
import subprocess
import shutil


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta-path", type=str, required=True, help="Metadata CSV/JSONL path")
    parser.add_argument("--out-dir", type=str, required=True, help="Output dataset directory")
    parser.add_argument("--num-videos", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-duration", type=int, default=4)
    parser.add_argument("--max-duration", type=int, default=60)
    parser.add_argument("--min-bytes", type=int, default=1_000_000)
    parser.add_argument("--min-frames", type=int, default=26)
    parser.add_argument("--resize-height", type=int, default=None)
    parser.add_argument("--resize-width", type=int, default=None)
    return parser.parse_args()


def _file_magic(path: Path) -> bytes:
    with open(path, "rb") as f:
        return f.read(4)


def _open_text(path: Path):
    head = _file_magic(path)
    if head == b"\x1f\x8b":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    if head == b"PAR1":
        raise ValueError(f"Parquet file detected (unsupported): {path}")
    return open(path, "rt", encoding="utf-8", errors="replace")


def read_rows(path: Path) -> list[dict]:
    head = _file_magic(path)
    if head == b"PK\x03\x04":
        with zipfile.ZipFile(path) as zf:
            names = [n for n in zf.namelist() if n.endswith(".csv")]
            if not names:
                raise ValueError(f"No CSV found in zip: {path}")
            name = names[0]
            with zf.open(name) as f:
                text = (line.decode("utf-8", errors="replace") for line in f)
                reader = csv.DictReader(text)
                return list(reader)
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
    if suffix in {".csv", ".tsv"}:
        with open(path, newline="") as f:
            return list(csv.DictReader(f))
    raise ValueError(f"Unsupported meta file: {path}")


def _is_youtube_url(url: str) -> bool:
    return "youtube.com" in url or "youtu.be" in url


def _validate_video(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return True
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=nw=1:nk=1",
        str(path),
    ]
    return subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def _count_frames(path: Path) -> int | None:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None
    # Try to read nb_frames directly.
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=nb_frames",
        "-of",
        "default=nw=1:nk=1",
        str(path),
    ]
    result = subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    if result.returncode == 0:
        value = result.stdout.strip()
        if value.isdigit():
            return int(value)
    # Fallback: estimate using duration * fps.
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=r_frame_rate:format=duration",
        "-of",
        "default=nw=1:nk=1",
        str(path),
    ]
    result = subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    if result.returncode != 0:
        return None
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(lines) < 2:
        return None
    try:
        fps_num, fps_den = lines[0].split("/")
        fps = float(fps_num) / float(fps_den)
        duration = float(lines[1])
        return int(round(fps * duration))
    except Exception:
        return None


def _resize_video(path: Path, height: int, width: int) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        print("WARNING: ffmpeg not found; skipping resize.")
        return True
    tmp_path = path.with_suffix(".tmp.mp4")
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(path),
        "-vf",
        f"scale={width}:{height}",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-movflags",
        "+faststart",
        str(tmp_path),
    ]
    result = subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if result.returncode != 0 or not tmp_path.exists():
        if tmp_path.exists():
            tmp_path.unlink()
        return False
    tmp_path.replace(path)
    return True


def download_file(url: str, out_path: Path, min_bytes: int) -> bool:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size >= min_bytes:
        return _validate_video(out_path)
    if _is_youtube_url(url):
        cmd = [
            "yt-dlp",
            "-f",
            "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/bv*+ba/b",
            "--merge-output-format",
            "mp4",
            "--merge-output-format",
            "mp4",
            "--no-part",
            "--quiet",
            "--no-playlist",
            "--js-runtimes",
            "node",
            "--extractor-args",
            "youtube:player_client=android",
            "-o",
            str(out_path),
            url,
        ]
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            return False
        return out_path.exists() and out_path.stat().st_size >= min_bytes and _validate_video(out_path)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        content_type = r.headers.get("content-type", "")
        if "text/html" in content_type:
            return False
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    if out_path.exists() and out_path.stat().st_size >= min_bytes and _validate_video(out_path):
        return True
    if out_path.exists():
        out_path.unlink()
    return False


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
    out_rows = []
    failures = 0
    for item in filtered:
        if len(out_rows) >= args.num_videos:
            break
        url = item.get("url") or item.get("video") or item.get("video_url")
        caption = item.get("caption") or item.get("text") or "panda video"
        if not url:
            continue
        filename = f"panda_{len(out_rows):04d}.mp4"
        video_path = videos_dir / filename
        ok = download_file(url, video_path, args.min_bytes)
        if not ok:
            failures += 1
            continue
        if args.min_frames:
            frame_count = _count_frames(video_path)
            if frame_count is None or frame_count < args.min_frames:
                failures += 1
                if video_path.exists():
                    video_path.unlink()
                continue
        if args.resize_height is not None and args.resize_width is not None:
            resized = _resize_video(video_path, args.resize_height, args.resize_width)
            if not resized:
                failures += 1
                if video_path.exists():
                    video_path.unlink()
                continue
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
    if len(out_rows) < args.num_videos:
        print(
            f"WARNING: Only downloaded {len(out_rows)} / {args.num_videos} videos "
            f"(failures: {failures}). Consider lowering --min-bytes or using yt-dlp."
        )


if __name__ == "__main__":
    main()
