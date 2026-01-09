"""
Shared timing utilities for Open-Sora experiments.

We record wall-clock timings per video (phase breakdown) and write:
- timing_per_video.jsonl
- timing_summary.json

This is intentionally lightweight and has no external deps.
"""

from __future__ import annotations

import json
import statistics
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional


def now_s() -> float:
    """High-resolution wall-clock timer in seconds."""
    return time.perf_counter()


@dataclass
class TimingRecord:
    idx: int
    video_name: str
    success: bool
    phases_s: Dict[str, float]
    total_s: float
    extra: Dict[str, Any]

    def to_dict(self) -> dict:
        out = {
            "idx": self.idx,
            "video_name": self.video_name,
            "success": self.success,
            "phases_s": dict(self.phases_s),
            "total_s": float(self.total_s),
        }
        if self.extra:
            out["extra"] = self.extra
        return out


class PhaseTimer:
    """
    Simple phase-based accumulator.

    Usage:
        phases = {}
        t = PhaseTimer(phases)
        with t.phase("encode"):
            ...
    """

    def __init__(self, phases_s: Dict[str, float]):
        self.phases_s = phases_s

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        t0 = now_s()
        try:
            yield
        finally:
            dt = now_s() - t0
            self.phases_s[name] = float(self.phases_s.get(name, 0.0) + dt)


def _safe_mean(xs: List[float]) -> Optional[float]:
    return float(statistics.mean(xs)) if xs else None


def _safe_median(xs: List[float]) -> Optional[float]:
    return float(statistics.median(xs)) if xs else None


def _safe_stdev(xs: List[float]) -> Optional[float]:
    if len(xs) < 2:
        return None
    return float(statistics.stdev(xs))


def _safe_quantile(xs: List[float], q: float) -> Optional[float]:
    if not xs:
        return None
    xs_sorted = sorted(xs)
    # Nearest-rank quantile, deterministic, no numpy dependency.
    k = int(round((len(xs_sorted) - 1) * q))
    k = max(0, min(k, len(xs_sorted) - 1))
    return float(xs_sorted[k])


def summarize(records: Iterable[dict]) -> dict:
    """
    Build run-level summary stats from per-video timing records.
    Expects dicts shaped like TimingRecord.to_dict().
    """
    items = list(records)
    ok = [it for it in items if it.get("success", False)]
    all_phases = set()
    for it in ok:
        all_phases.update((it.get("phases_s") or {}).keys())

    phase_stats: Dict[str, dict] = {}
    for p in sorted(all_phases):
        vals = [float(it["phases_s"].get(p, 0.0)) for it in ok]
        phase_stats[p] = {
            "mean_s": _safe_mean(vals),
            "median_s": _safe_median(vals),
            "stdev_s": _safe_stdev(vals),
            "p95_s": _safe_quantile(vals, 0.95),
        }

    totals = [float(it.get("total_s", 0.0)) for it in ok]
    return {
        "num_videos": len(items),
        "successful": len(ok),
        "failed": len(items) - len(ok),
        "total_s": {
            "mean_s": _safe_mean(totals),
            "median_s": _safe_median(totals),
            "stdev_s": _safe_stdev(totals),
            "p95_s": _safe_quantile(totals, 0.95),
        },
        "phases_s": phase_stats,
    }


def write_timing_files(output_dir: Path, records: List[dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    per_video_path = output_dir / "timing_per_video.jsonl"
    with open(per_video_path, "w") as f:
        for it in records:
            f.write(json.dumps(it, default=str) + "\n")

    summary_path = output_dir / "timing_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summarize(records), f, indent=2)

