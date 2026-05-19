#!/usr/bin/env python3
"""Analyze offered-load QPS from a workload JSONL file.

This follows the request scheduling shape in run_workload.py:
all sessions start concurrently, each session runs rounds sequentially, and
each round sleeps its `wait` before issuing the request. Service time is not
known from the dataset, so the reported QPS is the ideal offered arrival rate,
not measured server throughput.
"""

import argparse
import bisect
import json
import math
from collections import Counter
from pathlib import Path


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = int(len(values) * p / 100)
    return values[min(idx, len(values) - 1)]


def load_arrivals(path: Path) -> tuple[list[float], dict]:
    arrivals: list[float] = []
    session_durations: list[float] = []
    rounds_per_session: list[int] = []
    waits: list[float] = []
    total_input = 0
    total_output = 0
    sessions = 0

    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            sessions += 1
            t = 0.0
            rounds = obj.get("rounds") or []
            rounds_per_session.append(len(rounds))
            for rd in rounds:
                wait = float(rd.get("wait", 0.0) or 0.0)
                waits.append(wait)
                t += wait
                arrivals.append(t)
                total_input += int(rd.get("input", 0) or 0)
                total_output += int(rd.get("output", 0) or 0)
            session_durations.append(t)

    arrivals.sort()
    stats = {
        "sessions": sessions,
        "rounds": len(arrivals),
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "rounds_per_session": rounds_per_session,
        "session_durations": session_durations,
        "waits": waits,
    }
    return arrivals, stats


def max_window_qps(arrivals: list[float], window: float) -> tuple[float, int, float]:
    if not arrivals or window <= 0:
        return 0.0, 0, 0.0
    best_count = 0
    best_start = arrivals[0]
    for i, start in enumerate(arrivals):
        end = start + window
        j = bisect.bisect_right(arrivals, end)
        count = j - i
        if count > best_count:
            best_count = count
            best_start = start
    return best_count / window, best_count, best_start


def bucket_qps(arrivals: list[float], bucket_seconds: float) -> list[int]:
    if not arrivals or bucket_seconds <= 0:
        return []
    buckets = Counter(int(t // bucket_seconds) for t in arrivals)
    last = int(math.ceil(max(arrivals) / bucket_seconds))
    return [buckets.get(i, 0) for i in range(last + 1)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute dataset offered-load QPS")
    parser.add_argument("data", type=Path, help="workload JSONL path")
    parser.add_argument(
        "--windows",
        default="1,5,10,30,60",
        help="comma-separated peak-QPS windows in seconds",
    )
    parser.add_argument("--bucket-seconds", type=float, default=1.0)
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    args = parser.parse_args()

    arrivals, stats = load_arrivals(args.data)
    if not arrivals:
        raise SystemExit("no requests found")

    duration_from_zero = max(arrivals)
    active_duration = max(arrivals) - min(arrivals)
    avg_qps_from_zero = len(arrivals) / duration_from_zero if duration_from_zero > 0 else 0.0
    avg_qps_active = len(arrivals) / active_duration if active_duration > 0 else 0.0

    windows = [float(x) for x in args.windows.split(",") if x.strip()]
    peak_windows = []
    for window in windows:
        qps, count, start = max_window_qps(arrivals, window)
        peak_windows.append({
            "window_seconds": window,
            "qps": qps,
            "count": count,
            "start": start,
            "end": start + window,
        })

    buckets = bucket_qps(arrivals, args.bucket_seconds)
    bucket_qps_values = [x / args.bucket_seconds for x in buckets]

    summary = {
        "data": str(args.data),
        "sessions": stats["sessions"],
        "rounds": stats["rounds"],
        "total_input_tokens": stats["total_input_tokens"],
        "total_output_tokens": stats["total_output_tokens"],
        "first_arrival_s": min(arrivals),
        "last_arrival_s": max(arrivals),
        "duration_from_zero_s": duration_from_zero,
        "active_duration_s": active_duration,
        "avg_qps_from_zero": avg_qps_from_zero,
        "avg_qps_active_window": avg_qps_active,
        "peak_windows": peak_windows,
        "per_bucket_qps": {
            "bucket_seconds": args.bucket_seconds,
            "avg": sum(bucket_qps_values) / len(bucket_qps_values) if bucket_qps_values else 0.0,
            "p50": percentile(bucket_qps_values, 50),
            "p90": percentile(bucket_qps_values, 90),
            "p99": percentile(bucket_qps_values, 99),
            "max": max(bucket_qps_values) if bucket_qps_values else 0.0,
            "non_empty_buckets": sum(1 for x in buckets if x > 0),
            "total_buckets": len(buckets),
        },
        "rounds_per_session": {
            "min": min(stats["rounds_per_session"]),
            "avg": sum(stats["rounds_per_session"]) / len(stats["rounds_per_session"]),
            "max": max(stats["rounds_per_session"]),
        },
        "session_duration_s": {
            "min": min(stats["session_durations"]),
            "avg": sum(stats["session_durations"]) / len(stats["session_durations"]),
            "p50": percentile(stats["session_durations"], 50),
            "p90": percentile(stats["session_durations"], 90),
            "p99": percentile(stats["session_durations"], 99),
            "max": max(stats["session_durations"]),
        },
        "wait_s": {
            "min": min(stats["waits"]),
            "avg": sum(stats["waits"]) / len(stats["waits"]),
            "p50": percentile(stats["waits"], 50),
            "p90": percentile(stats["waits"], 90),
            "p99": percentile(stats["waits"], 99),
            "max": max(stats["waits"]),
        },
    }

    if args.json:
        print(json.dumps(summary, indent=2))
        return

    print(f"Dataset: {summary['data']}")
    print(f"Sessions: {summary['sessions']:,}")
    print(f"Rounds / requests: {summary['rounds']:,}")
    print(f"Expected tokens: {summary['total_input_tokens']:,} in / {summary['total_output_tokens']:,} out")
    print()
    print("Offered arrival timeline, assuming zero service time:")
    print(f"  first_arrival:       {summary['first_arrival_s']:.3f}s")
    print(f"  last_arrival:        {summary['last_arrival_s']:.3f}s")
    print(f"  avg_qps_from_zero:   {summary['avg_qps_from_zero']:.3f}")
    print(f"  avg_qps_active:      {summary['avg_qps_active_window']:.3f}")
    print()
    print("Peak sliding-window QPS:")
    for item in summary["peak_windows"]:
        print(
            f"  {item['window_seconds']:>6.1f}s: "
            f"{item['qps']:>8.3f} qps "
            f"({item['count']} reqs, t={item['start']:.3f}-{item['end']:.3f}s)"
        )
    print()
    p = summary["per_bucket_qps"]
    print(f"Per {p['bucket_seconds']:.1f}s bucket QPS:")
    print(
        f"  avg={p['avg']:.3f} p50={p['p50']:.3f} "
        f"p90={p['p90']:.3f} p99={p['p99']:.3f} max={p['max']:.3f} "
        f"non_empty={p['non_empty_buckets']}/{p['total_buckets']}"
    )
    r = summary["rounds_per_session"]
    print(f"Rounds/session: min={r['min']} avg={r['avg']:.2f} max={r['max']}")
    s = summary["session_duration_s"]
    print(
        "Session offered duration: "
        f"min={s['min']:.1f}s avg={s['avg']:.1f}s p50={s['p50']:.1f}s "
        f"p90={s['p90']:.1f}s p99={s['p99']:.1f}s max={s['max']:.1f}s"
    )
    w = summary["wait_s"]
    print(
        "Wait: "
        f"min={w['min']:.1f}s avg={w['avg']:.1f}s p50={w['p50']:.1f}s "
        f"p90={w['p90']:.1f}s p99={w['p99']:.1f}s max={w['max']:.1f}s"
    )
    print()
    print("Note: actual benchmark QPS can be lower because run_workload.py waits for")
    print("each round's response before applying the next round's wait in that session.")


if __name__ == "__main__":
    main()
