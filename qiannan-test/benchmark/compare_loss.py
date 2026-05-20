#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_loss(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if obj.get("type") == "loss" and obj.get("success", False):
                rows[str(obj["id"])] = obj
    return rows


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = int((len(ordered) - 1) * pct / 100)
    return ordered[idx]


def bootstrap_ci(values: list[float], seed: int, samples: int) -> list[float | None]:
    if not values:
        return [None, None]
    rng = random.Random(seed)
    means = [
        sum(rng.choice(values) for _ in values) / len(values)
        for _ in range(samples)
    ]
    return [percentile(means, 2.5), percentile(means, 97.5)]


def compare(
    base_rows: dict[str, dict[str, Any]],
    candidate_rows: dict[str, dict[str, Any]],
    seed: int,
    bootstrap_samples: int,
) -> dict[str, Any]:
    ids = sorted(base_rows.keys() & candidate_rows.keys())
    pairs = []
    for row_id in ids:
        base = base_rows[row_id]
        candidate = candidate_rows[row_id]
        pairs.append(
            {
                "id": row_id,
                "domain": base.get("domain", candidate.get("domain", "unknown")),
                "nll_base": float(base["nll"]),
                "nll_candidate": float(candidate["nll"]),
                "delta_nll": float(candidate["nll"]) - float(base["nll"]),
                "n_tokens": min(int(base["n_tokens"]), int(candidate["n_tokens"])),
            }
        )

    def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
        if not rows:
            return {
                "n": 0,
                "mean_nll_base": None,
                "mean_nll_candidate": None,
                "delta_nll": None,
                "delta_95ci": [None, None],
            }
        deltas = [float(row["delta_nll"]) for row in rows]
        return {
            "n": len(rows),
            "mean_nll_base": statistics.fmean(float(row["nll_base"]) for row in rows),
            "mean_nll_candidate": statistics.fmean(float(row["nll_candidate"]) for row in rows),
            "delta_nll": statistics.fmean(deltas),
            "delta_95ci": bootstrap_ci(deltas, seed, bootstrap_samples),
        }

    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        by_domain[str(pair["domain"])].append(pair)

    return {
        "type": "loss_comparison",
        "matched": len(pairs),
        "base_rows": len(base_rows),
        "candidate_rows": len(candidate_rows),
        "overall": aggregate(pairs),
        "domains": {
            domain: aggregate(rows)
            for domain, rows in sorted(by_domain.items())
        },
    }


def write_markdown(path: Path, result: dict[str, Any], base_label: str, candidate_label: str) -> None:
    def fmt(value: float | None) -> str:
        return "nan" if value is None else f"{value:.6f}"

    lines = [
        "# Loss Comparison",
        "",
        f"- Base: `{base_label}`",
        f"- Candidate: `{candidate_label}`",
        f"- Matched rows: `{result['matched']}`",
        "",
        "| domain | n | nll_base | nll_candidate | delta_nll | 95% CI |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for domain, row in {"overall": result["overall"], **result["domains"]}.items():
        ci = row["delta_95ci"]
        lines.append(
            f"| {domain} | {row['n']} | "
            f"{fmt(row['mean_nll_base'])} | {fmt(row['mean_nll_candidate'])} | "
            f"{fmt(row['delta_nll'])} | [{fmt(ci[0])}, {fmt(ci[1])}] |"
        )
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two run_loss.py outputs.")
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument("--base-label", default="base")
    parser.add_argument("--candidate-label", default="candidate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = compare(
        read_loss(args.base),
        read_loss(args.candidate),
        args.seed,
        args.bootstrap_samples,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
            f.write("\n")
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        write_markdown(args.markdown_output, result, args.base_label, args.candidate_label)
    if not args.output and not args.markdown_output:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
