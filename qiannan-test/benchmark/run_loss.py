#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import statistics
import time
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if obj.get("type") == "loss" and obj.get("success", True):
                done.add(str(obj["id"]))
    return done


def post_json(
    url: str,
    payload: dict[str, Any],
    timeout_s: float,
    retries: int,
    retry_base_delay_s: float,
) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    last_error: Exception | None = None
    for attempt in range(retries):
        request = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                body = response.read().decode("utf-8")
            obj = json.loads(body)
            if isinstance(obj, list):
                if len(obj) != 1:
                    raise RuntimeError(f"expected one response, got {len(obj)}")
                obj = obj[0]
            if not isinstance(obj, dict):
                raise RuntimeError(f"unexpected response type: {type(obj).__name__}")
            return obj
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            last_error = RuntimeError(f"HTTP {exc.code}: {body[:500]}")
        except Exception as exc:  # noqa: BLE001 - preserve original request error text.
            last_error = exc

        if attempt < retries - 1:
            time.sleep(retry_base_delay_s * (2**attempt))

    assert last_error is not None
    raise last_error


def extract_logprob(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        value = value[0]
    elif isinstance(value, dict):
        value = value.get("logprob")
    if value is None:
        return None
    return float(value)


def score_one(
    row: dict[str, Any],
    base_url: str,
    label: str,
    timeout_s: float,
    retries: int,
    retry_base_delay_s: float,
) -> dict[str, Any]:
    prompt_ids = row.get("prompt_ids")
    gen_ids = row.get("gen_ids")
    if not isinstance(prompt_ids, list) or not isinstance(gen_ids, list):
        raise ValueError(f"{row.get('id')}: expected prompt_ids and gen_ids lists")
    if not gen_ids:
        raise ValueError(f"{row.get('id')}: empty gen_ids")

    input_ids = prompt_ids + gen_ids
    payload = {
        "input_ids": input_ids,
        "sampling_params": {"temperature": 0, "max_new_tokens": 0},
        "return_logprob": True,
        "return_text_in_logprobs": False,
        "logprob_start_len": len(prompt_ids),
    }

    started = time.monotonic()
    response = post_json(
        base_url.rstrip("/") + "/generate",
        payload,
        timeout_s=timeout_s,
        retries=retries,
        retry_base_delay_s=retry_base_delay_s,
    )
    latency_s = time.monotonic() - started

    meta = response.get("meta_info") or {}
    raw_logprobs = meta.get("input_token_logprobs")
    if not isinstance(raw_logprobs, list):
        raise RuntimeError(f"{row.get('id')}: missing input_token_logprobs")

    continuation_logprobs = [
        lp
        for lp in (extract_logprob(x) for x in raw_logprobs[-len(gen_ids) :])
        if lp is not None
    ]
    if not continuation_logprobs:
        raise RuntimeError(f"{row.get('id')}: no continuation logprobs")

    nll = -sum(continuation_logprobs) / len(continuation_logprobs)
    return {
        "type": "loss",
        "success": True,
        "id": row["id"],
        "domain": row.get("domain", "unknown"),
        "label": label,
        "nll": nll,
        f"nll_{label}": nll,
        "n_tokens": len(continuation_logprobs),
        "prompt_tokens": len(prompt_ids),
        "gen_tokens": len(gen_ids),
        "total_tokens": len(input_ids),
        "latency_s": latency_s,
    }


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


def summarize(records: list[dict[str, Any]], label: str, seed: int, bootstrap_samples: int) -> dict[str, Any]:
    successes = [
        r for r in records
        if r.get("type") == "loss" and r.get("success", False)
    ]
    failures = [
        r for r in records
        if r.get("type") == "loss_error" or not r.get("success", True)
    ]

    def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
        nlls = [float(r["nll"]) for r in rows]
        token_total = sum(int(r["n_tokens"]) for r in rows)
        weighted = (
            sum(float(r["nll"]) * int(r["n_tokens"]) for r in rows) / token_total
            if token_total
            else None
        )
        latencies = [float(r["latency_s"]) for r in rows]
        return {
            "n": len(rows),
            "mean_nll": statistics.fmean(nlls) if nlls else None,
            "token_weighted_nll": weighted,
            "nll_95ci": bootstrap_ci(nlls, seed, bootstrap_samples),
            "tokens": token_total,
            "mean_latency_s": statistics.fmean(latencies) if latencies else None,
            "p95_latency_s": percentile(latencies, 95),
        }

    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in successes:
        by_domain[str(row.get("domain", "unknown"))].append(row)

    return {
        "type": "loss_summary",
        "label": label,
        "n": len(successes),
        "failed": len(failures),
        "overall": aggregate(successes),
        "domains": {
            domain: aggregate(rows)
            for domain, rows in sorted(by_domain.items())
        },
    }


def select_rows(
    rows: list[dict[str, Any]],
    domains: set[str] | None,
    per_domain: int | None,
    limit: int | None,
) -> list[dict[str, Any]]:
    if domains is not None:
        rows = [row for row in rows if row.get("domain") in domains]
    if per_domain is not None:
        selected: list[dict[str, Any]] = []
        counts: dict[str, int] = defaultdict(int)
        for row in rows:
            domain = str(row.get("domain", "unknown"))
            if counts[domain] >= per_domain:
                continue
            selected.append(row)
            counts[domain] += 1
        rows = selected
    if limit is not None:
        rows = rows[:limit]
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score teacher-forced NLL with SGLang /generate logprobs.")
    parser.add_argument("--data", required=True, type=Path, help="JSONL with id/domain/prompt_ids/gen_ids rows.")
    parser.add_argument("--base-url", required=True, help="Direct SGLang server URL.")
    parser.add_argument("--output", required=True, type=Path, help="Per-row JSONL output path.")
    parser.add_argument("--summary-output", required=True, type=Path, help="Summary JSON output path.")
    parser.add_argument("--label", default="model", help="Model label used in output fields.")
    parser.add_argument("--domains", help="Comma-separated domain filter.")
    parser.add_argument("--per-domain", type=int, help="Take at most this many rows per domain.")
    parser.add_argument("--limit", type=int, help="Take at most this many selected rows.")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-base-delay", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--allow-errors", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = read_jsonl(args.data)
    domains = {x.strip() for x in args.domains.split(",") if x.strip()} if args.domains else None
    rows = select_rows(rows, domains, args.per_domain, args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)

    done_ids = read_done_ids(args.output)
    pending = [row for row in rows if str(row.get("id")) not in done_ids]
    print(
        f"[loss] data={args.data} selected={len(rows)} done={len(done_ids)} pending={len(pending)} "
        f"label={args.label} concurrency={args.concurrency}",
        flush=True,
    )

    records: list[dict[str, Any]] = []
    if args.output.exists():
        records.extend(read_jsonl(args.output))

    completed = 0
    with args.output.open("a") as output_file:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {
                pool.submit(
                    score_one,
                    row,
                    args.base_url,
                    args.label,
                    args.timeout,
                    args.retries,
                    args.retry_base_delay,
                ): row
                for row in pending
            }
            for future in as_completed(futures):
                row = futures[future]
                try:
                    record = future.result()
                except Exception as exc:  # noqa: BLE001 - preserve per-row failure.
                    record = {
                        "type": "loss_error",
                        "success": False,
                        "id": row.get("id"),
                        "domain": row.get("domain", "unknown"),
                        "label": args.label,
                        "error": str(exc)[:1000],
                    }
                records.append(record)
                output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                output_file.flush()
                completed += 1
                if completed == 1 or completed % 10 == 0 or completed == len(pending):
                    print(f"[loss] completed {completed}/{len(pending)}", flush=True)

    summary = summarize(records, args.label, args.seed, args.bootstrap_samples)
    with args.summary_output.open("w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"[loss] summary: {args.summary_output}", flush=True)

    if summary["failed"] and not args.allow_errors:
        print(f"[loss] failed rows: {summary['failed']}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
