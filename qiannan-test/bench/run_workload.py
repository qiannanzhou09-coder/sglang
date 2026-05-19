#!/usr/bin/env python3
"""
Workload test client for LLM serving systems (SGLang/vLLM OpenAI-compatible API).
Reads workload_data.jsonl, simulates multi-round sessions with concurrent requests,
and reports TTFT / throughput / latency metrics.
"""

import asyncio
import json
import time
import argparse
from dataclasses import dataclass, field
from typing import Optional, List
import aiohttp

# Pre-allocate a large dummy text pool. "A " ≈ 1 token in Qwen tokenizer.
DUMMY_POOL = "A " * 135000


@dataclass
class RoundMetric:
    session_id: int
    round_idx: int
    input_tokens: int
    expected_output_tokens: int
    actual_output_tokens: int
    prompt_tokens: int
    cached_tokens: int
    ttft: float
    total_time: float
    success: bool
    error: str = ""
    session_time: float = 0.0


def dummy_text(num_tokens: int) -> str:
    """Return text that tokenizes to approximately num_tokens tokens."""
    return DUMMY_POOL[: num_tokens * 2]


def extract_cached_tokens(usage: dict) -> int:
    """Return cached prompt tokens from OpenAI-compatible usage metadata."""
    details = usage.get("prompt_tokens_details") or {}
    if isinstance(details, dict) and "cached_tokens" in details:
        return int(details.get("cached_tokens") or 0)
    return int(usage.get("cached_tokens") or 0)


def round_metric_record(m: RoundMetric) -> dict:
    """Serialize one completed round metric as a JSONL record."""
    return {
        "type": "round",
        "session_id": m.session_id,
        "round_idx": m.round_idx,
        "input_tokens": m.input_tokens,
        "expected_output_tokens": m.expected_output_tokens,
        "actual_output_tokens": m.actual_output_tokens,
        "prompt_tokens": m.prompt_tokens,
        "cached_tokens": m.cached_tokens,
        "cache_hit_rate": round(m.cached_tokens / m.prompt_tokens, 6)
        if m.prompt_tokens > 0
        else 0,
        "ttft": round(m.ttft, 4),
        "total_time": round(m.total_time, 4),
        "session_time": round(m.session_time, 4),
        "success": m.success,
        "error": m.error or None,
    }


async def record_round_metric(
    metric: RoundMetric,
    metrics: list,
    progress: dict,
    output_file=None,
    output_lock: Optional[asyncio.Lock] = None,
) -> None:
    """Append a metric in memory and optionally flush it to JSONL immediately."""
    metrics.append(metric)
    if metric.success:
        progress["rounds"] += 1
    else:
        progress["failed"] += 1

    if output_file is None:
        return

    line = json.dumps(round_metric_record(metric)) + "\n"
    if output_lock is None:
        output_file.write(line)
        output_file.flush()
        return

    async with output_lock:
        output_file.write(line)
        output_file.flush()


async def stream_round(
    http: aiohttp.ClientSession,
    url: str,
    messages: list,
    max_tokens: int,
    ignore_eos: bool,
    temperature: float,
    model: str,
) -> tuple:
    """Send one streaming chat-completion request. Returns (content, ttft, total_time, usage)."""
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if ignore_eos:
        payload["ignore_eos"] = True

    t0 = time.monotonic()
    ttft = None
    chunks: list[str] = []
    usage = {}

    async with http.post(url, json=payload) as resp:
        if resp.status != 200:
            body = await resp.text()
            raise RuntimeError(f"HTTP {resp.status}: {body[:300]}")

        async for raw_line in resp.content:
            for line in raw_line.decode("utf-8", errors="replace").split("\n"):
                line = line.strip()
                if not line.startswith("data: "):
                    continue
                if line == "data: [DONE]":
                    break
                try:
                    obj = json.loads(line[6:])
                except json.JSONDecodeError:
                    continue

                if "usage" in obj:
                    usage = obj["usage"]

                choices = obj.get("choices", [])
                if not choices:
                    continue
                delta = choices[0].get("delta", {})
                content = delta.get("content") or ""
                reasoning_content = delta.get("reasoning_content") or ""
                if ttft is None and (content or reasoning_content):
                    ttft = time.monotonic() - t0
                if content:
                    chunks.append(content)

    total_time = time.monotonic() - t0
    return "".join(chunks), ttft or total_time, total_time, usage


async def run_session(
    session_data: dict,
    endpoint: str,
    http: aiohttp.ClientSession,
    metrics: list,
    progress: dict,
    ignore_eos: bool,
    temperature: float,
    model: str,
    output_file=None,
    output_lock: Optional[asyncio.Lock] = None,
    max_retries: int = 3,
    retry_base_delay: float = 2.0,
):
    """Run all rounds of a single session sequentially."""
    sid = session_data["id"]
    messages = [{"role": "system", "content": session_data["system_prompt"]}]
    session_exec_time = 0.0

    for ri, rd in enumerate(session_data["rounds"]):
        if rd["wait"] > 0:
            await asyncio.sleep(rd["wait"])

        messages.append({"role": "user", "content": dummy_text(rd["input"])})

        round_start = time.monotonic()
        last_error = None
        for attempt in range(max_retries):
            try:
                content, ttft, total_time, usage = await stream_round(
                    http, endpoint, messages, rd["output"], ignore_eos, temperature, model
                )
                actual_out = usage.get("completion_tokens", 0)
                prompt_tokens = usage.get("prompt_tokens") or 0
                cached_tokens = extract_cached_tokens(usage)
                messages.append({"role": "assistant", "content": content})

                session_exec_time += time.monotonic() - round_start
                await record_round_metric(
                    RoundMetric(
                        session_id=sid,
                        round_idx=ri,
                        input_tokens=rd["input"],
                        expected_output_tokens=rd["output"],
                        actual_output_tokens=actual_out,
                        prompt_tokens=prompt_tokens,
                        cached_tokens=cached_tokens,
                        ttft=ttft,
                        total_time=total_time,
                        success=True,
                        session_time=session_exec_time,
                    ),
                    metrics,
                    progress,
                    output_file,
                    output_lock,
                )
                last_error = None
                break
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    delay = retry_base_delay * (2 ** attempt)
                    print(f"  [Session {sid} Round {ri}] retry {attempt+1}/{max_retries-1} "
                          f"after {delay:.0f}s: {e}", flush=True)
                    await asyncio.sleep(delay)

        if last_error is not None:
            session_exec_time += time.monotonic() - round_start
            await record_round_metric(
                RoundMetric(
                    session_id=sid,
                    round_idx=ri,
                    input_tokens=rd["input"],
                    expected_output_tokens=rd["output"],
                    actual_output_tokens=0,
                    prompt_tokens=0,
                    cached_tokens=0,
                    ttft=0,
                    total_time=0,
                    success=False,
                    error=str(last_error)[:300],
                    session_time=session_exec_time,
                ),
                metrics,
                progress,
                output_file,
                output_lock,
            )
            break

    progress["sessions"] += 1


async def run_session_admitted(
    session_data: dict,
    endpoint: str,
    http: aiohttp.ClientSession,
    metrics: list,
    progress: dict,
    ignore_eos: bool,
    temperature: float,
    model: str,
    output_file=None,
    output_lock: Optional[asyncio.Lock] = None,
    admission_semaphore: Optional[asyncio.Semaphore] = None,
):
    """Run one session after admission, counting the whole session as active."""

    async def _run() -> None:
        progress["started_sessions"] += 1
        progress["active_sessions"] += 1
        try:
            await run_session(
                session_data,
                endpoint,
                http,
                metrics,
                progress,
                ignore_eos=ignore_eos,
                temperature=temperature,
                model=model,
                output_file=output_file,
                output_lock=output_lock,
            )
        finally:
            progress["active_sessions"] -= 1

    if admission_semaphore is None:
        await _run()
        return

    async with admission_semaphore:
        await _run()


async def progress_reporter(progress: dict, total_sessions: int, total_rounds: int):
    """Print progress every 10 seconds."""
    while progress["sessions"] < total_sessions:
        await asyncio.sleep(10)
        elapsed = time.monotonic() - progress["t0"]
        print(
            f"  [{elapsed:7.1f}s] "
            f"sessions {progress['sessions']}/{total_sessions} | "
            f"active {progress['active_sessions']} | "
            f"started {progress['started_sessions']}/{total_sessions} | "
            f"rounds {progress['rounds']}/{total_rounds} | "
            f"failed {progress['failed']}",
            flush=True,
        )


async def summary_reporter(
    metrics: List[RoundMetric],
    progress: dict,
    total_sessions: int,
    total_rounds: int,
    interval_secs: float,
    summary_file,
    output_lock: asyncio.Lock,
):
    """Append cumulative summaries every interval_secs seconds."""
    interval_idx = 0
    while progress["sessions"] < total_sessions:
        await asyncio.sleep(interval_secs)
        interval_idx += 1
        elapsed = time.monotonic() - progress["t0"]
        summary = build_summary(metrics, elapsed)
        summary.update({
            "type": "interval_summary",
            "interval_index": interval_idx,
            "interval_secs": interval_secs,
            "elapsed_secs": round(elapsed, 2),
            "loaded_sessions": total_sessions,
            "started_sessions": progress.get("started_sessions", 0),
            "active_sessions": progress.get("active_sessions", 0),
            "max_active_sessions": progress.get("max_active_sessions"),
            "completed_sessions": progress["sessions"],
            "planned_rounds": total_rounds,
            "completion_ratio": round(progress["rounds"] / total_rounds, 6)
            if total_rounds > 0
            else 0,
        })
        async with output_lock:
            summary_file.write(json.dumps(summary) + "\n")
            summary_file.flush()


def percentile(data: list, p: float) -> float:
    if not data:
        return 0.0
    data_sorted = sorted(data)
    idx = int(len(data_sorted) * p / 100)
    return data_sorted[min(idx, len(data_sorted) - 1)]


def build_summary(metrics: List[RoundMetric], wall_time: float) -> dict:
    ok = [m for m in metrics if m.success]
    fail = [m for m in metrics if not m.success]

    by_session: dict = {}
    for m in metrics:
        if m.session_time > by_session.get(m.session_id, 0.0):
            by_session[m.session_id] = m.session_time
    total_sessions = len(by_session)
    avg_session_time = (sum(by_session.values()) / total_sessions) if total_sessions else 0.0

    if not ok:
        return {
            "successful_rounds": 0,
            "failed_rounds": len(fail),
            "total_sessions": total_sessions,
            "avg_session_time": round(avg_session_time, 4),
            "error": fail[0].error if fail else "unknown",
        }

    ttfts = [m.ttft for m in ok]
    totals = [m.total_time for m in ok]
    total_input = sum(m.input_tokens for m in ok)
    total_output = sum(m.actual_output_tokens or m.expected_output_tokens for m in ok)
    total_prompt = sum(m.prompt_tokens for m in ok)
    total_cached = sum(m.cached_tokens for m in ok)
    total_uncached_prompt = max(total_prompt - total_cached, 0)
    cache_hit_rate = (total_cached / total_prompt) if total_prompt > 0 else 0.0

    return {
        "wall_time": round(wall_time, 2),
        "successful_rounds": len(ok),
        "failed_rounds": len(fail),
        "total_sessions": total_sessions,
        "avg_session_time": round(avg_session_time, 4),
        "total_input_tokens": total_input,
        "total_prompt_tokens": total_prompt,
        "total_cached_tokens": total_cached,
        "total_uncached_prompt_tokens": total_uncached_prompt,
        "cache_hit_rate": round(cache_hit_rate, 6),
        "total_output_tokens": total_output,
        "output_throughput_tok_s": round(total_output / wall_time, 1) if wall_time > 0 else 0,
        "request_throughput_req_s": round(len(ok) / wall_time, 2) if wall_time > 0 else 0,
        "ttft": {
            "avg": round(sum(ttfts) / len(ttfts), 4),
            "p50": round(percentile(ttfts, 50), 4),
            "p90": round(percentile(ttfts, 90), 4),
            "p99": round(percentile(ttfts, 99), 4),
        },
        "round_latency": {
            "avg": round(sum(totals) / len(totals), 4),
            "p50": round(percentile(totals, 50), 4),
            "p90": round(percentile(totals, 90), 4),
            "p99": round(percentile(totals, 99), 4),
        },
    }


def print_report(metrics: List[RoundMetric], wall_time: float) -> dict:
    summary = build_summary(metrics, wall_time)
    ok = [m for m in metrics if m.success]
    fail = [m for m in metrics if not m.success]

    if not ok:
        print("\nAll requests failed!")
        if fail:
            print(f"First error: {fail[0].error}")
        return summary

    print()
    print("=" * 64)
    print("  WORKLOAD TEST REPORT")
    print("=" * 64)
    print(f"  Wall time:            {summary['wall_time']:.1f}s")
    print(f"  Successful rounds:    {summary['successful_rounds']}")
    print(f"  Failed rounds:        {summary['failed_rounds']}")
    print(f"  Total input tokens:   {summary['total_input_tokens']:>12,}")
    print(f"  Total prompt tokens:  {summary['total_prompt_tokens']:>12,}")
    print(f"  Cached prompt tokens: {summary['total_cached_tokens']:>12,}")
    print(f"  Cache hit rate:       {summary['cache_hit_rate']:>12.4%}")
    print(f"  Total output tokens:  {summary['total_output_tokens']:>12,}")
    print(f"  Output throughput:    {summary['output_throughput_tok_s']:>12,.1f} tok/s")
    print(f"  Request throughput:   {summary['request_throughput_req_s']:>12,.2f} req/s")
    print()
    t = summary["ttft"]
    print(f"  Time to First Token (TTFT):")
    print(f"    avg={t['avg']:.3f}s  p50={t['p50']:.3f}s  p90={t['p90']:.3f}s  p99={t['p99']:.3f}s")
    print()
    r = summary["round_latency"]
    print(f"  Round Latency:")
    print(f"    avg={r['avg']:.3f}s  p50={r['p50']:.3f}s  p90={r['p90']:.3f}s  p99={r['p99']:.3f}s")
    print("=" * 64)

    if fail:
        print(f"\n  Top errors:")
        from collections import Counter
        errs = Counter(m.error[:80] for m in fail)
        for err, cnt in errs.most_common(5):
            print(f"    [{cnt}x] {err}")

    return summary


async def main():
    parser = argparse.ArgumentParser(description="LLM serving workload tester")
    parser.add_argument("--data", default="workload_data.jsonl",
                        help="Path to workload JSONL file")
    parser.add_argument("--base-url", default="http://localhost:8000",
                        help="Serving endpoint base URL")
    parser.add_argument("--model", default="default",
                        help="Model name for API requests")
    parser.add_argument("--max-sessions", type=int, default=None,
                        help="Limit total sessions (for quick tests)")
    parser.add_argument("--max-active-sessions", type=int, default=None,
                        help="Limit concurrently admitted sessions; queued sessions start after an active session finishes")
    parser.add_argument("--no-ignore-eos", action="store_true",
                        help="Don't force ignore_eos (output may be shorter than requested)")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")
    parser.add_argument("--timeout", type=int, default=1800,
                        help="Per-request timeout in seconds")
    parser.add_argument("--max-duration", type=float, default=None,
                        help="Stop the client after this many seconds and write a partial summary")
    parser.add_argument("--summary-interval", type=float, default=None,
                        help="Append cumulative interval summaries every N seconds")
    parser.add_argument("--summary-output", default=None,
                        help="Path to write interval summaries JSONL")
    parser.add_argument("--output", default="workload_metrics.jsonl",
                        help="Path to write per-round metrics JSONL and final summary")
    args = parser.parse_args()
    if args.max_active_sessions is not None and args.max_active_sessions <= 0:
        parser.error("--max-active-sessions must be positive")

    # Load workload
    sessions = []
    with open(args.data) as f:
        for line in f:
            line = line.strip()
            if line:
                sessions.append(json.loads(line))
    if args.max_sessions:
        sessions = sessions[: args.max_sessions]

    total_rounds = sum(len(s["rounds"]) for s in sessions)
    total_input = sum(r["input"] for s in sessions for r in s["rounds"])
    total_output = sum(r["output"] for s in sessions for r in s["rounds"])

    print(f"Workload: {len(sessions)} sessions, {total_rounds} rounds")
    print(f"Expected tokens: {total_input:,} in / {total_output:,} out")
    print(f"Target: {args.base_url}")
    print(f"ignore_eos: {not args.no_ignore_eos}")
    if args.max_active_sessions is not None:
        print(f"Max active sessions: {args.max_active_sessions}")
    if args.max_duration is not None:
        print(f"Max duration: {args.max_duration:g}s")
    if args.summary_interval is not None:
        print(f"Summary interval: {args.summary_interval:g}s")
    print()

    all_metrics: list[RoundMetric] = []
    progress = {
        "sessions": 0,
        "started_sessions": 0,
        "active_sessions": 0,
        "max_active_sessions": args.max_active_sessions,
        "rounds": 0,
        "failed": 0,
        "t0": time.monotonic(),
    }

    connector = aiohttp.TCPConnector(limit=0, limit_per_host=0)
    timeout = aiohttp.ClientTimeout(total=args.timeout)

    out_path = args.output
    summary_out_path = args.summary_output
    output_lock = asyncio.Lock()
    summary_output_lock = asyncio.Lock()
    stop_reason = "completed"

    with open(out_path, "w") as output_file:
        summary_file = (
            open(summary_out_path, "w")
            if args.summary_interval is not None and summary_out_path is not None
            else output_file
        )
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as http:
            endpoint = f"{args.base_url}/v1/chat/completions"

            reporter = asyncio.create_task(
                progress_reporter(progress, len(sessions), total_rounds)
            )
            summary_task = (
                asyncio.create_task(
                    summary_reporter(
                        all_metrics,
                        progress,
                        len(sessions),
                        total_rounds,
                        args.summary_interval,
                        summary_file,
                        summary_output_lock
                        if summary_file is not output_file
                        else output_lock,
                    )
                )
                if args.summary_interval is not None
                else None
            )

            wall_start = time.monotonic()
            admission_semaphore = (
                asyncio.Semaphore(args.max_active_sessions)
                if args.max_active_sessions is not None
                else None
            )
            tasks = [
                asyncio.create_task(
                    run_session_admitted(
                        s, endpoint, http, all_metrics, progress,
                        ignore_eos=not args.no_ignore_eos,
                        temperature=args.temperature,
                        model=args.model,
                        output_file=output_file,
                        output_lock=output_lock,
                        admission_semaphore=admission_semaphore,
                    )
                )
                for s in sessions
            ]

            try:
                gather = asyncio.gather(*tasks)
                if args.max_duration is None:
                    await gather
                else:
                    await asyncio.wait_for(gather, timeout=args.max_duration)
            except asyncio.TimeoutError:
                stop_reason = "max_duration"
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
            finally:
                wall_time = time.monotonic() - wall_start
                reporter.cancel()
                try:
                    await reporter
                except asyncio.CancelledError:
                    pass
                if summary_task is not None:
                    summary_task.cancel()
                    try:
                        await summary_task
                    except asyncio.CancelledError:
                        pass

        if summary_file is not output_file:
            summary_file.close()

    summary = print_report(all_metrics, wall_time)
    summary.update({
        "stop_reason": stop_reason,
        "loaded_sessions": len(sessions),
        "started_sessions": progress["started_sessions"],
        "active_sessions": progress["active_sessions"],
        "max_active_sessions": args.max_active_sessions,
        "completed_sessions": progress["sessions"],
        "planned_rounds": total_rounds,
        "completion_ratio": round(progress["rounds"] / total_rounds, 6)
        if total_rounds > 0
        else 0,
    })
    if args.max_duration is not None:
        summary["max_duration"] = args.max_duration

    with open(out_path, "a") as f:
        f.write(json.dumps({"type": "summary", **summary}) + "\n")
    print(f"\nDetailed per-round metrics + summary saved to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
