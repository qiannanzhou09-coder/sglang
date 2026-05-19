#!/usr/bin/env python3
"""
Small OpenAI-compatible proxy for SGLang serving experiments.

length_aware routes short prompts directly to SGLang so DPA can load-balance,
and routes longer prompts through sglang_router cache_aware.
"""

from __future__ import annotations

import argparse
import asyncio
import codecs
import json
import math
import time
import uuid
from pathlib import Path
from typing import Any

import aiohttp
from aiohttp import web


HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
}


def text_from_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif isinstance(item.get("content"), str):
                    parts.append(item["content"])
        return "\n".join(parts)
    return str(content)


def estimate_prompt_tokens(payload: dict[str, Any], chars_per_token: float) -> int:
    messages = payload.get("messages") or []
    total_chars = 0
    if isinstance(messages, list):
        for message in messages:
            if isinstance(message, dict):
                total_chars += len(text_from_content(message.get("content")))
                total_chars += 8
    elif isinstance(payload.get("prompt"), str):
        total_chars = len(payload["prompt"])
    return max(1, int(total_chars / chars_per_token))


def choose_target(policy: str, prompt_tokens_estimate: int, threshold: int, direct_url: str, router_url: str) -> tuple[str, str]:
    if policy == "passthrough_direct":
        return direct_url, "direct"
    if policy == "passthrough_router":
        return router_url, "router"
    if policy != "length_aware":
        raise ValueError(f"unknown policy: {policy}")
    if prompt_tokens_estimate < threshold:
        return direct_url, "direct_short"
    return router_url, "router_long"


def filter_request_headers(headers: Any) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, value in headers.items():
        if key.lower() in HOP_BY_HOP_HEADERS:
            continue
        if key.lower() == "host":
            continue
        out[key] = value
    return out


def filter_response_headers(headers: Any) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, value in headers.items():
        if key.lower() in HOP_BY_HOP_HEADERS:
            continue
        out[key] = value
    return out


def extract_cached_tokens(usage: dict[str, Any]) -> int:
    details = usage.get("prompt_tokens_details") or {}
    if isinstance(details, dict) and "cached_tokens" in details:
        return int(details.get("cached_tokens") or 0)
    return int(usage.get("cached_tokens") or 0)


def percentile(data: list[float], p: float) -> float:
    if not data:
        return 0.0
    data_sorted = sorted(data)
    idx = int(len(data_sorted) * p / 100)
    return data_sorted[min(idx, len(data_sorted) - 1)]


def round_optional(value: float | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    return round(value, digits)


def observe_sse_line(
    line: str,
    observation: dict[str, Any],
    request_started: float,
    upstream_started: float,
    now: float,
) -> None:
    line = line.strip()
    if not line.startswith("data: ") or line == "data: [DONE]":
        return

    try:
        obj = json.loads(line[6:])
    except json.JSONDecodeError:
        return

    usage = obj.get("usage")
    if isinstance(usage, dict):
        observation["prompt_tokens"] = int(
            usage.get("prompt_tokens") or observation.get("prompt_tokens") or 0
        )
        observation["cached_tokens"] = extract_cached_tokens(usage)
        observation["completion_tokens"] = int(
            usage.get("completion_tokens")
            or observation.get("completion_tokens")
            or 0
        )

    if observation.get("ttft") is not None:
        return

    choices = obj.get("choices") or []
    if not choices:
        return
    delta = choices[0].get("delta") or {}
    content = delta.get("content") or ""
    reasoning_content = delta.get("reasoning_content") or ""
    if content or reasoning_content:
        observation["ttft"] = now - request_started
        observation["upstream_ttft"] = now - upstream_started


class ProxyState:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.http: aiohttp.ClientSession | None = None
        self.admission_semaphore: asyncio.Semaphore | None = None
        self.admission_lock: asyncio.Lock | None = None
        self.admission_condition: asyncio.Condition | None = None
        self.dynamic_task: asyncio.Task | None = None
        self.completed_stats_lock: asyncio.Lock | None = None
        self.completed_request_stats: list[dict[str, Any]] = []
        self.active_requests = 0
        self.waiting_requests = 0
        self.current_inflight_limit: int | None = None
        if args.dynamic_admission:
            initial_limit = (
                args.dynamic_initial_inflight_requests
                if args.dynamic_initial_inflight_requests is not None
                else args.max_inflight_requests
                if args.max_inflight_requests is not None
                else 128
            )
            self.current_inflight_limit = min(
                max(initial_limit, args.dynamic_min_inflight_requests),
                args.dynamic_max_inflight_requests,
            )
        elif args.max_inflight_requests is not None:
            self.current_inflight_limit = args.max_inflight_requests
        self.metrics_file = None
        if args.metrics_path:
            path = Path(args.metrics_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self.metrics_file = path.open("a", buffering=1)

    async def start(self) -> None:
        timeout = aiohttp.ClientTimeout(total=self.args.timeout)
        connector = aiohttp.TCPConnector(limit=0, limit_per_host=0)
        self.http = aiohttp.ClientSession(timeout=timeout, connector=connector)
        self.admission_lock = asyncio.Lock()
        self.admission_condition = asyncio.Condition(self.admission_lock)
        self.completed_stats_lock = asyncio.Lock()
        if self.args.max_inflight_requests is not None and not self.args.dynamic_admission:
            self.admission_semaphore = asyncio.Semaphore(self.args.max_inflight_requests)
        if self.args.dynamic_admission:
            self.dynamic_task = asyncio.create_task(self.dynamic_admission_loop())

    async def close(self) -> None:
        if self.dynamic_task is not None:
            self.dynamic_task.cancel()
            try:
                await self.dynamic_task
            except asyncio.CancelledError:
                pass
        if self.http is not None:
            await self.http.close()
        if self.metrics_file is not None:
            self.metrics_file.close()

    def log(self, obj: dict[str, Any]) -> None:
        if self.metrics_file is not None:
            self.metrics_file.write(json.dumps(obj, ensure_ascii=False) + "\n")

    async def acquire_admission(self) -> tuple[float, int, int, int | None]:
        if self.args.dynamic_admission:
            return await self.acquire_dynamic_admission()

        assert self.admission_lock is not None
        wait_start = time.monotonic()
        if self.admission_semaphore is None:
            async with self.admission_lock:
                self.active_requests += 1
                return 0.0, self.active_requests, self.waiting_requests, None

        waiting_added = False
        acquired = False
        admitted = False
        try:
            async with self.admission_lock:
                self.waiting_requests += 1
                waiting_added = True

            await self.admission_semaphore.acquire()
            acquired = True
            queue_wait = time.monotonic() - wait_start

            async with self.admission_lock:
                self.waiting_requests -= 1
                waiting_added = False
                self.active_requests += 1
                admitted = True
                return (
                    queue_wait,
                    self.active_requests,
                    self.waiting_requests,
                    self.current_inflight_limit,
                )
        except BaseException:
            async with self.admission_lock:
                if waiting_added:
                    self.waiting_requests -= 1
            if acquired and not admitted:
                self.admission_semaphore.release()
            raise

    async def acquire_dynamic_admission(self) -> tuple[float, int, int, int | None]:
        assert self.admission_condition is not None
        wait_start = time.monotonic()
        waiting_added = False
        try:
            async with self.admission_condition:
                self.waiting_requests += 1
                waiting_added = True
                while (
                    self.current_inflight_limit is not None
                    and self.active_requests >= self.current_inflight_limit
                ):
                    await self.admission_condition.wait()
                queue_wait = time.monotonic() - wait_start
                self.waiting_requests -= 1
                waiting_added = False
                self.active_requests += 1
                return (
                    queue_wait,
                    self.active_requests,
                    self.waiting_requests,
                    self.current_inflight_limit,
                )
        except BaseException:
            async with self.admission_condition:
                if waiting_added:
                    self.waiting_requests -= 1
                    self.admission_condition.notify_all()
            raise

    async def release_admission(self) -> tuple[int, int]:
        if self.args.dynamic_admission:
            assert self.admission_condition is not None
            async with self.admission_condition:
                self.active_requests = max(self.active_requests - 1, 0)
                active_requests = self.active_requests
                waiting_requests = self.waiting_requests
                self.admission_condition.notify_all()
                return active_requests, waiting_requests

        assert self.admission_lock is not None
        async with self.admission_lock:
            self.active_requests = max(self.active_requests - 1, 0)
            active_requests = self.active_requests
            waiting_requests = self.waiting_requests
        if self.admission_semaphore is not None:
            self.admission_semaphore.release()
        return active_requests, waiting_requests

    async def record_completed_request(self, stat: dict[str, Any]) -> None:
        if not self.args.dynamic_admission:
            return
        assert self.completed_stats_lock is not None
        async with self.completed_stats_lock:
            self.completed_request_stats.append(stat)

    async def drain_completed_request_stats(self) -> list[dict[str, Any]]:
        assert self.completed_stats_lock is not None
        async with self.completed_stats_lock:
            stats = self.completed_request_stats
            self.completed_request_stats = []
            return stats

    async def set_dynamic_limit(self, new_limit: int) -> tuple[int | None, int, int, int]:
        assert self.admission_condition is not None
        clamped = min(
            max(new_limit, self.args.dynamic_min_inflight_requests),
            self.args.dynamic_max_inflight_requests,
        )
        async with self.admission_condition:
            old_limit = self.current_inflight_limit
            self.current_inflight_limit = clamped
            active_requests = self.active_requests
            waiting_requests = self.waiting_requests
            self.admission_condition.notify_all()
            return old_limit, clamped, active_requests, waiting_requests

    async def admission_snapshot(self) -> dict[str, Any]:
        assert self.admission_lock is not None
        async with self.admission_lock:
            return {
                "dynamic": self.args.dynamic_admission,
                "active_requests": self.active_requests,
                "waiting_requests": self.waiting_requests,
                "current_inflight_limit": self.current_inflight_limit,
                "min_inflight_limit": self.args.dynamic_min_inflight_requests
                if self.args.dynamic_admission
                else None,
                "max_inflight_limit": self.args.dynamic_max_inflight_requests
                if self.args.dynamic_admission
                else self.args.max_inflight_requests,
            }

    def build_dynamic_interval_summary(
        self,
        stats: list[dict[str, Any]],
        window_secs: float,
        active_requests: int,
        waiting_requests: int,
    ) -> dict[str, Any]:
        ok = [
            s
            for s in stats
            if s.get("error") is None
            and isinstance(s.get("status"), int)
            and 200 <= s["status"] < 300
        ]
        prompt_tokens = sum(int(s.get("prompt_tokens") or 0) for s in ok)
        cached_tokens = sum(int(s.get("cached_tokens") or 0) for s in ok)
        completion_tokens = sum(int(s.get("completion_tokens") or 0) for s in ok)
        cache_hit_rate = (
            cached_tokens / prompt_tokens
            if prompt_tokens > 0
            else None
        )
        latencies = [float(s["latency"]) for s in stats if s.get("latency") is not None]
        ttfts = [float(s["ttft"]) for s in stats if s.get("ttft") is not None]
        upstream_ttfts = [
            float(s["upstream_ttft"])
            for s in stats
            if s.get("upstream_ttft") is not None
        ]
        queue_waits = [
            float(s["queue_wait"])
            for s in stats
            if s.get("queue_wait") is not None
        ]
        return {
            "window_secs": round(window_secs, 3),
            "requests": len(stats),
            "successful_requests": len(ok),
            "failed_requests": len(stats) - len(ok),
            "prompt_tokens": prompt_tokens,
            "cached_tokens": cached_tokens,
            "completion_tokens": completion_tokens,
            "cache_hit_rate": round_optional(cache_hit_rate),
            "request_throughput_req_s": round(len(ok) / window_secs, 3)
            if window_secs > 0
            else 0,
            "output_throughput_tok_s": round(completion_tokens / window_secs, 3)
            if window_secs > 0
            else 0,
            "latency": {
                "avg": round(sum(latencies) / len(latencies), 6) if latencies else 0,
                "p90": round(percentile(latencies, 90), 6),
                "p99": round(percentile(latencies, 99), 6),
            },
            "ttft": {
                "avg": round(sum(ttfts) / len(ttfts), 6) if ttfts else 0,
                "p90": round(percentile(ttfts, 90), 6),
                "p99": round(percentile(ttfts, 99), 6),
            },
            "upstream_ttft": {
                "avg": round(sum(upstream_ttfts) / len(upstream_ttfts), 6)
                if upstream_ttfts
                else 0,
                "p90": round(percentile(upstream_ttfts, 90), 6),
                "p99": round(percentile(upstream_ttfts, 99), 6),
            },
            "queue_wait": {
                "avg": round(sum(queue_waits) / len(queue_waits), 6)
                if queue_waits
                else 0,
                "p90": round(percentile(queue_waits, 90), 6),
                "p99": round(percentile(queue_waits, 99), 6),
            },
            "active_requests": active_requests,
            "waiting_requests": waiting_requests,
        }

    async def dynamic_admission_loop(self) -> None:
        last_tick = time.monotonic()
        while True:
            await asyncio.sleep(self.args.dynamic_control_interval)
            now = time.monotonic()
            window_secs = now - last_tick
            last_tick = now

            stats = await self.drain_completed_request_stats()
            snapshot = await self.admission_snapshot()
            current_limit = self.current_inflight_limit
            summary = self.build_dynamic_interval_summary(
                stats,
                window_secs,
                snapshot["active_requests"],
                snapshot["waiting_requests"],
            )

            action = "hold"
            reason = "stable"
            new_limit = current_limit
            if current_limit is None:
                reason = "unlimited"
            elif len(stats) < self.args.dynamic_min_samples:
                reason = "insufficient_samples"
            else:
                cache_hit = summary["cache_hit_rate"]
                upstream_ttft_p99 = summary["upstream_ttft"]["p99"]
                congested_reasons = []
                if cache_hit is not None and cache_hit < self.args.dynamic_low_cache_hit:
                    congested_reasons.append(
                        f"cache_hit<{self.args.dynamic_low_cache_hit:g}"
                    )
                if upstream_ttft_p99 > self.args.dynamic_high_ttft:
                    congested_reasons.append(
                        f"upstream_ttft_p99>{self.args.dynamic_high_ttft:g}"
                    )

                if congested_reasons:
                    action = "decrease"
                    reason = ",".join(congested_reasons)
                    new_limit = max(
                        self.args.dynamic_min_inflight_requests,
                        math.floor(current_limit * self.args.dynamic_decrease_factor),
                    )
                else:
                    cache_healthy = (
                        cache_hit is None
                        or cache_hit >= self.args.dynamic_high_cache_hit
                    )
                    upstream_healthy = (
                        upstream_ttft_p99
                        <= self.args.dynamic_high_ttft * 0.5
                    )
                    has_queued_demand = (
                        snapshot["waiting_requests"] > 0
                        or summary["queue_wait"]["p90"]
                        > self.args.dynamic_high_queue_wait
                    )
                    if (
                        cache_healthy
                        and upstream_healthy
                        and has_queued_demand
                        and current_limit < self.args.dynamic_max_inflight_requests
                    ):
                        action = "increase"
                        reason = "healthy_with_queued_demand"
                        new_limit = min(
                            self.args.dynamic_max_inflight_requests,
                            current_limit + self.args.dynamic_additive_step,
                        )

            old_limit = current_limit
            if new_limit is not None and new_limit != current_limit:
                old_limit, new_limit, active_requests, waiting_requests = (
                    await self.set_dynamic_limit(new_limit)
                )
                summary["active_requests"] = active_requests
                summary["waiting_requests"] = waiting_requests

            self.log({
                "type": "dynamic_admission_interval",
                "t": round(time.time(), 6),
                "action": action,
                "reason": reason,
                "old_inflight_limit": old_limit,
                "new_inflight_limit": new_limit,
                "min_inflight_limit": self.args.dynamic_min_inflight_requests,
                "max_inflight_limit": self.args.dynamic_max_inflight_requests,
                **summary,
            })


async def health(request: web.Request) -> web.Response:
    state: ProxyState = request.app["state"]
    return web.json_response({"ok": True, "admission": await state.admission_snapshot()})


async def proxy_handler(request: web.Request) -> web.StreamResponse:
    state: ProxyState = request.app["state"]
    assert state.http is not None

    started = time.monotonic()
    body = await request.read()
    payload: dict[str, Any] = {}
    if body:
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            payload = {}

    prompt_tokens_estimate = estimate_prompt_tokens(payload, state.args.chars_per_token)
    target_base, target_kind = choose_target(
        state.args.policy,
        prompt_tokens_estimate,
        state.args.length_threshold,
        state.args.direct_url,
        state.args.router_url,
    )
    target_url = target_base.rstrip("/") + request.rel_url.path_qs

    trace_id = request.headers.get("x-request-id") or uuid.uuid4().hex
    headers = filter_request_headers(request.headers)
    headers["x-request-id"] = trace_id
    headers["x-proxy-target"] = target_kind
    headers["x-prompt-tokens-estimate"] = str(prompt_tokens_estimate)

    status = 502
    error = None
    admitted = False
    queue_wait = 0.0
    active_requests_at_admit = 0
    waiting_requests_at_admit = 0
    inflight_limit_at_admit = None
    active_requests_after_done = 0
    waiting_requests_after_done = 0
    observation: dict[str, Any] = {
        "prompt_tokens": 0,
        "cached_tokens": 0,
        "completion_tokens": 0,
        "ttft": None,
        "upstream_ttft": None,
    }
    try:
        (
            queue_wait,
            active_requests_at_admit,
            waiting_requests_at_admit,
            inflight_limit_at_admit,
        ) = (
            await state.acquire_admission()
        )
        admitted = True
        upstream_started = time.monotonic()
        async with state.http.request(
            request.method,
            target_url,
            data=body,
            headers=headers,
        ) as upstream:
            status = upstream.status
            response = web.StreamResponse(
                status=upstream.status,
                reason=upstream.reason,
                headers=filter_response_headers(upstream.headers),
            )
            response.headers["x-proxy-target"] = target_kind
            response.headers["x-prompt-tokens-estimate"] = str(prompt_tokens_estimate)
            response.headers["x-proxy-queue-wait"] = f"{queue_wait:.6f}"
            response.headers["x-proxy-active-requests"] = str(active_requests_at_admit)
            response.headers["x-proxy-waiting-requests"] = str(waiting_requests_at_admit)
            if inflight_limit_at_admit is not None:
                response.headers["x-proxy-inflight-limit"] = str(inflight_limit_at_admit)
            if state.args.dynamic_admission:
                response.headers["x-proxy-dynamic-admission"] = "true"
            await response.prepare(request)

            decoder = codecs.getincrementaldecoder("utf-8")("replace")
            sse_buffer = ""
            async for chunk in upstream.content.iter_chunked(65536):
                text = decoder.decode(chunk)
                if text:
                    sse_buffer += text
                    while "\n" in sse_buffer:
                        line, sse_buffer = sse_buffer.split("\n", 1)
                        observe_sse_line(
                            line,
                            observation,
                            request_started=started,
                            upstream_started=upstream_started,
                            now=time.monotonic(),
                        )
                await response.write(chunk)
            tail = decoder.decode(b"", final=True)
            if tail:
                sse_buffer += tail
            if sse_buffer:
                for line in sse_buffer.splitlines():
                    observe_sse_line(
                        line,
                        observation,
                        request_started=started,
                        upstream_started=upstream_started,
                        now=time.monotonic(),
                    )
            await response.write_eof()
            return response
    except Exception as exc:
        error = str(exc)
        return web.json_response({"error": error}, status=502)
    finally:
        latency = time.monotonic() - started
        if admitted:
            active_requests_after_done, waiting_requests_after_done = (
                await state.release_admission()
            )
        prompt_tokens = int(observation.get("prompt_tokens") or 0)
        cached_tokens = int(observation.get("cached_tokens") or 0)
        completion_tokens = int(observation.get("completion_tokens") or 0)
        cache_hit_rate = cached_tokens / prompt_tokens if prompt_tokens > 0 else None
        request_stat = {
            "status": status,
            "latency": latency,
            "queue_wait": queue_wait,
            "ttft": observation.get("ttft"),
            "upstream_ttft": observation.get("upstream_ttft"),
            "prompt_tokens": prompt_tokens,
            "cached_tokens": cached_tokens,
            "completion_tokens": completion_tokens,
            "error": error,
        }
        await state.record_completed_request(request_stat)
        state.log({
            "type": "request",
            "t": round(time.time(), 6),
            "trace_id": trace_id,
            "method": request.method,
            "path": request.rel_url.path_qs,
            "target": target_kind,
            "target_url": target_url,
            "prompt_tokens_estimate": prompt_tokens_estimate,
            "status": status,
            "latency": round(latency, 6),
            "queue_wait": round(queue_wait, 6),
            "ttft": round_optional(observation.get("ttft")),
            "upstream_ttft": round_optional(observation.get("upstream_ttft")),
            "prompt_tokens": prompt_tokens,
            "cached_tokens": cached_tokens,
            "completion_tokens": completion_tokens,
            "cache_hit_rate": round_optional(cache_hit_rate),
            "active_requests_at_admit": active_requests_at_admit,
            "waiting_requests_at_admit": waiting_requests_at_admit,
            "active_requests_after_done": active_requests_after_done,
            "waiting_requests_after_done": waiting_requests_after_done,
            "max_inflight_requests": state.args.max_inflight_requests,
            "dynamic_admission": state.args.dynamic_admission,
            "inflight_limit_at_admit": inflight_limit_at_admit,
            "error": error,
        })


async def models_handler(request: web.Request) -> web.StreamResponse:
    state: ProxyState = request.app["state"]
    assert state.http is not None
    target_url = state.args.direct_url.rstrip("/") + request.rel_url.path_qs
    async with state.http.request(request.method, target_url, headers=filter_request_headers(request.headers)) as upstream:
        body = await upstream.read()
        return web.Response(
            status=upstream.status,
            reason=upstream.reason,
            headers=filter_response_headers(upstream.headers),
            body=body,
        )


def build_app(args: argparse.Namespace) -> web.Application:
    app = web.Application(client_max_size=args.client_max_size)
    state = ProxyState(args)
    app["state"] = state
    app.router.add_get("/health", health)
    app.router.add_route("*", "/v1/models", models_handler)
    app.router.add_route("*", "/v1/chat/completions", proxy_handler)
    app.router.add_route("*", "/{tail:.*}", proxy_handler)
    app.on_startup.append(lambda app: app["state"].start())
    app.on_cleanup.append(lambda app: app["state"].close())
    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Length-aware SGLang proxy")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8888)
    parser.add_argument("--direct-url", default="http://127.0.0.1:8000")
    parser.add_argument("--router-url", default="http://127.0.0.1:30000")
    parser.add_argument(
        "--policy",
        choices=["length_aware", "passthrough_direct", "passthrough_router"],
        default="length_aware",
    )
    parser.add_argument("--length-threshold", type=int, default=2048)
    parser.add_argument(
        "--chars-per-token",
        type=float,
        default=2.0,
        help="Approximate token estimator. T1 dummy text is close to 2 chars/token.",
    )
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--metrics-path", default="")
    parser.add_argument("--client-max-size", type=int, default=512 * 1024 * 1024)
    parser.add_argument(
        "--max-inflight-requests",
        type=int,
        default=None,
        help="Request-level admission window. Requests above this limit wait in proxy.",
    )
    parser.add_argument(
        "--dynamic-admission",
        action="store_true",
        help="Enable congestion-based dynamic adjustment of the proxy admission window.",
    )
    parser.add_argument(
        "--dynamic-min-inflight-requests",
        type=int,
        default=64,
        help="Lower bound for dynamic admission window.",
    )
    parser.add_argument(
        "--dynamic-initial-inflight-requests",
        type=int,
        default=None,
        help="Initial dynamic admission window. Defaults to --max-inflight-requests or 128.",
    )
    parser.add_argument(
        "--dynamic-max-inflight-requests",
        type=int,
        default=256,
        help="Upper bound for dynamic admission window.",
    )
    parser.add_argument(
        "--dynamic-control-interval",
        type=float,
        default=60.0,
        help="Seconds between dynamic admission control decisions.",
    )
    parser.add_argument(
        "--dynamic-min-samples",
        type=int,
        default=20,
        help="Minimum completed requests in a control interval before changing the window.",
    )
    parser.add_argument(
        "--dynamic-low-cache-hit",
        type=float,
        default=0.20,
        help="Decrease the window when interval cache hit rate falls below this value.",
    )
    parser.add_argument(
        "--dynamic-high-cache-hit",
        type=float,
        default=0.45,
        help="Allow additive increase when interval cache hit rate is at least this value.",
    )
    parser.add_argument(
        "--dynamic-high-ttft",
        type=float,
        default=60.0,
        help="Decrease the window when interval upstream TTFT p99 exceeds this many seconds.",
    )
    parser.add_argument(
        "--dynamic-high-queue-wait",
        type=float,
        default=1.0,
        help="Treat interval queue wait p90 above this many seconds as queued demand.",
    )
    parser.add_argument(
        "--dynamic-additive-step",
        type=int,
        default=16,
        help="Add this many inflight slots on a healthy interval with queued demand.",
    )
    parser.add_argument(
        "--dynamic-decrease-factor",
        type=float,
        default=0.70,
        help="Multiply current inflight window by this factor on congestion.",
    )
    args = parser.parse_args()
    if args.max_inflight_requests is not None and args.max_inflight_requests <= 0:
        parser.error("--max-inflight-requests must be positive")
    if args.dynamic_min_inflight_requests <= 0:
        parser.error("--dynamic-min-inflight-requests must be positive")
    if args.dynamic_max_inflight_requests < args.dynamic_min_inflight_requests:
        parser.error("--dynamic-max-inflight-requests must be >= --dynamic-min-inflight-requests")
    if (
        args.dynamic_initial_inflight_requests is not None
        and args.dynamic_initial_inflight_requests <= 0
    ):
        parser.error("--dynamic-initial-inflight-requests must be positive")
    if args.dynamic_control_interval <= 0:
        parser.error("--dynamic-control-interval must be positive")
    if args.dynamic_min_samples <= 0:
        parser.error("--dynamic-min-samples must be positive")
    if not 0 <= args.dynamic_low_cache_hit <= 1:
        parser.error("--dynamic-low-cache-hit must be in [0, 1]")
    if not 0 <= args.dynamic_high_cache_hit <= 1:
        parser.error("--dynamic-high-cache-hit must be in [0, 1]")
    if args.dynamic_high_cache_hit < args.dynamic_low_cache_hit:
        parser.error("--dynamic-high-cache-hit must be >= --dynamic-low-cache-hit")
    if args.dynamic_high_ttft <= 0:
        parser.error("--dynamic-high-ttft must be positive")
    if args.dynamic_high_queue_wait < 0:
        parser.error("--dynamic-high-queue-wait must be non-negative")
    if args.dynamic_additive_step <= 0:
        parser.error("--dynamic-additive-step must be positive")
    if not 0 < args.dynamic_decrease_factor < 1:
        parser.error("--dynamic-decrease-factor must be in (0, 1)")
    return args


def main() -> None:
    args = parse_args()
    web.run_app(build_app(args), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
