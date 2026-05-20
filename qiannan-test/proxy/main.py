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
import re
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


def parse_int_header(headers: Any, name: str) -> int | None:
    value = headers.get(name)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def truthy_header(value: str | None) -> bool:
    if value is None:
        return False
    return value.lower() in {"1", "true", "yes", "on"}


PROMETHEUS_METRIC_RE = re.compile(
    r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^}]*\})?\s+([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)$"
)


def parse_prometheus_metrics(text: str) -> dict[str, list[float]]:
    metrics: dict[str, list[float]] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = PROMETHEUS_METRIC_RE.match(line)
        if match is None:
            continue
        name, value = match.groups()
        try:
            metrics.setdefault(name, []).append(float(value))
        except ValueError:
            continue
    return metrics


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
        self.session_condition: asyncio.Condition | None = None
        self.dynamic_task: asyncio.Task | None = None
        self.metrics_task: asyncio.Task | None = None
        self.session_cleanup_task: asyncio.Task | None = None
        self.completed_stats_lock: asyncio.Lock | None = None
        self.server_metrics_lock: asyncio.Lock | None = None
        self.completed_request_stats: list[dict[str, Any]] = []
        self.latest_server_metrics: dict[str, list[float]] = {}
        self.latest_server_metrics_error: str | None = None
        self.latest_server_metrics_t: float | None = None
        self.active_requests = 0
        self.waiting_requests = 0
        self.active_sessions: dict[str, dict[str, Any]] = {}
        self.waiting_session_requests = 0
        self.current_session_limit: int | None = None
        self.current_inflight_limit: int | None = None
        if args.dynamic_admission:
            initial_limit = (
                args.dynamic_initial_inflight_requests
                if args.dynamic_initial_inflight_requests is not None
                else args.max_active_sessions
                if args.max_active_sessions is not None
                else args.max_inflight_requests
                if args.max_inflight_requests is not None
                else 128
            )
            self.current_session_limit = min(
                max(initial_limit, args.dynamic_min_inflight_requests),
                args.dynamic_max_inflight_requests,
            )
        elif args.max_active_sessions is not None:
            self.current_session_limit = args.max_active_sessions
        if args.max_inflight_requests is not None:
            self.current_inflight_limit = args.max_inflight_requests
        self.server_metrics_url = args.server_metrics_url
        if args.dynamic_admission and not self.server_metrics_url:
            self.server_metrics_url = args.direct_url.rstrip("/") + "/metrics"
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
        self.session_condition = asyncio.Condition(asyncio.Lock())
        self.completed_stats_lock = asyncio.Lock()
        self.server_metrics_lock = asyncio.Lock()
        if self.args.max_inflight_requests is not None:
            self.admission_semaphore = asyncio.Semaphore(self.args.max_inflight_requests)
        if self.args.max_active_sessions is not None or self.args.dynamic_admission:
            self.session_cleanup_task = asyncio.create_task(self.session_cleanup_loop())
        if self.args.dynamic_admission:
            if self.server_metrics_url:
                self.metrics_task = asyncio.create_task(self.server_metrics_loop())
            self.dynamic_task = asyncio.create_task(self.dynamic_admission_loop())
        self.log({
            "type": "proxy_start",
            "t": round(time.time(), 6),
            "policy": self.args.policy,
            "direct_url": self.args.direct_url,
            "router_url": self.args.router_url,
            "max_inflight_requests": self.args.max_inflight_requests,
            "max_active_sessions": self.args.max_active_sessions,
            "dynamic_admission": self.args.dynamic_admission,
            "current_inflight_limit": self.current_inflight_limit,
            "current_session_limit": self.current_session_limit,
            "server_metrics_url": self.server_metrics_url,
            "session_idle_timeout": self.args.session_idle_timeout,
        })

    async def close(self) -> None:
        if self.dynamic_task is not None:
            self.dynamic_task.cancel()
            try:
                await self.dynamic_task
            except asyncio.CancelledError:
                pass
        if self.metrics_task is not None:
            self.metrics_task.cancel()
            try:
                await self.metrics_task
            except asyncio.CancelledError:
                pass
        if self.session_cleanup_task is not None:
            self.session_cleanup_task.cancel()
            try:
                await self.session_cleanup_task
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

    async def release_admission(self) -> tuple[int, int]:
        assert self.admission_lock is not None
        async with self.admission_lock:
            self.active_requests = max(self.active_requests - 1, 0)
            active_requests = self.active_requests
            waiting_requests = self.waiting_requests
        if self.admission_semaphore is not None:
            self.admission_semaphore.release()
        return active_requests, waiting_requests

    def session_admission_enabled(self) -> bool:
        return self.args.max_active_sessions is not None or self.args.dynamic_admission

    def effective_session_limit(self) -> int | None:
        if self.args.dynamic_admission:
            return self.current_session_limit
        return self.args.max_active_sessions

    def active_session_count_locked(self) -> int:
        return sum(
            1
            for meta in self.active_sessions.values()
            if not meta.get("paused", False)
        )

    def projected_active_session_count_locked(self) -> int:
        return sum(
            1
            for meta in self.active_sessions.values()
            if not meta.get("paused", False)
            and not meta.get("pause_requested", False)
        )

    def paused_session_count_locked(self) -> int:
        return sum(
            1
            for meta in self.active_sessions.values()
            if meta.get("paused", False)
        )

    def pause_requested_session_count_locked(self) -> int:
        return sum(
            1
            for meta in self.active_sessions.values()
            if meta.get("pause_requested", False)
        )

    def mark_pause_victims_locked(self, new_limit: int) -> list[dict[str, Any]]:
        now = time.monotonic()
        projected_active = self.projected_active_session_count_locked()
        excess = max(projected_active - new_limit, 0)
        if excess <= 0:
            return []

        def victim_key(item: tuple[str, dict[str, Any]]) -> tuple[Any, ...]:
            _, meta = item
            in_flight = int(meta.get("in_flight", 0)) > 0
            idle_since = meta.get("last_round_done_at") or meta.get("last_seen") or now
            idle_seconds = now - idle_since
            cache_hit = meta.get("last_cache_hit_rate")
            cache_hit_sort = cache_hit if cache_hit is not None else 1.0
            prompt_tokens = int(meta.get("prompt_tokens_estimate") or 0)
            last_resumed_at = float(meta.get("last_resumed_at") or 0.0)
            recently_resumed = (
                self.args.session_pause_min_secs > 0
                and now - last_resumed_at < self.args.session_pause_min_secs
            )
            return (
                recently_resumed,
                in_flight,
                -idle_seconds,
                cache_hit_sort,
                -prompt_tokens,
                int(meta.get("pause_count") or 0),
                float(meta.get("started") or now),
            )

        candidates = [
            (session_id, meta)
            for session_id, meta in self.active_sessions.items()
            if not meta.get("paused", False)
            and not meta.get("pause_requested", False)
        ]
        candidates.sort(key=victim_key)

        victims: list[dict[str, Any]] = []
        for session_id, meta in candidates[:excess]:
            in_flight = int(meta.get("in_flight", 0))
            if in_flight > 0:
                meta["pause_requested"] = True
                state = "pause_requested"
            else:
                meta["paused"] = True
                meta["pause_requested"] = False
                meta["pause_count"] = int(meta.get("pause_count") or 0) + 1
                meta["last_pause_at"] = now
                meta["resume_not_before"] = now + self.args.session_pause_min_secs
                state = "paused"
            victim = {
                "session_id": session_id,
                "state": state,
                "in_flight": in_flight,
                "prompt_tokens_estimate": meta.get("prompt_tokens_estimate"),
                "last_cache_hit_rate": meta.get("last_cache_hit_rate"),
                "pause_count": meta.get("pause_count", 0),
            }
            victims.append(victim)
            self.log({
                "type": "session_pause_mark",
                "t": round(time.time(), 6),
                "new_session_limit": new_limit,
                **victim,
            })
        return victims

    async def acquire_session_admission(
        self,
        session_id: str | None,
        prompt_tokens_estimate: int,
    ) -> tuple[float, int, int, int | None, bool]:
        if not self.session_admission_enabled():
            return 0.0, 0, 0, None, False
        if session_id is None:
            return 0.0, 0, 0, self.effective_session_limit(), False

        assert self.session_condition is not None
        wait_start = time.monotonic()
        waiting_added = False
        try:
            async with self.session_condition:
                self.waiting_session_requests += 1
                waiting_added = True
                while True:
                    now = time.monotonic()
                    limit = self.effective_session_limit()
                    meta = self.active_sessions.get(session_id)

                    if meta is not None:
                        if (
                            meta.get("pause_requested", False)
                            and int(meta.get("in_flight") or 0) <= 0
                        ):
                            meta["pause_requested"] = False
                            meta["paused"] = True
                            meta["pause_count"] = int(meta.get("pause_count") or 0) + 1
                            meta["last_pause_at"] = now
                            meta["resume_not_before"] = (
                                now + self.args.session_pause_min_secs
                            )

                        if meta.get("paused", False):
                            can_resume = (
                                self.args.dynamic_admission
                                and (
                                    limit is None
                                    or self.active_session_count_locked() < limit
                                )
                                and now >= float(meta.get("resume_not_before") or 0.0)
                            )
                            if can_resume:
                                meta["paused"] = False
                                meta["last_resumed_at"] = now
                                self.log({
                                    "type": "session_resume",
                                    "t": round(time.time(), 6),
                                    "session_id": session_id,
                                    "current_session_limit": limit,
                                    "active_sessions": self.active_session_count_locked(),
                                    "paused_sessions": self.paused_session_count_locked(),
                                })
                            else:
                                await self.session_condition.wait()
                                continue

                        if meta.get("pause_requested", False):
                            await self.session_condition.wait()
                            continue

                        queue_wait = time.monotonic() - wait_start
                        self.waiting_session_requests -= 1
                        waiting_added = False
                        meta["last_seen"] = time.monotonic()
                        meta["requests"] = int(meta.get("requests") or 0) + 1
                        meta["in_flight"] = int(meta.get("in_flight") or 0) + 1
                        meta["prompt_tokens_estimate"] = max(
                            int(meta.get("prompt_tokens_estimate") or 0),
                            prompt_tokens_estimate,
                        )
                        return (
                            queue_wait,
                            self.active_session_count_locked(),
                            self.waiting_session_requests,
                            limit,
                            True,
                        )

                    if (
                        limit is not None
                        and self.active_session_count_locked() >= limit
                    ):
                        await self.session_condition.wait()
                        continue

                    self.active_sessions[session_id] = {
                        "started": now,
                        "last_seen": now,
                        "last_round_done_at": None,
                        "last_pause_at": None,
                        "last_resumed_at": now,
                        "resume_not_before": 0.0,
                        "requests": 1,
                        "in_flight": 1,
                        "paused": False,
                        "pause_requested": False,
                        "pause_count": 0,
                        "prompt_tokens_estimate": prompt_tokens_estimate,
                        "last_cache_hit_rate": None,
                    }
                    queue_wait = time.monotonic() - wait_start
                    self.waiting_session_requests -= 1
                    waiting_added = False
                    return (
                        queue_wait,
                        self.active_session_count_locked(),
                        self.waiting_session_requests,
                        limit,
                        True,
                    )
        except BaseException:
            async with self.session_condition:
                if waiting_added:
                    self.waiting_session_requests -= 1
                    self.session_condition.notify_all()
            raise

    async def release_session_if_done(
        self,
        session_id: str | None,
        should_release: bool,
        prompt_tokens_estimate: int = 0,
        cache_hit_rate: float | None = None,
    ) -> tuple[int, int]:
        if (
            not self.session_admission_enabled()
            or session_id is None
        ):
            return self.active_session_count_locked(), self.waiting_session_requests

        assert self.session_condition is not None
        async with self.session_condition:
            now = time.monotonic()
            meta = self.active_sessions.get(session_id)
            if meta is None:
                return (
                    self.active_session_count_locked(),
                    self.waiting_session_requests,
                )
            meta["in_flight"] = max(int(meta.get("in_flight") or 0) - 1, 0)
            meta["last_round_done_at"] = now
            meta["last_seen"] = now
            meta["prompt_tokens_estimate"] = max(
                int(meta.get("prompt_tokens_estimate") or 0),
                prompt_tokens_estimate,
            )
            if cache_hit_rate is not None:
                meta["last_cache_hit_rate"] = cache_hit_rate

            if should_release:
                self.active_sessions.pop(session_id, None)
            elif (
                self.args.dynamic_admission
                and meta.get("pause_requested", False)
                and int(meta.get("in_flight") or 0) <= 0
            ):
                meta["pause_requested"] = False
                meta["paused"] = True
                meta["pause_count"] = int(meta.get("pause_count") or 0) + 1
                meta["last_pause_at"] = now
                meta["resume_not_before"] = now + self.args.session_pause_min_secs
                self.log({
                    "type": "session_paused",
                    "t": round(time.time(), 6),
                    "session_id": session_id,
                    "current_session_limit": self.current_session_limit,
                    "prompt_tokens_estimate": meta.get("prompt_tokens_estimate"),
                    "last_cache_hit_rate": meta.get("last_cache_hit_rate"),
                    "pause_count": meta.get("pause_count", 0),
                })

            active_sessions = self.active_session_count_locked()
            waiting_sessions = self.waiting_session_requests
            self.session_condition.notify_all()
            return active_sessions, waiting_sessions

    async def session_snapshot(self) -> dict[str, Any]:
        if self.session_condition is None:
            return {
                "known_sessions": 0,
                "active_sessions": 0,
                "paused_sessions": 0,
                "pause_requested_sessions": 0,
                "waiting_session_requests": 0,
                "max_active_sessions": self.args.max_active_sessions,
                "current_session_limit": self.current_session_limit,
                "min_session_limit": None,
                "max_session_limit": self.args.max_active_sessions,
            }

        async with self.session_condition:
            return {
                "known_sessions": len(self.active_sessions),
                "active_sessions": self.active_session_count_locked(),
                "paused_sessions": self.paused_session_count_locked(),
                "pause_requested_sessions": self.pause_requested_session_count_locked(),
                "waiting_session_requests": self.waiting_session_requests,
                "max_active_sessions": self.args.max_active_sessions,
                "current_session_limit": self.current_session_limit,
                "min_session_limit": self.args.dynamic_min_inflight_requests
                if self.args.dynamic_admission
                else None,
                "max_session_limit": self.args.dynamic_max_inflight_requests
                if self.args.dynamic_admission
                else self.args.max_active_sessions,
            }

    async def session_cleanup_loop(self) -> None:
        while True:
            await asyncio.sleep(self.args.session_cleanup_interval)
            assert self.session_condition is not None
            now = time.monotonic()
            expired: list[str] = []
            async with self.session_condition:
                for session_id, meta in self.active_sessions.items():
                    if int(meta.get("in_flight") or 0) > 0:
                        continue
                    if now - meta["last_seen"] > self.args.session_idle_timeout:
                        expired.append(session_id)
                for session_id in expired:
                    self.active_sessions.pop(session_id, None)
                if expired:
                    self.session_condition.notify_all()
            for session_id in expired:
                self.log({
                    "type": "session_admission_timeout",
                    "t": round(time.time(), 6),
                    "session_id": session_id,
                    "session_idle_timeout": self.args.session_idle_timeout,
                })

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

    async def set_dynamic_session_limit(
        self,
        new_limit: int,
    ) -> tuple[int | None, int, int, int, int, int, list[dict[str, Any]]]:
        assert self.session_condition is not None
        clamped = min(
            max(new_limit, self.args.dynamic_min_inflight_requests),
            self.args.dynamic_max_inflight_requests,
        )
        async with self.session_condition:
            old_limit = self.current_session_limit
            self.current_session_limit = clamped
            victims = self.mark_pause_victims_locked(clamped)
            active_sessions = self.active_session_count_locked()
            waiting_sessions = self.waiting_session_requests
            paused_sessions = self.paused_session_count_locked()
            pause_requested_sessions = self.pause_requested_session_count_locked()
            self.session_condition.notify_all()
            return (
                old_limit,
                clamped,
                active_sessions,
                waiting_sessions,
                paused_sessions,
                pause_requested_sessions,
                victims,
            )

    async def server_metrics_loop(self) -> None:
        while True:
            try:
                await self.scrape_server_metrics_once()
            except Exception as exc:
                assert self.server_metrics_lock is not None
                async with self.server_metrics_lock:
                    self.latest_server_metrics_error = str(exc)
                    self.latest_server_metrics_t = time.time()
            await asyncio.sleep(self.args.server_metrics_interval)

    async def scrape_server_metrics_once(self) -> None:
        if not self.server_metrics_url:
            return
        assert self.http is not None
        assert self.server_metrics_lock is not None
        async with self.http.get(self.server_metrics_url) as response:
            response.raise_for_status()
            text = await response.text()
        parsed = parse_prometheus_metrics(text)
        async with self.server_metrics_lock:
            self.latest_server_metrics = parsed
            self.latest_server_metrics_error = None
            self.latest_server_metrics_t = time.time()

    async def server_feedback_snapshot(self) -> dict[str, Any]:
        assert self.server_metrics_lock is not None
        async with self.server_metrics_lock:
            metrics = {
                key: list(values)
                for key, values in self.latest_server_metrics.items()
            }
            error = self.latest_server_metrics_error
            metrics_t = self.latest_server_metrics_t

        def values_for(*names: str) -> list[float]:
            values: list[float] = []
            for name in names:
                values.extend(metrics.get(name, []))
            return values

        usage_values = values_for(
            "sglang:token_usage",
            "sglang:full_token_usage",
        )
        cache_values = values_for("sglang:cache_hit_rate")
        positive_cache_values = [v for v in cache_values if v > 0]
        return {
            "metrics_url": self.server_metrics_url,
            "metrics_age": round(time.time() - metrics_t, 3)
            if metrics_t is not None
            else None,
            "metrics_error": error,
            "kv_cache_usage": max(usage_values) if usage_values else None,
            "cache_hit_rate": (
                sum(positive_cache_values) / len(positive_cache_values)
                if positive_cache_values
                else max(cache_values)
                if cache_values
                else None
            ),
        }

    async def admission_snapshot(self) -> dict[str, Any]:
        assert self.admission_lock is not None
        async with self.admission_lock:
            return {
                "dynamic": self.args.dynamic_admission,
                "active_requests": self.active_requests,
                "waiting_requests": self.waiting_requests,
                "current_inflight_limit": self.current_inflight_limit,
                "min_inflight_limit": None,
                "max_inflight_limit": self.args.max_inflight_requests,
            }

    def build_dynamic_interval_summary(
        self,
        stats: list[dict[str, Any]],
        window_secs: float,
        active_requests: int,
        waiting_requests: int,
        session_snapshot: dict[str, Any],
        server_feedback: dict[str, Any],
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
            "known_sessions": session_snapshot["known_sessions"],
            "active_sessions": session_snapshot["active_sessions"],
            "paused_sessions": session_snapshot["paused_sessions"],
            "pause_requested_sessions": session_snapshot[
                "pause_requested_sessions"
            ],
            "waiting_session_requests": session_snapshot[
                "waiting_session_requests"
            ],
            "kv_cache_usage": round_optional(server_feedback["kv_cache_usage"]),
            "server_cache_hit_rate": round_optional(
                server_feedback["cache_hit_rate"]
            ),
            "server_metrics_age": server_feedback["metrics_age"],
            "server_metrics_error": server_feedback["metrics_error"],
        }

    async def dynamic_admission_loop(self) -> None:
        last_tick = time.monotonic()
        while True:
            await asyncio.sleep(self.args.dynamic_control_interval)
            now = time.monotonic()
            window_secs = now - last_tick
            last_tick = now

            stats = await self.drain_completed_request_stats()
            admission_snapshot = await self.admission_snapshot()
            session_snapshot = await self.session_snapshot()
            server_feedback = await self.server_feedback_snapshot()
            current_limit = self.current_session_limit
            summary = self.build_dynamic_interval_summary(
                stats,
                window_secs,
                admission_snapshot["active_requests"],
                admission_snapshot["waiting_requests"],
                session_snapshot,
                server_feedback,
            )

            action = "hold"
            reason = "stable"
            new_limit = current_limit
            if current_limit is None:
                reason = "unlimited"
            else:
                kv_cache_usage = server_feedback["kv_cache_usage"]
                cache_hit = (
                    server_feedback["cache_hit_rate"]
                    if server_feedback["cache_hit_rate"] is not None
                    else summary["cache_hit_rate"]
                )
                upstream_ttft_p99 = summary["upstream_ttft"]["p99"]
                if kv_cache_usage is not None:
                    if kv_cache_usage < self.args.dynamic_low_kv_usage:
                        action = "increase"
                        reason = f"kv_usage<{self.args.dynamic_low_kv_usage:g}"
                        new_limit = min(
                            self.args.dynamic_max_inflight_requests,
                            current_limit + self.args.dynamic_additive_step,
                        )
                    elif (
                        kv_cache_usage > self.args.dynamic_high_kv_usage
                        and cache_hit is not None
                        and cache_hit < self.args.dynamic_cache_hit_threshold
                    ):
                        action = "decrease"
                        reason = (
                            f"kv_usage>{self.args.dynamic_high_kv_usage:g},"
                            f"cache_hit<{self.args.dynamic_cache_hit_threshold:g}"
                        )
                        new_limit = max(
                            self.args.dynamic_min_inflight_requests,
                            math.floor(
                                current_limit * self.args.dynamic_decrease_factor
                            ),
                        )
                    else:
                        reason = "kv_feedback_stable"
                elif len(stats) < self.args.dynamic_min_samples:
                    reason = "insufficient_samples_no_kv_feedback"
                else:
                    congested_reasons = []
                    if (
                        cache_hit is not None
                        and cache_hit < self.args.dynamic_cache_hit_threshold
                    ):
                        congested_reasons.append(
                            f"cache_hit<{self.args.dynamic_cache_hit_threshold:g}"
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
                            math.floor(
                                current_limit * self.args.dynamic_decrease_factor
                            ),
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
                            session_snapshot["waiting_session_requests"] > 0
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
                            reason = "healthy_with_queued_session_demand"
                            new_limit = min(
                                self.args.dynamic_max_inflight_requests,
                                current_limit + self.args.dynamic_additive_step,
                            )

            old_limit = current_limit
            pause_victims: list[dict[str, Any]] = []
            if new_limit is not None and new_limit != current_limit:
                (
                    old_limit,
                    new_limit,
                    active_sessions,
                    waiting_sessions,
                    paused_sessions,
                    pause_requested_sessions,
                    pause_victims,
                ) = (
                    await self.set_dynamic_session_limit(new_limit)
                )
                summary["active_sessions"] = active_sessions
                summary["waiting_session_requests"] = waiting_sessions
                summary["paused_sessions"] = paused_sessions
                summary["pause_requested_sessions"] = pause_requested_sessions

            self.log({
                "type": "dynamic_admission_interval",
                "t": round(time.time(), 6),
                "action": action,
                "reason": reason,
                "old_session_limit": old_limit,
                "new_session_limit": new_limit,
                "min_session_limit": self.args.dynamic_min_inflight_requests,
                "max_session_limit": self.args.dynamic_max_inflight_requests,
                "pause_victims": pause_victims,
                **summary,
            })


async def health(request: web.Request) -> web.Response:
    state: ProxyState = request.app["state"]
    return web.json_response({
        "ok": True,
        "admission": await state.admission_snapshot(),
        "session_admission": await state.session_snapshot(),
        "server_feedback": await state.server_feedback_snapshot()
        if state.args.dynamic_admission
        else None,
    })


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
    session_id = request.headers.get("x-session-id")
    session_round_idx = parse_int_header(request.headers, "x-session-round-idx")
    session_rounds = parse_int_header(request.headers, "x-session-rounds")
    session_is_final = truthy_header(request.headers.get("x-session-final-round"))
    if (
        not session_is_final
        and session_round_idx is not None
        and session_rounds is not None
    ):
        session_is_final = session_round_idx >= session_rounds - 1
    if state.session_admission_enabled() and session_id is None:
        return web.json_response(
            {"error": "session admission requires x-session-id header"},
            status=400,
        )

    headers = filter_request_headers(request.headers)
    headers["x-request-id"] = trace_id
    headers["x-proxy-target"] = target_kind
    headers["x-prompt-tokens-estimate"] = str(prompt_tokens_estimate)

    status = 502
    error = None
    session_slot_acquired = False
    session_queue_wait = 0.0
    active_sessions_at_admit = 0
    waiting_session_requests_at_admit = 0
    max_active_sessions_at_admit = None
    active_sessions_after_done = 0
    waiting_session_requests_after_done = 0
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
            session_queue_wait,
            active_sessions_at_admit,
            waiting_session_requests_at_admit,
            max_active_sessions_at_admit,
            session_slot_acquired,
        ) = await state.acquire_session_admission(
            session_id,
            prompt_tokens_estimate,
        )
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
            if session_slot_acquired:
                response.headers["x-proxy-session-queue-wait"] = f"{session_queue_wait:.6f}"
                response.headers["x-proxy-active-sessions"] = str(active_sessions_at_admit)
                response.headers["x-proxy-waiting-session-requests"] = str(
                    waiting_session_requests_at_admit
                )
                response.headers["x-proxy-max-active-sessions"] = str(
                    max_active_sessions_at_admit
                )
                if state.args.dynamic_admission:
                    response.headers["x-proxy-active-session-limit"] = str(
                        max_active_sessions_at_admit
                    )
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
        session_release = session_is_final or error is not None or status >= 400
        (
            active_sessions_after_done,
            waiting_session_requests_after_done,
        ) = await state.release_session_if_done(
            session_id,
            session_slot_acquired and session_release,
            prompt_tokens_estimate=prompt_tokens_estimate,
            cache_hit_rate=cache_hit_rate,
        )
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
            "session_id": session_id,
            "session_round_idx": session_round_idx,
            "session_rounds": session_rounds,
            "session_is_final": session_is_final,
            "prompt_tokens_estimate": prompt_tokens_estimate,
            "status": status,
            "latency": round(latency, 6),
            "queue_wait": round(queue_wait, 6),
            "session_queue_wait": round(session_queue_wait, 6),
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
            "active_sessions_at_admit": active_sessions_at_admit,
            "waiting_session_requests_at_admit": waiting_session_requests_at_admit,
            "active_sessions_after_done": active_sessions_after_done,
            "waiting_session_requests_after_done": waiting_session_requests_after_done,
            "max_active_sessions": state.args.max_active_sessions,
            "session_limit_at_admit": max_active_sessions_at_admit,
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
        help="Static request-level safety cap. Dynamic admission does not tune this window.",
    )
    parser.add_argument(
        "--max-active-sessions",
        type=int,
        default=None,
        help="Session-level admission window. Requires x-session-id and holds a slot until the final round.",
    )
    parser.add_argument(
        "--session-idle-timeout",
        type=float,
        default=3600.0,
        help="Release an admitted session if no request for this many seconds.",
    )
    parser.add_argument(
        "--session-cleanup-interval",
        type=float,
        default=30.0,
        help="Seconds between idle session cleanup passes.",
    )
    parser.add_argument(
        "--dynamic-admission",
        action="store_true",
        help="Enable congestion-based dynamic adjustment of the active session window.",
    )
    parser.add_argument(
        "--dynamic-min-inflight-requests",
        "--dynamic-min-active-sessions",
        dest="dynamic_min_inflight_requests",
        type=int,
        default=16,
        help="Lower bound for dynamic active session window. The old inflight name is kept for compatibility.",
    )
    parser.add_argument(
        "--dynamic-initial-inflight-requests",
        "--dynamic-initial-active-sessions",
        dest="dynamic_initial_inflight_requests",
        type=int,
        default=None,
        help="Initial dynamic active session window. Defaults to --max-active-sessions, --max-inflight-requests, or 128.",
    )
    parser.add_argument(
        "--dynamic-max-inflight-requests",
        "--dynamic-max-active-sessions",
        dest="dynamic_max_inflight_requests",
        type=int,
        default=256,
        help="Upper bound for dynamic active session window. The old inflight name is kept for compatibility.",
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
        help="Legacy cache-hit threshold. Used as fallback when --dynamic-cache-hit-threshold is not set.",
    )
    parser.add_argument(
        "--dynamic-cache-hit-threshold",
        type=float,
        default=None,
        help="Decrease the session window when KV usage is high and cache hit rate is below this value.",
    )
    parser.add_argument(
        "--dynamic-high-cache-hit",
        type=float,
        default=0.45,
        help="Fallback no-KV-feedback threshold for additive increase.",
    )
    parser.add_argument(
        "--dynamic-low-kv-usage",
        type=float,
        default=0.20,
        help="Additively increase active session window when SGLang token_usage is below this value.",
    )
    parser.add_argument(
        "--dynamic-high-kv-usage",
        type=float,
        default=0.50,
        help="Multiplicatively decrease active session window when token_usage is above this value and hit rate is low.",
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
        default=2,
        help="Add this many active session slots on an under-utilized interval.",
    )
    parser.add_argument(
        "--dynamic-decrease-factor",
        type=float,
        default=0.50,
        help="Multiply current active session window by this factor on congestion.",
    )
    parser.add_argument(
        "--session-pause-min-secs",
        type=float,
        default=0.0,
        help="Minimum pause duration before a paused session can be resumed.",
    )
    parser.add_argument(
        "--server-metrics-url",
        default="",
        help="Prometheus metrics URL for SGLang. Defaults to --direct-url/metrics when dynamic admission is enabled.",
    )
    parser.add_argument(
        "--server-metrics-interval",
        type=float,
        default=1.0,
        help="Seconds between SGLang metrics scrapes.",
    )
    args = parser.parse_args()
    if args.max_inflight_requests is not None and args.max_inflight_requests <= 0:
        parser.error("--max-inflight-requests must be positive")
    if args.max_active_sessions is not None and args.max_active_sessions <= 0:
        parser.error("--max-active-sessions must be positive")
    if args.session_idle_timeout <= 0:
        parser.error("--session-idle-timeout must be positive")
    if args.session_cleanup_interval <= 0:
        parser.error("--session-cleanup-interval must be positive")
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
    if args.dynamic_cache_hit_threshold is None:
        args.dynamic_cache_hit_threshold = args.dynamic_low_cache_hit
    if not 0 <= args.dynamic_cache_hit_threshold <= 1:
        parser.error("--dynamic-cache-hit-threshold must be in [0, 1]")
    if not 0 <= args.dynamic_high_cache_hit <= 1:
        parser.error("--dynamic-high-cache-hit must be in [0, 1]")
    if args.dynamic_high_cache_hit < args.dynamic_low_cache_hit:
        parser.error("--dynamic-high-cache-hit must be >= --dynamic-low-cache-hit")
    if not 0 <= args.dynamic_low_kv_usage <= 1:
        parser.error("--dynamic-low-kv-usage must be in [0, 1]")
    if not 0 <= args.dynamic_high_kv_usage <= 1:
        parser.error("--dynamic-high-kv-usage must be in [0, 1]")
    if args.dynamic_high_kv_usage < args.dynamic_low_kv_usage:
        parser.error("--dynamic-high-kv-usage must be >= --dynamic-low-kv-usage")
    if args.dynamic_high_ttft <= 0:
        parser.error("--dynamic-high-ttft must be positive")
    if args.dynamic_high_queue_wait < 0:
        parser.error("--dynamic-high-queue-wait must be non-negative")
    if args.dynamic_additive_step <= 0:
        parser.error("--dynamic-additive-step must be positive")
    if not 0 < args.dynamic_decrease_factor < 1:
        parser.error("--dynamic-decrease-factor must be in (0, 1)")
    if args.session_pause_min_secs < 0:
        parser.error("--session-pause-min-secs must be non-negative")
    if args.server_metrics_interval <= 0:
        parser.error("--server-metrics-interval must be positive")
    return args


def main() -> None:
    args = parse_args()
    web.run_app(build_app(args), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
