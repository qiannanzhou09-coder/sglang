#!/usr/bin/env python3
"""
Small OpenAI-compatible proxy for SGLang serving experiments.

length_aware routes short prompts directly to SGLang so DPA can load-balance,
and routes longer prompts through sglang_router cache_aware.
"""

import argparse
import json
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


class ProxyState:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.http: aiohttp.ClientSession | None = None
        self.metrics_file = None
        if args.metrics_path:
            path = Path(args.metrics_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self.metrics_file = path.open("a", buffering=1)

    async def start(self) -> None:
        timeout = aiohttp.ClientTimeout(total=self.args.timeout)
        connector = aiohttp.TCPConnector(limit=0, limit_per_host=0)
        self.http = aiohttp.ClientSession(timeout=timeout, connector=connector)

    async def close(self) -> None:
        if self.http is not None:
            await self.http.close()
        if self.metrics_file is not None:
            self.metrics_file.close()

    def log(self, obj: dict[str, Any]) -> None:
        if self.metrics_file is not None:
            self.metrics_file.write(json.dumps(obj, ensure_ascii=False) + "\n")


async def health(_: web.Request) -> web.Response:
    return web.json_response({"ok": True})


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
    try:
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
            await response.prepare(request)
            async for chunk in upstream.content.iter_chunked(65536):
                await response.write(chunk)
            await response.write_eof()
            return response
    except Exception as exc:
        error = str(exc)
        return web.json_response({"error": error}, status=502)
    finally:
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
            "latency": round(time.monotonic() - started, 6),
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    web.run_app(build_app(args), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
