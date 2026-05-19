#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


BENCHMARK_DIR = Path(__file__).resolve().parent
TEST_ROOT = BENCHMARK_DIR.parent
REPO_ROOT = TEST_ROOT.parent
CONFIG_DIR = BENCHMARK_DIR / "configs"
PROFILE_DIR = BENCHMARK_DIR / "profiles"
SERVER_PYTHON_BIN = os.environ.get("SERVER_PYTHON_BIN", os.environ.get("PYTHON_BIN", "python"))
ROUTER_PYTHON_BIN = os.environ.get("ROUTER_PYTHON_BIN", os.environ.get("PYTHON_BIN", "python"))
PROXY_PYTHON_BIN = os.environ.get("PROXY_PYTHON_BIN", "python3")
CLIENT_PYTHON_BIN = os.environ.get("CLIENT_PYTHON_BIN", "python3")


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if key == "profile":
            continue
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def case_files() -> list[Path]:
    return sorted(CONFIG_DIR.glob("*.json"))


def case_ids() -> list[str]:
    return [path.stem for path in case_files()]


def load_case(case_id: str) -> dict[str, Any]:
    path = CONFIG_DIR / f"{case_id}.json"
    if not path.exists():
        raise SystemExit(f"Unknown case '{case_id}'. Use 'list' to see valid cases.")

    case = load_json(path)
    profile_name = case.get("profile")
    if not profile_name:
        return case

    profile_path = PROFILE_DIR / f"{profile_name}.json"
    if not profile_path.exists():
        raise SystemExit(f"Missing profile '{profile_name}' for case '{case_id}'.")

    merged = deep_merge(load_json(profile_path), case)
    merged["id"] = case["id"]
    merged["description"] = case.get("description", "")
    merged["profile"] = profile_name
    return normalize_case(merged)


def case_order(case: dict[str, Any]) -> int:
    return int(case["id"].split("_", 1)[0])


def normalize_case(case: dict[str, Any]) -> dict[str, Any]:
    router = case.get("router", {})
    if router.get("enabled"):
        router.setdefault("prometheus_host", router.get("host", "0.0.0.0"))
        router.setdefault("prometheus_port", 29000 + case_order(case))
    return case


def repo_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def resolved_model_path(model: dict[str, Any]) -> str:
    env_name = model.get("env")
    if env_name and os.environ.get(env_name):
        return os.environ[env_name]

    fallback_env = model.get("fallback_env")
    if fallback_env and os.environ.get(fallback_env):
        return os.environ[fallback_env]

    return model["default_path"]


def append_pair(cmd: list[str], flag: str, value: Any) -> None:
    if value is not None:
        cmd.extend([flag, str(value)])


def server_command(case: dict[str, Any]) -> tuple[list[str], dict[str, str]]:
    server = case["server"]
    model_path = resolved_model_path(server["model"])

    if server["model"].get("kind") == "pruned":
        config_path = Path(model_path) / "config.json"
        if not config_path.exists():
            raise RuntimeError(
                f"Pruned model is missing: {config_path}\n"
                "Set PRUNED_MODEL_PATH if the pruned checkpoint is elsewhere."
            )

    cmd = [
        SERVER_PYTHON_BIN,
        "-m",
        "sglang.launch_server",
        "--model-path",
        model_path,
    ]

    append_pair(cmd, "--host", server.get("host"))
    append_pair(cmd, "--port", server.get("port"))
    append_pair(cmd, "--mem-fraction-static", server.get("mem_fraction_static"))
    append_pair(cmd, "--context-length", server.get("context_length"))
    append_pair(cmd, "--reasoning-parser", server.get("reasoning_parser"))
    append_pair(cmd, "--tool-call-parser", server.get("tool_call_parser"))
    append_pair(cmd, "--mamba-scheduler-strategy", server.get("mamba_scheduler_strategy"))
    append_pair(cmd, "--chunked-prefill-size", server.get("chunked_prefill_size"))
    append_pair(cmd, "--schedule-policy", server.get("schedule_policy"))

    if server.get("enable_cache_report"):
        cmd.append("--enable-cache-report")
    if server.get("enable_metrics"):
        cmd.append("--enable-metrics")
    if server.get("enable_mixed_chunk"):
        cmd.append("--enable-mixed-chunk")

    append_pair(cmd, "--schedule-conservativeness", server.get("schedule_conservativeness"))
    append_pair(cmd, "--kv-cache-dtype", server.get("kv_cache_dtype"))

    hicache = server.get("hicache", {})
    if hicache.get("enabled"):
        cmd.append("--enable-hierarchical-cache")
        append_pair(cmd, "--hicache-ratio", hicache.get("ratio"))
        append_pair(cmd, "--hicache-io-backend", hicache.get("io_backend"))
        append_pair(cmd, "--hicache-storage-prefetch-policy", hicache.get("storage_prefetch_policy"))
        append_pair(cmd, "--hicache-write-policy", hicache.get("write_policy"))

    append_pair(cmd, "--tp-size", server.get("tp_size"))
    append_pair(cmd, "--dp-size", server.get("dp_size"))
    if server.get("enable_dp_attention"):
        cmd.append("--enable-dp-attention")

    env = os.environ.copy()
    env["SGLANG_ENABLE_SPEC_V2"] = str(server.get("sglang_enable_spec_v2", "0"))
    if server.get("speculative_algorithm"):
        env["SPECULATIVE_ALGO"] = str(server["speculative_algorithm"])
    return cmd, env


def router_command(case: dict[str, Any]) -> list[str]:
    server = case["server"]
    router = case["router"]
    server_url = f"http://127.0.0.1:{server['port']}"

    cmd = [
        ROUTER_PYTHON_BIN,
        "-m",
        "sglang_router.launch_router",
        "--worker-urls",
        server_url,
        "--policy",
        router["policy"],
        "--host",
        router["host"],
        "--port",
        str(router["port"]),
        "--prometheus-host",
        router["prometheus_host"],
        "--prometheus-port",
        str(router["prometheus_port"]),
    ]
    if router.get("dp_aware"):
        cmd.append("--dp-aware")
    return cmd


def proxy_command(case: dict[str, Any], run_dir: Path) -> list[str]:
    server = case["server"]
    router = case["router"]
    proxy = case["proxy"]

    cmd = [
        PROXY_PYTHON_BIN,
        str(TEST_ROOT / "proxy" / "main.py"),
        "--host",
        proxy["host"],
        "--port",
        str(proxy["port"]),
        "--direct-url",
        f"http://127.0.0.1:{server['port']}",
        "--router-url",
        f"http://127.0.0.1:{router['port']}",
        "--policy",
        proxy["policy"],
        "--metrics-path",
        str(run_dir / "proxy_metrics.jsonl"),
    ]

    append_pair(cmd, "--max-active-sessions", proxy.get("max_active_sessions"))
    append_pair(cmd, "--session-idle-timeout", proxy.get("session_idle_timeout"))
    if proxy.get("dynamic_admission"):
        cmd.append("--dynamic-admission")
    append_pair(cmd, "--dynamic-min-inflight-requests", proxy.get("dynamic_min_inflight_requests"))
    append_pair(cmd, "--dynamic-initial-inflight-requests", proxy.get("dynamic_initial_inflight_requests"))
    append_pair(cmd, "--dynamic-max-inflight-requests", proxy.get("dynamic_max_inflight_requests"))
    append_pair(cmd, "--dynamic-control-interval", proxy.get("dynamic_control_interval"))
    return cmd


def is_port_open(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)
        return sock.connect_ex((host, port)) == 0


def ensure_ports_free(case: dict[str, Any]) -> None:
    checks = [("server", case["server"]["port"])]
    router = case.get("router", {})
    if router.get("enabled"):
        checks.append(("router", router["port"]))
        checks.append(("router prometheus", router["prometheus_port"]))
    proxy = case.get("proxy", {})
    if proxy.get("enabled"):
        checks.append(("proxy", proxy["port"]))

    busy = [(name, port) for name, port in checks if is_port_open(int(port))]
    if busy:
        detail = "\n".join(f"  {name}: 127.0.0.1:{port}" for name, port in busy)
        raise RuntimeError(
            "Required port is already in use. Stop the old process or choose another port.\n"
            f"{detail}"
        )


def client_base_url(case: dict[str, Any]) -> str:
    if case.get("proxy", {}).get("enabled"):
        return f"http://127.0.0.1:{case['proxy']['port']}"
    if case.get("router", {}).get("enabled"):
        return f"http://127.0.0.1:{case['router']['port']}"
    return f"http://127.0.0.1:{case['server']['port']}"


def workload_command(case: dict[str, Any], run_dir: Path) -> list[str]:
    client = case["client"]
    cmd = [
        CLIENT_PYTHON_BIN,
        str(TEST_ROOT / "bench" / "run_workload.py"),
        "--data",
        str(repo_path(client["data_path"])),
        "--base-url",
        client_base_url(case),
        "--output",
        str(run_dir / "client_metrics.jsonl"),
        "--summary-interval",
        str(client["summary_interval"]),
        "--summary-output",
        str(run_dir / "interval_summaries.jsonl"),
    ]

    if os.environ.get("MAX_SESSIONS"):
        append_pair(cmd, "--max-sessions", os.environ["MAX_SESSIONS"])
    if os.environ.get("MAX_DURATION"):
        append_pair(cmd, "--max-duration", os.environ["MAX_DURATION"])
    if os.environ.get("NO_IGNORE_EOS") == "1":
        cmd.append("--no-ignore-eos")
    return cmd


class ManagedProcess:
    def __init__(self, name: str, cmd: list[str], log_path: Path, env: dict[str, str] | None = None):
        self.name = name
        self.cmd = cmd
        self.log_path = log_path
        self.env = env
        self.proc: subprocess.Popen[bytes] | None = None
        self.log_file = None

    def start(self) -> None:
        self.log_file = self.log_path.open("wb")
        self.proc = subprocess.Popen(
            self.cmd,
            cwd=REPO_ROOT,
            env=self.env,
            stdout=self.log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    @property
    def pid(self) -> int:
        if self.proc is None:
            return 0
        return self.proc.pid

    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def stop(self) -> None:
        if self.proc is None:
            return
        if self.running():
            print(f"[bench] stopping {self.name} pid={self.pid}", flush=True)
            try:
                os.killpg(self.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(self.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self.proc.wait(timeout=10)
        if self.log_file is not None:
            self.log_file.close()


def tail(path: Path, lines: int = 200) -> str:
    if not path.exists():
        return ""
    data = path.read_text(errors="replace").splitlines()
    return "\n".join(data[-lines:])


def http_status_200(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            return response.status == 200
    except Exception:
        return False


def server_ready(base_url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{base_url.rstrip('/')}/v1/models", timeout=5) as response:
            if response.status != 200:
                return False
            json.load(response)
            return True
    except Exception:
        return False


def service_ready(base_url: str) -> bool:
    return http_status_200(f"{base_url.rstrip('/')}/health")


def wait_ready(
    process: ManagedProcess,
    base_url: str,
    timeout_s: int,
    poll_s: int,
    ready_check,
) -> None:
    deadline = time.time() + timeout_s
    print(f"[bench] waiting for {process.name}: {base_url}", flush=True)
    while time.time() < deadline:
        if not process.running():
            raise RuntimeError(
                f"{process.name} exited before it was ready.\n"
                f"Last log lines:\n{tail(process.log_path)}"
            )
        if ready_check(base_url):
            print(f"[bench] {process.name} is ready", flush=True)
            return
        time.sleep(poll_s)
    raise RuntimeError(
        f"timed out waiting for {process.name}.\n"
        f"Last log lines:\n{tail(process.log_path)}"
    )


def write_summary(run_dir: Path, case: dict[str, Any]) -> None:
    summary = None
    metrics_path = run_dir / "client_metrics.jsonl"
    with metrics_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if obj.get("type") == "summary":
                summary = obj
    if summary is None:
        raise RuntimeError(f"no summary found in {metrics_path}")

    summary.update(
        {
            "run_id": case["id"],
            "config": case["id"],
            "base_url": client_base_url(case),
        }
    )
    with (run_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
        f.write("\n")


def write_commands(run_dir: Path, commands: dict[str, Any]) -> None:
    with (run_dir / "commands.json").open("w") as f:
        json.dump(commands, f, indent=2, ensure_ascii=False)
        f.write("\n")


def run_workload(cmd: list[str], log_path: Path) -> None:
    with log_path.open("w") as log:
        proc = subprocess.Popen(
            cmd,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
        code = proc.wait()
    if code != 0:
        raise subprocess.CalledProcessError(code, cmd)


def choose_run_dir(output_root: Path, case_id: str) -> Path:
    if os.environ.get("BENCH_RUN_NAME"):
        return output_root / os.environ["BENCH_RUN_NAME"]

    base = output_root / case_id
    if not base.exists() or not any(base.iterdir()):
        return base

    return output_root / f"{case_id}_{time.strftime('%Y%m%d_%H%M%S')}"


def run_case(case_id: str, output_root: Path) -> None:
    case = load_case(case_id)
    run_dir = choose_run_dir(output_root, case["id"])
    run_dir.mkdir(parents=True, exist_ok=True)

    with (run_dir / "resolved_config.json").open("w") as f:
        json.dump(case, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"[bench] case: {case['id']}", flush=True)
    print(f"[bench] {case.get('description', '')}", flush=True)
    print(f"[bench] output: {run_dir}", flush=True)

    ensure_ports_free(case)

    server_cmd, server_env = server_command(case)
    router_cmd = router_command(case) if case.get("router", {}).get("enabled") else None
    proxy_cmd = proxy_command(case, run_dir) if case.get("proxy", {}).get("enabled") else None
    client_cmd = workload_command(case, run_dir)
    write_commands(
        run_dir,
        {
            "server": server_cmd,
            "server_env": {
                "SGLANG_ENABLE_SPEC_V2": server_env.get("SGLANG_ENABLE_SPEC_V2"),
                "SPECULATIVE_ALGO": server_env.get("SPECULATIVE_ALGO"),
                "PYTHONPATH": server_env.get("PYTHONPATH"),
            },
            "router": router_cmd,
            "proxy": proxy_cmd,
            "client": client_cmd,
        },
    )

    processes: list[ManagedProcess] = []

    try:
        server = ManagedProcess("server", server_cmd, run_dir / "server.log", server_env)
        processes.append(server)
        server.start()
        (run_dir / "server.pid").write_text(f"{server.pid}\n")
        wait_ready(
            server,
            f"http://127.0.0.1:{case['server']['port']}",
            int(os.environ.get("SERVER_READY_TIMEOUT", "1800")),
            int(os.environ.get("READY_POLL_INTERVAL", "5")),
            server_ready,
        )

        if case.get("router", {}).get("enabled"):
            assert router_cmd is not None
            router = ManagedProcess("router", router_cmd, run_dir / "router.log")
            processes.append(router)
            router.start()
            (run_dir / "router.pid").write_text(f"{router.pid}\n")
            wait_ready(
                router,
                f"http://127.0.0.1:{case['router']['port']}",
                int(os.environ.get("ROUTER_READY_TIMEOUT", "300")),
                int(os.environ.get("READY_POLL_INTERVAL", "5")),
                service_ready,
            )

        if case.get("proxy", {}).get("enabled"):
            assert proxy_cmd is not None
            proxy = ManagedProcess("proxy", proxy_cmd, run_dir / "proxy.log")
            processes.append(proxy)
            proxy.start()
            (run_dir / "proxy.pid").write_text(f"{proxy.pid}\n")
            wait_ready(
                proxy,
                f"http://127.0.0.1:{case['proxy']['port']}",
                int(os.environ.get("PROXY_READY_TIMEOUT", "120")),
                int(os.environ.get("READY_POLL_INTERVAL", "5")),
                service_ready,
            )

        client_log = run_dir / "client_console.log"
        print(f"[bench] running workload: {client_base_url(case)}", flush=True)
        run_workload(client_cmd, client_log)
        write_summary(run_dir, case)
        print(f"[bench] summary: {run_dir / 'summary.json'}", flush=True)
    finally:
        for process in reversed(processes):
            process.stop()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run qiannan benchmark cases.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list")
    run_one = sub.add_parser("run")
    run_one.add_argument("case_id")
    run_all = sub.add_parser("all")
    run_all.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    if args.command == "list":
        print("\n".join(case_ids()))
        return 0

    output_root = Path(os.environ.get("BENCH_OUTPUT_ROOT", str(TEST_ROOT / "benchmark_runs")))

    if args.command == "run":
        run_case(args.case_id, output_root)
        return 0

    failed: list[str] = []
    for case_id in case_ids():
        print(f"\n[bench] ===== {case_id} =====", flush=True)
        try:
            run_case(case_id, output_root)
        except Exception as exc:
            print(f"[bench] FAIL {case_id}: {exc}", file=sys.stderr, flush=True)
            failed.append(case_id)
            if not args.continue_on_error:
                break

    if failed:
        print("[bench] failed cases:", file=sys.stderr)
        for case_id in failed:
            print(f"  {case_id}", file=sys.stderr)
        return 1

    print(f"\n[bench] all cases completed: {output_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
