#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
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
CLEANUP_GRACE_SECONDS = float(os.environ.get("BENCH_CLEANUP_GRACE_SECONDS", "10"))


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


def default_config_dir() -> Path:
    return Path(os.environ.get("BENCH_CONFIG_DIR", str(CONFIG_DIR)))


def default_output_root() -> Path:
    return Path(os.environ.get("BENCH_OUTPUT_ROOT", str(TEST_ROOT / "benchmark_runs")))


def case_files(config_dir: Path) -> list[Path]:
    return sorted(config_dir.glob("*.json"))


def case_ids(config_dir: Path) -> list[str]:
    return [path.stem for path in case_files(config_dir)]


def load_case(case_id: str, config_dir: Path) -> dict[str, Any]:
    path = config_dir / f"{case_id}.json"
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
    prefix = case["id"].split("_", 1)[0]
    match = re.search(r"\d+", prefix)
    if match is None:
        raise ValueError(f"case id must contain an ordering number: {case['id']}")
    return int(match.group(0))


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


def path_is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


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
    append_pair(cmd, "--max-running-requests", server.get("max_running_requests"))
    append_pair(cmd, "--max-prefill-tokens", server.get("max_prefill_tokens"))
    append_pair(cmd, "--prefill-max-requests", server.get("prefill_max_requests"))
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
    append_pair(cmd, "--dynamic-min-active-sessions", proxy.get("dynamic_min_active_sessions"))
    append_pair(cmd, "--dynamic-initial-active-sessions", proxy.get("dynamic_initial_active_sessions"))
    append_pair(cmd, "--dynamic-max-active-sessions", proxy.get("dynamic_max_active_sessions"))
    append_pair(cmd, "--dynamic-min-inflight-requests", proxy.get("dynamic_min_inflight_requests"))
    append_pair(cmd, "--dynamic-initial-inflight-requests", proxy.get("dynamic_initial_inflight_requests"))
    append_pair(cmd, "--dynamic-max-inflight-requests", proxy.get("dynamic_max_inflight_requests"))
    append_pair(cmd, "--dynamic-control-interval", proxy.get("dynamic_control_interval"))
    append_pair(cmd, "--dynamic-cache-hit-threshold", proxy.get("dynamic_cache_hit_threshold"))
    append_pair(cmd, "--dynamic-low-kv-usage", proxy.get("dynamic_low_kv_usage"))
    append_pair(cmd, "--dynamic-high-kv-usage", proxy.get("dynamic_high_kv_usage"))
    append_pair(cmd, "--dynamic-additive-step", proxy.get("dynamic_additive_step"))
    append_pair(cmd, "--dynamic-decrease-factor", proxy.get("dynamic_decrease_factor"))
    append_pair(cmd, "--session-pause-min-secs", proxy.get("session_pause_min_secs"))
    append_pair(cmd, "--server-metrics-url", proxy.get("server_metrics_url"))
    append_pair(cmd, "--server-metrics-interval", proxy.get("server_metrics_interval"))
    return cmd


def is_port_open(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)
        return sock.connect_ex((host, port)) == 0


def required_ports(case: dict[str, Any]) -> list[tuple[str, int]]:
    checks = [("server", case["server"]["port"])]
    router = case.get("router", {})
    if router.get("enabled"):
        checks.append(("router", router["port"]))
        checks.append(("router prometheus", router["prometheus_port"]))
    proxy = case.get("proxy", {})
    if proxy.get("enabled"):
        checks.append(("proxy", proxy["port"]))
    return [(name, int(port)) for name, port in checks]


def ensure_ports_free(case: dict[str, Any]) -> None:
    checks = required_ports(case)

    busy = [(name, port) for name, port in checks if is_port_open(int(port))]
    if busy:
        detail = "\n".join(f"  {name}: 127.0.0.1:{port}" for name, port in busy)
        raise RuntimeError(
            "Required port is still in use after pre-run cleanup. "
            "Stop the old process or choose another port.\n"
            f"{detail}"
        )


def run_probe(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def parse_pid_lines(output: str) -> set[int]:
    pids: set[int] = set()
    for token in output.replace(",", " ").split():
        if token.isdigit():
            pids.add(int(token))
    return pids


def pids_for_port(port: int) -> set[int]:
    lsof = shutil.which("lsof")
    if lsof is not None:
        result = run_probe([lsof, f"-tiTCP:{port}", "-sTCP:LISTEN"])
        if result.returncode == 0:
            return parse_pid_lines(result.stdout)

    fuser = shutil.which("fuser")
    if fuser is not None:
        result = run_probe([fuser, "-n", "tcp", str(port)])
        if result.returncode == 0:
            return parse_pid_lines(result.stdout)

    return set()


def pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def pid_is_current_user(pid: int) -> bool:
    proc_path = Path("/proc") / str(pid)
    if not proc_path.exists() or not hasattr(os, "getuid"):
        return True
    try:
        return proc_path.stat().st_uid == os.getuid()
    except FileNotFoundError:
        return False
    except OSError:
        return True


def read_pid_cmdline(pid: int) -> str:
    cmdline_path = Path("/proc") / str(pid) / "cmdline"
    try:
        data = cmdline_path.read_bytes()
        if data:
            return " ".join(part.decode(errors="replace") for part in data.split(b"\0") if part)
    except OSError:
        pass

    ps = shutil.which("ps")
    if ps is None:
        return ""
    result = run_probe([ps, "-p", str(pid), "-o", "command="])
    if result.returncode == 0:
        return result.stdout.strip()
    return ""


def read_pid_cwd(pid: int) -> Path | None:
    try:
        return Path(os.readlink(Path("/proc") / str(pid) / "cwd")).resolve()
    except OSError:
        return None


def describe_pid(pid: int) -> str:
    cmdline = read_pid_cmdline(pid)
    if cmdline:
        return cmdline
    return f"pid={pid}"


def benchmark_related_pid(pid: int) -> bool:
    cmdline = read_pid_cmdline(pid)
    markers = (
        str(REPO_ROOT),
        "sglang.launch_server",
        "sglang_router.launch_router",
        "run_benchmark.py",
        "run_workload.py",
    )
    if any(marker in cmdline for marker in markers):
        return True

    cwd = read_pid_cwd(pid)
    return cwd is not None and path_is_relative_to(cwd, REPO_ROOT)


def wait_for_pids_to_exit(pids: set[int], timeout_s: float) -> set[int]:
    deadline = time.time() + timeout_s
    remaining = {pid for pid in pids if pid_exists(pid)}
    while remaining and time.time() < deadline:
        time.sleep(0.2)
        remaining = {pid for pid in remaining if pid_exists(pid)}
    return remaining


def kill_pids(pids: set[int], reason: str) -> None:
    current_pid = os.getpid()
    current_pgid = os.getpgrp()
    targets = {
        pid
        for pid in pids
        if pid != current_pid and pid_exists(pid) and pid_is_current_user(pid)
    }
    if not targets:
        return

    pgids: set[int] = set()
    direct_pids: set[int] = set()
    for pid in sorted(targets):
        try:
            pgid = os.getpgid(pid)
        except OSError:
            continue
        if pgid == current_pgid:
            direct_pids.add(pid)
        else:
            pgids.add(pgid)

    for pid in sorted(targets):
        print(f"[bench] cleanup: killing {reason}: pid={pid} cmd={describe_pid(pid)}", flush=True)

    for pgid in sorted(pgids):
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except PermissionError:
            direct_pids.update(pid for pid in targets if pid_exists(pid))

    for pid in sorted(direct_pids):
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

    remaining = wait_for_pids_to_exit(targets, CLEANUP_GRACE_SECONDS)
    if not remaining:
        return

    remaining_pgids: set[int] = set()
    for pid in sorted(remaining):
        if not pid_exists(pid):
            continue
        try:
            pgid = os.getpgid(pid)
        except OSError:
            continue
        if pgid != current_pgid:
            remaining_pgids.add(pgid)

    for pgid in sorted(remaining_pgids):
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except PermissionError:
            pass

    for pid in sorted(remaining):
        if not pid_exists(pid):
            continue
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except PermissionError:
            print(f"[bench] cleanup: cannot kill pid={pid}; permission denied", flush=True)


def cleanup_port_processes(case: dict[str, Any]) -> None:
    for name, port in required_ports(case):
        pids = pids_for_port(port)
        if pids:
            kill_pids(pids, f"{name} port 127.0.0.1:{port}")


def gpu_process_pids() -> set[int]:
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return set()

    result = run_probe(
        [
            nvidia_smi,
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    if result.returncode != 0:
        return set()

    pids: set[int] = set()
    for line in result.stdout.splitlines():
        first = line.split(",", 1)[0].strip()
        if first.isdigit():
            pids.add(int(first))
    return pids


def cleanup_gpu_processes() -> None:
    pids = gpu_process_pids()
    if not pids:
        return

    if os.environ.get("BENCH_ONLY_KILL_BENCHMARK_GPU_PROCESSES") == "1":
        targets = {
            pid
            for pid in pids
            if pid_is_current_user(pid) and benchmark_related_pid(pid)
        }
    else:
        targets = {pid for pid in pids if pid_is_current_user(pid)}

    if targets:
        kill_pids(targets, "stale GPU process")


def cleanup_before_run(case: dict[str, Any]) -> None:
    if os.environ.get("BENCH_SKIP_PRE_RUN_CLEANUP") == "1":
        ensure_ports_free(case)
        return

    cleanup_port_processes(case)
    cleanup_gpu_processes()
    ensure_ports_free(case)


def client_base_url(case: dict[str, Any]) -> str:
    if case.get("proxy", {}).get("enabled"):
        return f"http://127.0.0.1:{case['proxy']['port']}"
    if case.get("router", {}).get("enabled"):
        return f"http://127.0.0.1:{case['router']['port']}"
    return f"http://127.0.0.1:{case['server']['port']}"


def direct_server_base_url(case: dict[str, Any]) -> str:
    return f"http://127.0.0.1:{case['server']['port']}"


def client_enabled(case: dict[str, Any]) -> bool:
    return case.get("client", {}).get("enabled", True)


def loss_enabled(case: dict[str, Any]) -> bool:
    return case.get("loss", {}).get("enabled", False)


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


def loss_base_url(case: dict[str, Any]) -> str:
    mode = case.get("loss", {}).get("base_url_mode", "direct")
    if mode == "direct":
        return direct_server_base_url(case)
    if mode == "client":
        return client_base_url(case)
    raise ValueError(f"unknown loss.base_url_mode: {mode}")


def loss_command(case: dict[str, Any], run_dir: Path) -> list[str]:
    loss = case["loss"]
    cmd = [
        CLIENT_PYTHON_BIN,
        str(BENCHMARK_DIR / "run_loss.py"),
        "--data",
        str(repo_path(loss["data_path"])),
        "--base-url",
        loss_base_url(case),
        "--output",
        str(run_dir / loss.get("output", "loss_metrics.jsonl")),
        "--summary-output",
        str(run_dir / loss.get("summary_output", "loss_summary.json")),
        "--label",
        str(loss.get("label", case["id"])),
        "--concurrency",
        str(loss.get("concurrency", 16)),
        "--timeout",
        str(loss.get("timeout", 900)),
        "--retries",
        str(loss.get("retries", 3)),
    ]

    append_pair(cmd, "--limit", loss.get("limit"))
    append_pair(cmd, "--per-domain", loss.get("per_domain"))
    domains = loss.get("domains")
    if isinstance(domains, list):
        domains = ",".join(str(x) for x in domains)
    append_pair(cmd, "--domains", domains)
    append_pair(cmd, "--bootstrap-samples", loss.get("bootstrap_samples"))
    append_pair(cmd, "--seed", loss.get("seed"))
    if loss.get("allow_errors"):
        cmd.append("--allow-errors")
    return cmd


def loads_sampler_config(case: dict[str, Any], run_dir: Path) -> dict[str, Any] | None:
    configured = case.get("samplers", {}).get("loads", {})
    if not isinstance(configured, dict):
        configured = {}

    enabled = bool(configured.get("enabled", False))
    if os.environ.get("BENCH_SAMPLE_LOADS") == "1":
        enabled = True
    if not enabled:
        return None

    include = configured.get("include", "all")
    if isinstance(include, list):
        include = ",".join(str(item) for item in include)

    interval = float(
        os.environ.get("BENCH_LOAD_SAMPLE_INTERVAL", configured.get("interval", 5.0))
    )
    timeout = float(configured.get("timeout", 2.0))
    output = configured.get("output", "load_metrics.jsonl")
    url = f"{direct_server_base_url(case).rstrip('/')}/v1/loads?include={include}"
    return {
        "enabled": True,
        "url": url,
        "interval": interval,
        "timeout": timeout,
        "output": str(run_dir / output),
        "include": include,
    }


class LoadMetricsSampler:
    def __init__(self, case_id: str, config: dict[str, Any]):
        self.case_id = case_id
        self.url = str(config["url"])
        self.interval = float(config["interval"])
        self.timeout = float(config["timeout"])
        self.output_path = Path(config["output"])
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.output_file = None
        self.t0 = 0.0
        self.sample_index = 0

    def start(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_file = self.output_path.open("w", buffering=1)
        self.t0 = time.monotonic()
        self._write(
            {
                "type": "load_sampler_start",
                "case_id": self.case_id,
                "t": round(time.time(), 6),
                "url": self.url,
                "interval": self.interval,
                "timeout": self.timeout,
            }
        )
        self.sample_once()
        self.thread = threading.Thread(
            target=self._run,
            name=f"load-metrics-sampler-{self.case_id}",
            daemon=True,
        )
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=max(self.timeout + 1.0, 2.0))
        self._write(
            {
                "type": "load_sampler_stop",
                "case_id": self.case_id,
                "t": round(time.time(), 6),
                "elapsed_secs": round(time.monotonic() - self.t0, 3),
                "samples": self.sample_index,
            }
        )
        if self.output_file is not None:
            self.output_file.close()
            self.output_file = None

    def _run(self) -> None:
        while not self.stop_event.wait(self.interval):
            self.sample_once()

    def sample_once(self) -> None:
        self.sample_index += 1
        now = time.time()
        elapsed = time.monotonic() - self.t0
        base_record = {
            "case_id": self.case_id,
            "sample_index": self.sample_index,
            "t": round(now, 6),
            "elapsed_secs": round(elapsed, 3),
            "url": self.url,
        }
        try:
            with urllib.request.urlopen(self.url, timeout=self.timeout) as response:
                payload = json.load(response)
            self._write_load_records(base_record, payload)
        except Exception as exc:
            self._write(
                {
                    **base_record,
                    "type": "load_error",
                    "error": str(exc),
                }
            )

    def _write_load_records(self, base_record: dict[str, Any], payload: dict[str, Any]) -> None:
        aggregate = payload.get("aggregate")
        if isinstance(aggregate, dict):
            self._write(
                {
                    **base_record,
                    "type": "load_aggregate",
                    "server_timestamp": payload.get("timestamp"),
                    "version": payload.get("version"),
                    "dp_rank_count": payload.get("dp_rank_count"),
                    **aggregate,
                }
            )

        loads = payload.get("loads") or []
        for load in loads:
            if not isinstance(load, dict):
                continue
            self._write(
                {
                    **base_record,
                    "type": "load",
                    "server_timestamp": payload.get("timestamp"),
                    **load,
                }
            )

    def _write(self, obj: dict[str, Any]) -> None:
        if self.output_file is None:
            return
        self.output_file.write(json.dumps(obj, ensure_ascii=False) + "\n")


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


def run_case(case_id: str, output_root: Path, config_dir: Path) -> None:
    case = load_case(case_id, config_dir)
    run_dir = choose_run_dir(output_root, case["id"])
    run_dir.mkdir(parents=True, exist_ok=True)

    with (run_dir / "resolved_config.json").open("w") as f:
        json.dump(case, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"[bench] case: {case['id']}", flush=True)
    print(f"[bench] {case.get('description', '')}", flush=True)
    print(f"[bench] output: {run_dir}", flush=True)

    cleanup_before_run(case)

    server_cmd, server_env = server_command(case)
    router_cmd = router_command(case) if case.get("router", {}).get("enabled") else None
    proxy_cmd = proxy_command(case, run_dir) if case.get("proxy", {}).get("enabled") else None
    client_cmd = workload_command(case, run_dir) if client_enabled(case) else None
    loss_cmd = loss_command(case, run_dir) if loss_enabled(case) else None
    load_sampler_cfg = loads_sampler_config(case, run_dir)
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
            "loss": loss_cmd,
            "loads_sampler": load_sampler_cfg,
        },
    )

    processes: list[ManagedProcess] = []
    samplers: list[LoadMetricsSampler] = []

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

        if load_sampler_cfg is not None:
            load_sampler = LoadMetricsSampler(case["id"], load_sampler_cfg)
            samplers.append(load_sampler)
            print(f"[bench] sampling loads: {load_sampler.output_path}", flush=True)
            load_sampler.start()

        if client_cmd is not None:
            client_log = run_dir / "client_console.log"
            print(f"[bench] running workload: {client_base_url(case)}", flush=True)
            run_workload(client_cmd, client_log)
            write_summary(run_dir, case)
            print(f"[bench] summary: {run_dir / 'summary.json'}", flush=True)

        if loss_cmd is not None:
            loss_log = run_dir / "loss_console.log"
            print(f"[bench] running loss scoring: {loss_base_url(case)}", flush=True)
            run_workload(loss_cmd, loss_log)
            print(f"[bench] loss summary: {run_dir / case['loss'].get('summary_output', 'loss_summary.json')}", flush=True)
    finally:
        for sampler in reversed(samplers):
            sampler.stop()
        for process in reversed(processes):
            process.stop()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run qiannan benchmark cases.")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common_args(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument(
            "--config-dir",
            type=Path,
            default=default_config_dir(),
            help="Directory containing case JSON files.",
        )
        subparser.add_argument(
            "--output-root",
            type=Path,
            default=default_output_root(),
            help="Directory for benchmark run outputs.",
        )

    list_cmd = sub.add_parser("list")
    add_common_args(list_cmd)
    run_one = sub.add_parser("run")
    add_common_args(run_one)
    run_one.add_argument("case_id")
    run_all = sub.add_parser("all")
    add_common_args(run_all)
    run_all.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    if args.command == "list":
        print("\n".join(case_ids(args.config_dir)))
        return 0

    if args.command == "run":
        run_case(args.case_id, args.output_root, args.config_dir)
        return 0

    failed: list[str] = []
    for case_id in case_ids(args.config_dir):
        print(f"\n[bench] ===== {case_id} =====", flush=True)
        try:
            run_case(case_id, args.output_root, args.config_dir)
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

    print(f"\n[bench] all cases completed: {args.output_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
