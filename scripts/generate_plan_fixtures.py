#!/usr/bin/env python3
"""Generate the six workload fixtures described in document/PLAN.md."""

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "bench" / "workload_data.jsonl"
OUT_DIR = ROOT / "fixtures"
T6_SEED = 20260519


def load_workload(path: Path) -> list[dict]:
    sessions = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                sessions.append(json.loads(line))
    return sessions


def write_jsonl(path: Path, sessions: list[dict]) -> None:
    with path.open("w") as f:
        for session in sessions:
            f.write(json.dumps(session, ensure_ascii=False, separators=(",", ":")) + "\n")


def make_rounds(count: int, input_tokens, output_tokens: int, wait: float) -> list[dict]:
    rounds = []
    for round_idx in range(count):
        value = input_tokens(round_idx) if callable(input_tokens) else input_tokens
        rounds.append({"wait": wait, "input": value, "output": output_tokens})
    return rounds


def make_synthetic_sessions(
    count: int,
    rounds_per_session: int,
    prompt_for_session,
    input_tokens,
    output_tokens: int,
    wait: float,
) -> list[dict]:
    sessions = []
    for idx in range(count):
        sessions.append(
            {
                "id": idx + 1,
                "system_prompt": prompt_for_session(idx),
                "rounds": make_rounds(
                    rounds_per_session,
                    input_tokens(idx) if callable(input_tokens) else input_tokens,
                    output_tokens,
                    wait,
                ),
            }
        )
    return sessions


def unique_t2_prompt(idx: int) -> str:
    digest = hashlib.sha256(f"T2-session-{idx:02d}".encode()).hexdigest()
    return (
        f"{digest} Unique system prompt for fixture T2 session {idx:02d}. "
        "Respond as a neutral assistant. This prompt is intentionally not shared."
    )


def build_t6_official_50(workload: list[dict]) -> list[dict]:
    by_prompt = defaultdict(list)
    for session in workload:
        by_prompt[session["system_prompt"]].append(session)

    rng = random.Random(T6_SEED)
    selected = []
    for prompt in sorted(by_prompt):
        group = by_prompt[prompt]
        if len(group) < 10:
            raise ValueError(f"system_prompt group has fewer than 10 sessions: {len(group)}")
        selected.extend(rng.sample(group, 10))

    return sorted(selected, key=lambda item: item["id"])


def main() -> None:
    workload = load_workload(SOURCE)
    prompts = list(dict.fromkeys(session["system_prompt"] for session in workload))
    if len(prompts) != 5:
        raise ValueError(f"expected 5 system prompts, found {len(prompts)}")

    OUT_DIR.mkdir(exist_ok=True)

    write_jsonl(
        OUT_DIR / "T1_sysprompt_pure.jsonl",
        make_synthetic_sessions(
            count=20,
            rounds_per_session=100,
            prompt_for_session=lambda _: prompts[0],
            input_tokens=1000,
            output_tokens=128,
            wait=0,
        ),
    )

    write_jsonl(
        OUT_DIR / "T2_unique_sysprompt.jsonl",
        make_synthetic_sessions(
            count=50,
            rounds_per_session=20,
            prompt_for_session=unique_t2_prompt,
            input_tokens=1000,
            output_tokens=128,
            wait=0,
        ),
    )

    long_inputs = [10000, 11500, 13000, 14500, 16000]
    write_jsonl(
        OUT_DIR / "T3_burst_long_input.jsonl",
        make_synthetic_sessions(
            count=20,
            rounds_per_session=5,
            prompt_for_session=lambda _: prompts[0],
            input_tokens=lambda session_idx: (
                lambda round_idx: long_inputs[(session_idx + round_idx) % len(long_inputs)]
            ),
            output_tokens=128,
            wait=0,
        ),
    )

    write_jsonl(
        OUT_DIR / "T4_decode_heavy.jsonl",
        make_synthetic_sessions(
            count=20,
            rounds_per_session=10,
            prompt_for_session=lambda idx: prompts[idx % len(prompts)],
            input_tokens=1000,
            output_tokens=1024,
            wait=0,
        ),
    )

    write_jsonl(
        OUT_DIR / "T5_long_wait.jsonl",
        make_synthetic_sessions(
            count=20,
            rounds_per_session=20,
            prompt_for_session=lambda idx: prompts[idx % len(prompts)],
            input_tokens=1000,
            output_tokens=256,
            wait=60,
        ),
    )

    write_jsonl(OUT_DIR / "T6_official_50.jsonl", build_t6_official_50(workload))


if __name__ == "__main__":
    main()
