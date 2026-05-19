#!/usr/bin/env python3
"""Compute per-session request/context lengths from workload_data.jsonl.

The workload stores each session as one JSON object with multiple rounds. During
benchmarking, run_workload.py sends the full conversation history on every round:
system prompt, previous user/assistant messages, and the current user message.

This script uses the numeric token counts in each round. It intentionally
excludes system prompt tokens and chat-template overhead because the workload
does not store tokenizer-exact counts for those strings.
"""

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class SessionLength:
    session_id: int
    rounds: int
    total_input_tokens: int
    total_output_tokens: int
    session_total_tokens: int
    wait_duration_s: float
    max_round_input_tokens: int
    max_round_output_tokens: int
    max_single_round_tokens: int
    max_single_round_idx: int
    max_prompt_tokens: int
    max_prompt_round_idx: int
    max_sequence_tokens: int
    max_sequence_round_idx: int


def analyze_session(obj: dict) -> SessionLength:
    rounds = obj.get("rounds") or []
    session_id = int(obj["id"])

    context_tokens = 0
    total_input = 0
    total_output = 0
    wait_duration = 0.0

    max_round_input = 0
    max_round_output = 0
    max_single_round = 0
    max_single_round_idx = -1
    max_prompt = 0
    max_prompt_round_idx = -1
    max_sequence = 0
    max_sequence_round_idx = -1

    for idx, rd in enumerate(rounds):
        input_tokens = int(rd.get("input", 0) or 0)
        output_tokens = int(rd.get("output", 0) or 0)
        wait_duration += float(rd.get("wait", 0.0) or 0.0)

        # Prompt for this request contains all previous user+assistant tokens
        # plus the current user input. Current output is generated after prompt.
        prompt_tokens = context_tokens + input_tokens
        sequence_tokens = prompt_tokens + output_tokens
        single_round_tokens = input_tokens + output_tokens

        total_input += input_tokens
        total_output += output_tokens

        if input_tokens > max_round_input:
            max_round_input = input_tokens
        if output_tokens > max_round_output:
            max_round_output = output_tokens
        if single_round_tokens > max_single_round:
            max_single_round = single_round_tokens
            max_single_round_idx = idx
        if prompt_tokens > max_prompt:
            max_prompt = prompt_tokens
            max_prompt_round_idx = idx
        if sequence_tokens > max_sequence:
            max_sequence = sequence_tokens
            max_sequence_round_idx = idx

        context_tokens += input_tokens + output_tokens

    return SessionLength(
        session_id=session_id,
        rounds=len(rounds),
        total_input_tokens=total_input,
        total_output_tokens=total_output,
        session_total_tokens=total_input + total_output,
        wait_duration_s=round(wait_duration, 3),
        max_round_input_tokens=max_round_input,
        max_round_output_tokens=max_round_output,
        max_single_round_tokens=max_single_round,
        max_single_round_idx=max_single_round_idx,
        max_prompt_tokens=max_prompt,
        max_prompt_round_idx=max_prompt_round_idx,
        max_sequence_tokens=max_sequence,
        max_sequence_round_idx=max_sequence_round_idx,
    )


def load_lengths(path: Path) -> list[SessionLength]:
    lengths: list[SessionLength] = []
    with path.open() as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            lengths.append(analyze_session(obj))
    return lengths


def percentile(values: list[int], p: float) -> int:
    if not values:
        return 0
    values = sorted(values)
    idx = int(len(values) * p / 100)
    return values[min(idx, len(values) - 1)]


def write_csv(path: Path, rows: list[SessionLength]) -> None:
    fieldnames = list(asdict(rows[0]).keys()) if rows else []
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def print_table(rows: list[SessionLength]) -> None:
    header = (
        "session_id  rounds  max_prompt  prompt_round  "
        "max_sequence  sequence_round  max_single_round  total_tokens"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row.session_id:>10}  "
            f"{row.rounds:>6}  "
            f"{row.max_prompt_tokens:>10,}  "
            f"{row.max_prompt_round_idx:>12}  "
            f"{row.max_sequence_tokens:>12,}  "
            f"{row.max_sequence_round_idx:>14}  "
            f"{row.max_single_round_tokens:>16,}  "
            f"{row.session_total_tokens:>12,}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute the longest request/context length for each session."
    )
    parser.add_argument(
        "data",
        nargs="?",
        type=Path,
        default=Path("workload_data.jsonl"),
        help="workload JSONL path",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="print top N sessions by max_sequence_tokens; ignored with --all",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="print all sessions instead of only the top N",
    )
    parser.add_argument("--csv", type=Path, help="write all per-session rows to CSV")
    parser.add_argument("--json", action="store_true", help="print all rows as JSON")
    args = parser.parse_args()

    rows = load_lengths(args.data)
    if not rows:
        raise SystemExit("no sessions found")

    if args.csv:
        write_csv(args.csv, rows)

    if args.json:
        print(json.dumps([asdict(row) for row in rows], indent=2))
        return

    total_rounds = sum(row.rounds for row in rows)
    sequence_values = [row.max_sequence_tokens for row in rows]
    prompt_values = [row.max_prompt_tokens for row in rows]
    total_values = [row.session_total_tokens for row in rows]

    max_sequence = max(rows, key=lambda row: row.max_sequence_tokens)
    max_prompt = max(rows, key=lambda row: row.max_prompt_tokens)
    max_total = max(rows, key=lambda row: row.session_total_tokens)
    max_rounds = max(rows, key=lambda row: row.rounds)

    print(f"Dataset: {args.data}")
    print(f"Sessions: {len(rows):,}")
    print(f"Rounds / requests: {total_rounds:,}")
    print()
    print("Definitions: token counts exclude system prompt tokens and chat-template overhead.")
    print("  max_prompt: longest prompt sent by a request in that session")
    print("  max_sequence: max_prompt plus that request's expected output tokens")
    print("  max_single_round: current input plus current expected output only")
    print()
    print(
        "Max sequence/session: "
        f"avg={sum(sequence_values) / len(sequence_values):,.1f} "
        f"p50={percentile(sequence_values, 50):,} "
        f"p90={percentile(sequence_values, 90):,} "
        f"p99={percentile(sequence_values, 99):,} "
        f"max={max_sequence.max_sequence_tokens:,} "
        f"(session {max_sequence.session_id}, round {max_sequence.max_sequence_round_idx})"
    )
    print(
        "Max prompt/session:   "
        f"avg={sum(prompt_values) / len(prompt_values):,.1f} "
        f"p50={percentile(prompt_values, 50):,} "
        f"p90={percentile(prompt_values, 90):,} "
        f"p99={percentile(prompt_values, 99):,} "
        f"max={max_prompt.max_prompt_tokens:,} "
        f"(session {max_prompt.session_id}, round {max_prompt.max_prompt_round_idx})"
    )
    print(
        "Total tokens/session: "
        f"avg={sum(total_values) / len(total_values):,.1f} "
        f"p50={percentile(total_values, 50):,} "
        f"p90={percentile(total_values, 90):,} "
        f"p99={percentile(total_values, 99):,} "
        f"max={max_total.session_total_tokens:,} "
        f"(session {max_total.session_id})"
    )
    print(f"Max rounds/session:  {max_rounds.rounds:,} (session {max_rounds.session_id})")
    print()

    rows_to_print = sorted(rows, key=lambda row: row.max_sequence_tokens, reverse=True)
    if args.all:
        print("All sessions by max_sequence_tokens:")
    else:
        rows_to_print = rows_to_print[: args.top]
        print(f"Top {len(rows_to_print)} sessions by max_sequence_tokens:")
    print_table(rows_to_print)

    if args.csv:
        print()
        print(f"Wrote CSV: {args.csv}")


if __name__ == "__main__":
    main()
