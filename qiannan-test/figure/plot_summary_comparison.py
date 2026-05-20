#!/usr/bin/env python3
"""Plot a before/after dashboard from two benchmark summary JSON files.

Default usage from the repository root:

    python3 qiannan-test/figure/plot_summary_comparison.py

The generated SVG is dependency-free and is written under
qiannan-test/figure/summary_compare/ by default.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from plot_interval_metrics import (
    AXIS,
    BG,
    GRID,
    MUTED,
    TEXT,
    compact,
    line,
    make_scaler,
    rect,
    svg_text,
    tick_values,
)


SCRIPT_DIR = Path(__file__).resolve().parent
RUNS_DIR = SCRIPT_DIR.parent / "runs"
DEFAULT_BASELINE = RUNS_DIR / "dp2tp8_hicache_summary.json"
DEFAULT_ADMISSION = RUNS_DIR / "02_admission64_dp2tp8_summary"
DEFAULT_OUT = SCRIPT_DIR / "summary_compare" / "admission_control_optimization.svg"

GREEN = "#059669"
RED = "#dc2626"
BLUE = "#2563eb"
AMBER = "#d97706"
LIGHT = "#f9fafb"


@dataclass(frozen=True)
class Kpi:
    label: str
    before: float
    after: float
    unit: str
    higher_is_better: bool
    formatter: Callable[[float], str]


@dataclass(frozen=True)
class RatioRow:
    label: str
    value: float
    detail: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two SGLang summary JSON files as one SVG dashboard."
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=DEFAULT_BASELINE,
        help=f"baseline summary JSON (default: {DEFAULT_BASELINE})",
    )
    parser.add_argument(
        "--admission",
        type=Path,
        default=DEFAULT_ADMISSION,
        help=f"admission-control summary JSON (default: {DEFAULT_ADMISSION})",
    )
    parser.add_argument(
        "--baseline-label",
        default="No admission control",
        help="label for the baseline run",
    )
    parser.add_argument(
        "--admission-label",
        default="Admission control",
        help="label for the admission-control run",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUT,
        help=f"output SVG path (default: {DEFAULT_OUT})",
    )
    return parser.parse_args()


def load_summary(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    if data.get("type") not in (None, "summary"):
        raise ValueError(f"{path} is not a summary JSON object")
    return data


def to_float(value: Any) -> float:
    if value is None or isinstance(value, bool):
        raise ValueError("missing numeric value")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite numeric value: {value!r}")
    return number


def get_path(record: dict[str, Any], *keys: str) -> float:
    current: Any = record
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            raise KeyError(".".join(keys))
        current = current[key]
    return to_float(current)


def fmt_plain(value: float) -> str:
    return compact(value)


def fmt_seconds(value: float) -> str:
    return f"{compact(value)}s"


def fmt_rate(value: float) -> str:
    return compact(value)


def fmt_pct_value(value: float) -> str:
    return f"{value * 100.0:.1f}%"


def fmt_pct_points(value: float) -> str:
    return f"{value:.1f} pp"


def fmt_factor(value: float) -> str:
    return f"{value:.2f}x"


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(SCRIPT_DIR.parent.parent))
    except ValueError:
        return str(path)


def pct_change(before: float, after: float) -> float | None:
    if before == 0:
        return None
    return (after - before) / before * 100.0


def reduction_pct(before: float, after: float) -> float:
    if before <= 0:
        return 0.0
    return max(0.0, (before - after) / before * 100.0)


def ratio(after: float, before: float) -> float:
    if before <= 0:
        return 0.0
    return after / before


def speedup(before: float, after: float) -> float:
    if after <= 0:
        return 0.0
    return before / after


def wrap_svg_text(
    x: float,
    y: float,
    text: str,
    *,
    max_chars: int,
    size: int = 12,
    fill: str = MUTED,
    anchor: str = "start",
    weight: str = "400",
    line_height: int = 16,
) -> list[str]:
    words = text.split()
    if not words:
        return []
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        if len(current) + 1 + len(word) <= max_chars:
            current += " " + word
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return [
        svg_text(x, y + idx * line_height, line, size=size, fill=fill, anchor=anchor, weight=weight)
        for idx, line in enumerate(lines)
    ]


def draw_panel(
    parts: list[str],
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    subtitle: str,
) -> tuple[float, float, float, float]:
    parts.append(rect(x, y, width, height, LIGHT, stroke=GRID, radius=5))
    parts.append(svg_text(x + 18, y + 28, title, size=16, weight="700"))
    if subtitle:
        parts.extend(
            wrap_svg_text(
                x + 18,
                y + 50,
                subtitle,
                max_chars=max(26, int((width - 36) / 7.0)),
                size=12,
                fill=MUTED,
            )
        )
    return (x + 18, y + 74, width - 36, height - 94)


def draw_kpi_cards(
    parts: list[str],
    x: float,
    y: float,
    width: float,
    baseline_label: str,
    admission_label: str,
    kpis: Sequence[Kpi],
) -> float:
    gap = 14
    cols = 3
    card_width = (width - gap * (cols - 1)) / cols
    card_height = 118
    for idx, kpi in enumerate(kpis):
        col = idx % cols
        row = idx // cols
        cx = x + col * (card_width + gap)
        cy = y + row * (card_height + gap)
        improved = kpi.after >= kpi.before if kpi.higher_is_better else kpi.after <= kpi.before
        color = GREEN if improved else RED
        parts.append(rect(cx, cy, card_width, card_height, "#ffffff", stroke=GRID, radius=5))
        parts.append(svg_text(cx + 14, cy + 24, kpi.label, size=13, fill=MUTED, weight="700"))
        parts.append(svg_text(cx + 14, cy + 58, kpi.formatter(kpi.after), size=24, weight="700", fill=TEXT))

        if kpi.unit:
            parts.append(svg_text(cx + 14, cy + 78, kpi.unit, size=11, fill=MUTED))

        if kpi.label in ("Cache Hit Rate", "Completion Ratio"):
            delta_text = fmt_pct_points((kpi.after - kpi.before) * 100.0)
        elif kpi.higher_is_better:
            change = pct_change(kpi.before, kpi.after)
            delta_text = "n/a" if change is None else f"{change:+.1f}%"
        else:
            delta_text = f"-{reduction_pct(kpi.before, kpi.after):.1f}%"

        parts.append(svg_text(cx + card_width - 14, cy + 58, delta_text, size=13, fill=color, anchor="end", weight="700"))
        parts.append(
            svg_text(
                cx + 14,
                cy + 102,
                f"{baseline_label}: {kpi.formatter(kpi.before)} -> {admission_label}: {kpi.formatter(kpi.after)}",
                size=11,
                fill=MUTED,
            )
        )
    return y + 2 * card_height + gap


def draw_horizontal_ratio_bars(
    parts: list[str],
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    subtitle: str,
    rows: Sequence[RatioRow],
    *,
    x_label: str,
    max_value: float | None = None,
    color: str = BLUE,
    baseline_line: float | None = 1.0,
) -> None:
    px, py, pw, ph = draw_panel(parts, x, y, width, height, title, subtitle)
    label_width = 178
    chart_x = px + label_width
    chart_width = pw - label_width - 10
    row_gap = ph / max(len(rows), 1)
    bar_height = min(28.0, row_gap * 0.48)
    xmax = max_value or max([row.value for row in rows] + [1.0]) * 1.12
    xmax = max(xmax, 1.0)
    x_scale = make_scaler(0.0, xmax, chart_x, chart_x + chart_width)

    tick_count = 6 if xmax >= 5 else 5
    for tick in tick_values(0.0, xmax, tick_count):
        tx = x_scale(tick)
        parts.append(line(tx, py, tx, py + ph - 18, GRID, dash="3 5"))
        parts.append(svg_text(tx, py + ph, compact(tick), size=10, fill=MUTED, anchor="middle"))

    if baseline_line is not None and 0.0 <= baseline_line <= xmax:
        bx = x_scale(baseline_line)
        parts.append(line(bx, py - 4, bx, py + ph - 18, AXIS, width=1.4))
        parts.append(svg_text(bx + 5, py - 8, "baseline", size=10, fill=MUTED))

    for idx, row in enumerate(rows):
        cy = py + idx * row_gap + row_gap * 0.5
        parts.append(svg_text(px, cy + 4, row.label, size=12, fill=TEXT, weight="700"))
        bar_y = cy - bar_height / 2
        bar_w = max(1.0, x_scale(row.value) - chart_x)
        bar_end = chart_x + bar_w
        parts.append(rect(chart_x, bar_y, bar_w, bar_height, color, radius=4))
        if bar_end + 150 > chart_x + chart_width and bar_w > 120:
            label_x = bar_end - 8
            label_anchor = "end"
            value_fill = "#ffffff"
            detail_fill = "#d1fae5" if color == GREEN else "#dbeafe"
        else:
            label_x = bar_end + 8
            label_anchor = "start"
            value_fill = TEXT
            detail_fill = MUTED
        parts.append(svg_text(label_x, cy - 2, fmt_factor(row.value), size=12, fill=value_fill, anchor=label_anchor, weight="700"))
        parts.append(svg_text(label_x, cy + 14, row.detail, size=10, fill=detail_fill, anchor=label_anchor))

    parts.append(svg_text(chart_x + chart_width / 2, py + ph + 22, x_label, size=11, fill=MUTED, anchor="middle"))


def draw_reduction_bars(
    parts: list[str],
    x: float,
    y: float,
    width: float,
    height: float,
    rows: Sequence[RatioRow],
) -> None:
    px, py, pw, ph = draw_panel(
        parts,
        x,
        y,
        width,
        height,
        "Latency Percentile Reduction",
        "Percent reduction after admission control; labels show baseline -> admission seconds.",
    )
    label_width = 120
    chart_x = px + label_width
    chart_width = pw - label_width - 10
    row_gap = ph / max(len(rows), 1)
    bar_height = min(22.0, row_gap * 0.56)
    x_scale = make_scaler(0.0, 100.0, chart_x, chart_x + chart_width)

    for tick in [0, 25, 50, 75, 100]:
        tx = x_scale(float(tick))
        parts.append(line(tx, py, tx, py + ph - 18, GRID, dash="3 5"))
        parts.append(svg_text(tx, py + ph, f"{tick}%", size=10, fill=MUTED, anchor="middle"))

    for idx, row in enumerate(rows):
        cy = py + idx * row_gap + row_gap * 0.5
        parts.append(svg_text(px, cy + 4, row.label, size=12, fill=TEXT, weight="700"))
        bar_y = cy - bar_height / 2
        bar_w = max(1.0, x_scale(row.value) - chart_x)
        parts.append(rect(chart_x, bar_y, bar_w, bar_height, GREEN, radius=4))
        label_x = min(chart_x + chart_width - 4, chart_x + bar_w + 8)
        anchor = "end" if label_x >= chart_x + chart_width - 6 else "start"
        value_fill = "#ffffff" if anchor == "end" and bar_w > 110 else TEXT
        detail_fill = "#d1fae5" if anchor == "end" and bar_w > 110 else MUTED
        parts.append(svg_text(label_x, cy - 1, f"{row.value:.1f}%", size=11, fill=value_fill, anchor=anchor, weight="700"))
        parts.append(svg_text(label_x, cy + 14, row.detail, size=10, fill=detail_fill, anchor=anchor))


def draw_prompt_mix(
    parts: list[str],
    x: float,
    y: float,
    width: float,
    height: float,
    baseline_label: str,
    admission_label: str,
    baseline: dict[str, Any],
    admission: dict[str, Any],
) -> None:
    px, py, pw, ph = draw_panel(
        parts,
        x,
        y,
        width,
        height,
        "Prompt Token Cache Mix",
        "Cached vs uncached prompt-token share in the completed run.",
    )
    bar_x = px + 128
    bar_width = pw - 144
    bar_height = 42
    rows = [
        (baseline_label, baseline),
        (admission_label, admission),
    ]
    for idx, (label, record) in enumerate(rows):
        cached = get_path(record, "total_cached_tokens")
        uncached = get_path(record, "total_uncached_prompt_tokens")
        total = cached + uncached
        row_y = py + idx * 96 + 20
        parts.append(svg_text(px, row_y + 27, label, size=13, fill=TEXT, weight="700"))
        parts.append(svg_text(px, row_y + 47, f"{compact(total)} prompt tokens", size=11, fill=MUTED))
        parts.append(rect(bar_x, row_y, bar_width, bar_height, "#ffffff", stroke=GRID, radius=5))
        if total <= 0:
            continue
        cached_w = bar_width * cached / total
        uncached_w = bar_width - cached_w
        parts.append(rect(bar_x, row_y, cached_w, bar_height, GREEN, radius=5))
        if uncached_w > 0:
            parts.append(rect(bar_x + cached_w, row_y, uncached_w, bar_height, AMBER, radius=0))
        if cached_w >= 88:
            parts.append(svg_text(bar_x + cached_w / 2, row_y + 27, f"{cached / total * 100.0:.1f}% cached", size=12, fill="#ffffff", anchor="middle", weight="700"))
        else:
            parts.append(svg_text(bar_x + cached_w + 8, row_y + 27, f"{cached / total * 100.0:.1f}% cached", size=12, fill=TEXT, weight="700"))
        if uncached_w >= 108:
            parts.append(svg_text(bar_x + cached_w + uncached_w / 2, row_y + 27, f"{uncached / total * 100.0:.1f}% uncached", size=12, fill="#ffffff", anchor="middle", weight="700"))

    legend_y = py + ph - 8
    parts.append(rect(bar_x, legend_y - 12, 16, 12, GREEN, radius=2))
    parts.append(svg_text(bar_x + 24, legend_y, "Cached prompt tokens", size=12, fill=MUTED))
    parts.append(rect(bar_x + 220, legend_y - 12, 16, 12, AMBER, radius=2))
    parts.append(svg_text(bar_x + 244, legend_y, "Uncached prompt tokens", size=12, fill=MUTED))


def render_dashboard(
    baseline: dict[str, Any],
    admission: dict[str, Any],
    baseline_label: str,
    admission_label: str,
    baseline_path: Path,
    admission_path: Path,
    output_path: Path,
) -> None:
    width = 1280
    height = 1580
    margin = 34
    content_width = width - margin * 2
    col_gap = 22
    col_width = (content_width - col_gap) / 2

    kpis = [
        Kpi(
            "Output Throughput",
            get_path(baseline, "output_throughput_tok_s"),
            get_path(admission, "output_throughput_tok_s"),
            "tok/s",
            True,
            fmt_rate,
        ),
        Kpi(
            "Request Throughput",
            get_path(baseline, "request_throughput_req_s"),
            get_path(admission, "request_throughput_req_s"),
            "req/s",
            True,
            fmt_rate,
        ),
        Kpi(
            "Cache Hit Rate",
            get_path(baseline, "cache_hit_rate"),
            get_path(admission, "cache_hit_rate"),
            "cached prompt share",
            True,
            fmt_pct_value,
        ),
        Kpi(
            "Avg TTFT",
            get_path(baseline, "ttft", "avg"),
            get_path(admission, "ttft", "avg"),
            "seconds",
            False,
            fmt_seconds,
        ),
        Kpi(
            "Avg Round Latency",
            get_path(baseline, "round_latency", "avg"),
            get_path(admission, "round_latency", "avg"),
            "seconds",
            False,
            fmt_seconds,
        ),
        Kpi(
            "Completion Ratio",
            get_path(baseline, "completion_ratio"),
            get_path(admission, "completion_ratio"),
            "completed / planned rounds",
            True,
            fmt_pct_value,
        ),
    ]

    higher_rows = [
        RatioRow(
            "Output throughput",
            ratio(get_path(admission, "output_throughput_tok_s"), get_path(baseline, "output_throughput_tok_s")),
            f"{fmt_rate(get_path(baseline, 'output_throughput_tok_s'))} -> {fmt_rate(get_path(admission, 'output_throughput_tok_s'))} tok/s",
        ),
        RatioRow(
            "Request throughput",
            ratio(get_path(admission, "request_throughput_req_s"), get_path(baseline, "request_throughput_req_s")),
            f"{fmt_rate(get_path(baseline, 'request_throughput_req_s'))} -> {fmt_rate(get_path(admission, 'request_throughput_req_s'))} req/s",
        ),
        RatioRow(
            "Cache hit rate",
            ratio(get_path(admission, "cache_hit_rate"), get_path(baseline, "cache_hit_rate")),
            f"{fmt_pct_value(get_path(baseline, 'cache_hit_rate'))} -> {fmt_pct_value(get_path(admission, 'cache_hit_rate'))}",
        ),
        RatioRow(
            "Completion ratio",
            ratio(get_path(admission, "completion_ratio"), get_path(baseline, "completion_ratio")),
            f"{fmt_pct_value(get_path(baseline, 'completion_ratio'))} -> {fmt_pct_value(get_path(admission, 'completion_ratio'))}",
        ),
        RatioRow(
            "Successful rounds",
            ratio(get_path(admission, "successful_rounds"), get_path(baseline, "successful_rounds")),
            f"{compact(get_path(baseline, 'successful_rounds'))} -> {compact(get_path(admission, 'successful_rounds'))}",
        ),
    ]

    lower_rows = [
        RatioRow(
            "Wall time",
            speedup(get_path(baseline, "wall_time"), get_path(admission, "wall_time")),
            f"{fmt_seconds(get_path(baseline, 'wall_time'))} -> {fmt_seconds(get_path(admission, 'wall_time'))}",
        ),
        RatioRow(
            "Avg session time",
            speedup(get_path(baseline, "avg_session_time"), get_path(admission, "avg_session_time")),
            f"{fmt_seconds(get_path(baseline, 'avg_session_time'))} -> {fmt_seconds(get_path(admission, 'avg_session_time'))}",
        ),
        RatioRow(
            "TTFT avg",
            speedup(get_path(baseline, "ttft", "avg"), get_path(admission, "ttft", "avg")),
            f"{fmt_seconds(get_path(baseline, 'ttft', 'avg'))} -> {fmt_seconds(get_path(admission, 'ttft', 'avg'))}",
        ),
        RatioRow(
            "TTFT p99",
            speedup(get_path(baseline, "ttft", "p99"), get_path(admission, "ttft", "p99")),
            f"{fmt_seconds(get_path(baseline, 'ttft', 'p99'))} -> {fmt_seconds(get_path(admission, 'ttft', 'p99'))}",
        ),
        RatioRow(
            "Round latency avg",
            speedup(get_path(baseline, "round_latency", "avg"), get_path(admission, "round_latency", "avg")),
            f"{fmt_seconds(get_path(baseline, 'round_latency', 'avg'))} -> {fmt_seconds(get_path(admission, 'round_latency', 'avg'))}",
        ),
        RatioRow(
            "Round latency p99",
            speedup(get_path(baseline, "round_latency", "p99"), get_path(admission, "round_latency", "p99")),
            f"{fmt_seconds(get_path(baseline, 'round_latency', 'p99'))} -> {fmt_seconds(get_path(admission, 'round_latency', 'p99'))}",
        ),
    ]

    reduction_rows: list[RatioRow] = []
    for metric_label, metric_key in (("TTFT", "ttft"), ("Round", "round_latency")):
        for percentile in ("avg", "p50", "p90", "p99"):
            before = get_path(baseline, metric_key, percentile)
            after = get_path(admission, metric_key, percentile)
            reduction_rows.append(
                RatioRow(
                    f"{metric_label} {percentile}",
                    reduction_pct(before, after),
                    f"{fmt_seconds(before)} -> {fmt_seconds(after)}",
                )
            )

    failed_before = get_path(baseline, "failed_rounds")
    failed_after = get_path(admission, "failed_rounds")
    failed_text = f"Failed rounds: {compact(failed_before)} -> {compact(failed_after)}"
    planned_text = (
        f"Planned rounds: {compact(get_path(admission, 'planned_rounds'))}; "
        f"completed sessions: {compact(get_path(admission, 'completed_sessions'))}/{compact(get_path(admission, 'loaded_sessions'))}"
    )
    subtitle = (
        f"{baseline.get('run_id', baseline_label)} vs {admission.get('run_id', admission_label)}. "
        f"{failed_text}. {planned_text}."
    )

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        rect(0, 0, width, height, BG),
        svg_text(margin, 42, "Admission Control Optimization Summary", size=25, weight="700"),
        svg_text(margin, 68, subtitle, size=13, fill=MUTED),
    ]

    kpi_bottom = draw_kpi_cards(parts, margin, 100, content_width, baseline_label, admission_label, kpis)
    row1_y = kpi_bottom + 28
    draw_horizontal_ratio_bars(
        parts,
        margin,
        row1_y,
        col_width,
        352,
        "Higher Is Better",
        "Admission-control value divided by baseline. Baseline equals 1x.",
        higher_rows,
        x_label="relative multiplier",
        max_value=3.7,
        color=BLUE,
    )
    draw_horizontal_ratio_bars(
        parts,
        margin + col_width + col_gap,
        row1_y,
        col_width,
        352,
        "Lower Is Better",
        "Baseline value divided by admission-control value. Larger bars mean faster/lower latency.",
        lower_rows,
        x_label="speedup / reduction factor",
        max_value=20.0,
        color=GREEN,
        baseline_line=None,
    )

    row2_y = row1_y + 382
    draw_reduction_bars(parts, margin, row2_y, content_width, 438, reduction_rows)

    row3_y = row2_y + 468
    draw_prompt_mix(
        parts,
        margin,
        row3_y,
        content_width,
        290,
        baseline_label,
        admission_label,
        baseline,
        admission,
    )

    foot_y = height - 28
    parts.append(
        svg_text(
            margin,
            foot_y,
            "Source summaries: "
            f"{display_path(baseline_path)} and "
            f"{display_path(admission_path)}",
            size=11,
            fill=MUTED,
        )
    )
    parts.append("</svg>")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    baseline = load_summary(args.baseline)
    admission = load_summary(args.admission)
    render_dashboard(
        baseline,
        admission,
        args.baseline_label,
        args.admission_label,
        args.baseline,
        args.admission,
        args.output,
    )
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
