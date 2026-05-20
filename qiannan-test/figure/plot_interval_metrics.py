#!/usr/bin/env python3
"""Plot interval_summary benchmark metrics as SVG figures.

Default usage from the repository root:

    python3 qiannan-test/figure/plot_interval_metrics.py

The input can be a JSON object, a JSON array, or JSONL with one summary per
line. Figures are written as standalone SVG files and require only the Python
standard library.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR.parent / "runs" / "interval_summaries.jsonl"
DEFAULT_OUT_DIR = SCRIPT_DIR / "out"

BG = "#ffffff"
TEXT = "#111827"
MUTED = "#6b7280"
GRID = "#e5e7eb"
AXIS = "#9ca3af"
PALETTE = [
    "#2563eb",
    "#dc2626",
    "#059669",
    "#d97706",
    "#7c3aed",
    "#0891b2",
]


@dataclass(frozen=True)
class Series:
    label: str
    values: list[float | None]
    color: str


@dataclass(frozen=True)
class LinePanel:
    title: str
    ylabel: str
    series: list[Series]


@dataclass(frozen=True)
class BarPanel:
    title: str
    ylabel: str
    categories: list[str]
    series: list[Series]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot SGLang interval_summary JSON/JSONL metrics as SVG figures."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"input JSON/JSONL file, or '-' for stdin (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "-o",
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"directory for generated SVG files (default: {DEFAULT_OUT_DIR})",
    )
    return parser.parse_args()


def load_records(input_path: Path) -> list[dict[str, Any]]:
    if str(input_path) == "-":
        text = sys.stdin.read()
    else:
        text = input_path.read_text(encoding="utf-8")

    text = text.strip()
    if not text:
        raise ValueError("input is empty")

    if text[0] == "[":
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError("JSON array input must contain objects")
        records = data
    else:
        records = []
        for line_no, line in enumerate(text.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"failed to parse JSON on line {line_no}: {exc}") from exc

    summaries = [r for r in records if isinstance(r, dict)]
    summaries = [
        r
        for r in summaries
        if r.get("type") in (None, "interval_summary") or "interval_delta" in r
    ]
    if not summaries:
        raise ValueError("no interval summary records found")

    return sorted(
        summaries,
        key=lambda r: (
            to_float(r.get("interval_index")) is None,
            to_float(r.get("interval_index")) or to_float(r.get("elapsed_secs")) or 0.0,
        ),
    )


def to_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def get_path(record: dict[str, Any], *keys: str) -> float | None:
    current: Any = record
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return to_float(current)


def values(records: Sequence[dict[str, Any]], *keys: str, scale: float = 1.0) -> list[float | None]:
    result: list[float | None] = []
    for record in records:
        value = get_path(record, *keys)
        result.append(None if value is None else value * scale)
    return result


def compact(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    abs_value = abs(value)
    units = ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K"))
    for scale, suffix in units:
        if abs_value >= scale:
            return trim(value / scale, digits) + suffix
    return trim(value, digits)


def trim(value: float, digits: int = 2) -> str:
    if abs(value) >= 100:
        digits = 1
    if abs(value) >= 1000:
        digits = 0
    text = f"{value:.{digits}f}"
    return text.rstrip("0").rstrip(".")


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f}%"


def percent_value(record: dict[str, Any], *keys: str) -> float | None:
    value = get_path(record, *keys)
    return None if value is None else value * 100.0


def esc(text: Any) -> str:
    return escape(str(text), quote=True)


def svg_text(
    x: float,
    y: float,
    text: str,
    size: int = 13,
    fill: str = TEXT,
    anchor: str = "start",
    weight: str = "400",
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Inter, Arial, sans-serif" '
        f'font-size="{size}" fill="{fill}" text-anchor="{anchor}" '
        f'font-weight="{weight}">{esc(text)}</text>'
    )


def line(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    stroke: str,
    width: float = 1.0,
    dash: str | None = None,
) -> str:
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{stroke}" stroke-width="{width}"{dash_attr}/>'
    )


def rect(
    x: float,
    y: float,
    width: float,
    height: float,
    fill: str,
    stroke: str | None = None,
    radius: float = 0.0,
) -> str:
    stroke_attr = f' stroke="{stroke}" stroke-width="1"' if stroke else ""
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" '
        f'rx="{radius:.1f}" fill="{fill}"{stroke_attr}/>'
    )


def nice_range(data: Sequence[float | None], zero_floor: bool = True) -> tuple[float, float]:
    finite = [v for v in data if v is not None and math.isfinite(v)]
    if not finite:
        return (0.0, 1.0)
    ymin = min(finite)
    ymax = max(finite)
    if zero_floor and ymin >= 0:
        ymin = 0.0
    if ymin == ymax:
        pad = abs(ymax) * 0.1 if ymax else 1.0
        return (ymin - pad, ymax + pad)
    pad = (ymax - ymin) * 0.08
    return (ymin, ymax + pad)


def tick_values(start: float, end: float, count: int = 5) -> list[float]:
    if count <= 1 or start == end:
        return [start]
    return [start + (end - start) * i / (count - 1) for i in range(count)]


def make_scaler(domain_min: float, domain_max: float, range_min: float, range_max: float):
    if domain_min == domain_max:
        domain_min -= 0.5
        domain_max += 0.5

    def scale(value: float) -> float:
        ratio = (value - domain_min) / (domain_max - domain_min)
        return range_min + ratio * (range_max - range_min)

    return scale


def render_line_chart(
    title: str,
    subtitle: str,
    x_values: list[float],
    x_label: str,
    panels: list[LinePanel],
    output_path: Path,
    width: int = 1200,
) -> None:
    left = 86
    right = 34
    top = 76
    panel_height = 270
    bottom = 52
    gap = 24
    height = top + len(panels) * panel_height + (len(panels) - 1) * gap + bottom
    plot_width = width - left - right

    if not x_values:
        raise ValueError("x values are empty")
    xmin, xmax = min(x_values), max(x_values)
    if xmin == xmax:
        xmin -= 0.5
        xmax += 0.5
    x_scale = make_scaler(xmin, xmax, left, left + plot_width)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        rect(0, 0, width, height, BG),
        svg_text(24, 34, title, size=22, weight="700"),
        svg_text(24, 58, subtitle, size=13, fill=MUTED),
    ]

    for panel_idx, panel in enumerate(panels):
        y0 = top + panel_idx * (panel_height + gap)
        plot_top = y0 + 34
        plot_bottom = y0 + panel_height - 45
        plot_height = plot_bottom - plot_top
        all_values: list[float | None] = []
        for series in panel.series:
            all_values.extend(series.values)
        ymin, ymax = nice_range(all_values)
        y_scale = make_scaler(ymin, ymax, plot_bottom, plot_top)

        parts.append(svg_text(left, y0 + 18, panel.title, size=15, weight="700"))
        parts.append(svg_text(24, plot_top - 12, panel.ylabel, size=12, fill=MUTED))

        for tick in tick_values(ymin, ymax):
            y = y_scale(tick)
            parts.append(line(left, y, left + plot_width, y, GRID))
            parts.append(svg_text(left - 10, y + 4, compact(tick), size=11, fill=MUTED, anchor="end"))

        for tick in tick_values(xmin, xmax):
            x = x_scale(tick)
            parts.append(line(x, plot_top, x, plot_bottom, GRID, dash="3 5"))
            parts.append(svg_text(x, plot_bottom + 20, compact(tick), size=11, fill=MUTED, anchor="middle"))

        parts.append(line(left, plot_bottom, left + plot_width, plot_bottom, AXIS, 1.2))
        parts.append(line(left, plot_top, left, plot_bottom, AXIS, 1.2))

        legend_x = left + 280
        legend_y = y0 + 18
        for legend_idx, series in enumerate(panel.series):
            lx = legend_x + legend_idx * 170
            parts.append(line(lx, legend_y - 4, lx + 24, legend_y - 4, series.color, 3))
            parts.append(svg_text(lx + 32, legend_y, series.label, size=12, fill=MUTED))

        for series in panel.series:
            commands: list[str] = []
            point_count = 0
            pen_down = False
            for x_raw, y_raw in zip(x_values, series.values):
                if y_raw is None:
                    pen_down = False
                    continue
                x = x_scale(x_raw)
                y = y_scale(y_raw)
                commands.append(("M" if not pen_down else "L") + f"{x:.1f},{y:.1f}")
                point_count += 1
                pen_down = True
            if commands:
                parts.append(
                    f'<path d="{" ".join(commands)}" fill="none" stroke="{series.color}" '
                    f'stroke-width="2.4" stroke-linejoin="round" stroke-linecap="round"/>'
                )
                if point_count <= 120:
                    for x_raw, y_raw in zip(x_values, series.values):
                        if y_raw is None:
                            continue
                        parts.append(
                            f'<circle cx="{x_scale(x_raw):.1f}" cy="{y_scale(y_raw):.1f}" '
                            f'r="2.4" fill="{series.color}"/>'
                        )

        parts.append(svg_text(left + plot_width / 2, plot_bottom + 40, x_label, size=12, fill=MUTED, anchor="middle"))

    parts.append("</svg>")
    output_path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def render_bar_chart(
    title: str,
    subtitle: str,
    panels: list[BarPanel],
    output_path: Path,
    width: int = 1200,
) -> None:
    left = 86
    right = 36
    top = 76
    panel_height = 280
    bottom = 34
    gap = 24
    height = top + len(panels) * panel_height + (len(panels) - 1) * gap + bottom
    plot_width = width - left - right

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        rect(0, 0, width, height, BG),
        svg_text(24, 34, title, size=22, weight="700"),
        svg_text(24, 58, subtitle, size=13, fill=MUTED),
    ]

    for panel_idx, panel in enumerate(panels):
        y0 = top + panel_idx * (panel_height + gap)
        plot_top = y0 + 34
        plot_bottom = y0 + panel_height - 50
        plot_height = plot_bottom - plot_top
        all_values: list[float | None] = []
        for series in panel.series:
            all_values.extend(series.values)
        ymin, ymax = nice_range(all_values)
        y_scale = make_scaler(ymin, ymax, plot_bottom, plot_top)

        parts.append(svg_text(left, y0 + 18, panel.title, size=15, weight="700"))
        parts.append(svg_text(24, plot_top - 12, panel.ylabel, size=12, fill=MUTED))
        for tick in tick_values(ymin, ymax):
            y = y_scale(tick)
            parts.append(line(left, y, left + plot_width, y, GRID))
            parts.append(svg_text(left - 10, y + 4, compact(tick), size=11, fill=MUTED, anchor="end"))
        parts.append(line(left, plot_bottom, left + plot_width, plot_bottom, AXIS, 1.2))
        parts.append(line(left, plot_top, left, plot_bottom, AXIS, 1.2))

        group_count = max(len(panel.categories), 1)
        group_width = plot_width / group_count
        series_count = max(len(panel.series), 1)
        bar_width = min(48, group_width * 0.72 / series_count)

        for group_idx, category in enumerate(panel.categories):
            group_center = left + group_width * (group_idx + 0.5)
            parts.append(svg_text(group_center, plot_bottom + 20, category, size=12, fill=MUTED, anchor="middle"))
            for series_idx, series in enumerate(panel.series):
                value = series.values[group_idx] if group_idx < len(series.values) else None
                if value is None:
                    continue
                x = group_center - (series_count * bar_width) / 2 + series_idx * bar_width
                y = y_scale(value)
                h = max(0.0, plot_bottom - y)
                parts.append(rect(x + 2, y, bar_width - 4, h, series.color, radius=3))
                parts.append(svg_text(x + bar_width / 2, y - 6, compact(value), size=11, fill=MUTED, anchor="middle"))

        legend_x = left + 280
        legend_y = y0 + 18
        for legend_idx, series in enumerate(panel.series):
            lx = legend_x + legend_idx * 170
            parts.append(rect(lx, legend_y - 11, 14, 10, series.color, radius=2))
            parts.append(svg_text(lx + 22, legend_y, series.label, size=12, fill=MUTED))

    parts.append("</svg>")
    output_path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def render_token_mix_chart(
    total_record: dict[str, Any],
    window_record: dict[str, Any],
    output_path: Path,
    width: int = 1100,
    height: int = 430,
) -> None:
    total_values = [
        get_path(total_record, "total_cached_tokens"),
        get_path(total_record, "total_uncached_prompt_tokens"),
        get_path(total_record, "total_output_tokens"),
    ]
    window_values = [
        get_path(window_record, "interval_delta", "total_cached_tokens"),
        get_path(window_record, "interval_delta", "total_uncached_prompt_tokens"),
        get_path(window_record, "interval_delta", "total_output_tokens"),
    ]
    labels = ["Cached prompt", "Uncached prompt", "Output"]
    colors = [PALETTE[2], PALETTE[3], PALETTE[0]]
    rows = [("Cumulative", total_values), ("Last populated window", window_values)]

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        rect(0, 0, width, height, BG),
        svg_text(24, 34, "Token Mix", size=22, weight="700"),
        svg_text(24, 58, "Cached prompt, uncached prompt, and output token share.", size=13, fill=MUTED),
    ]

    left = 210
    bar_width = 760
    bar_height = 44
    start_y = 130
    row_gap = 116

    for row_idx, (row_label, raw_values) in enumerate(rows):
        y = start_y + row_idx * row_gap
        finite = [v or 0.0 for v in raw_values]
        total = sum(finite)
        parts.append(svg_text(24, y + 29, row_label, size=14, weight="700"))
        parts.append(svg_text(24, y + 52, f"Total: {compact(total)} tokens", size=12, fill=MUTED))
        parts.append(rect(left, y, bar_width, bar_height, "#f9fafb", stroke=GRID, radius=5))
        cursor = left
        for label, color, value in zip(labels, colors, finite):
            if total <= 0 or value <= 0:
                continue
            segment_width = bar_width * value / total
            parts.append(rect(cursor, y, segment_width, bar_height, color))
            if segment_width >= 92:
                parts.append(
                    svg_text(
                        cursor + segment_width / 2,
                        y + 28,
                        f"{value / total * 100:.1f}%",
                        size=12,
                        fill="#ffffff",
                        anchor="middle",
                        weight="700",
                    )
                )
            cursor += segment_width

    legend_y = height - 72
    legend_x = left
    for idx, (label, color) in enumerate(zip(labels, colors)):
        x = legend_x + idx * 210
        parts.append(rect(x, legend_y - 12, 16, 12, color, radius=2))
        parts.append(svg_text(x + 24, legend_y, label, size=13, fill=MUTED))

    parts.append("</svg>")
    output_path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def elapsed_minutes(records: Sequence[dict[str, Any]]) -> list[float]:
    result: list[float] = []
    for idx, record in enumerate(records, 1):
        raw = (
            get_path(record, "interval_delta", "window_end_secs")
            or get_path(record, "elapsed_secs")
            or get_path(record, "wall_time")
            or float(idx)
        )
        result.append(raw / 60.0)
    return result


def last_with_window_metrics(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    metric_paths = [
        ("interval_delta", "request_throughput_req_s"),
        ("interval_delta", "output_throughput_tok_s"),
        ("interval_delta", "cache_hit_rate"),
        ("interval_delta", "ttft", "p99"),
        ("interval_delta", "round_latency", "p99"),
    ]
    for record in reversed(records):
        if any(get_path(record, *path) is not None for path in metric_paths):
            return record
    return records[-1]


def latest_subtitle(total_record: dict[str, Any], window_record: dict[str, Any]) -> str:
    elapsed = get_path(total_record, "elapsed_secs") or get_path(total_record, "wall_time")
    window_end = (
        get_path(window_record, "interval_delta", "window_end_secs")
        or get_path(window_record, "elapsed_secs")
        or get_path(window_record, "wall_time")
    )
    elapsed_text = compact(elapsed / 60.0) + " min" if elapsed is not None else "n/a"
    window_text = compact(window_end / 60.0) + " min" if window_end is not None else "n/a"
    return f"Cumulative through {elapsed_text}; latest populated window ends at {window_text}."


def write_latest_comparison(records: list[dict[str, Any]], out_dir: Path) -> None:
    total_record = records[-1]
    window_record = last_with_window_metrics(records)
    panels = [
        BarPanel(
            title="Request Throughput",
            ylabel="requests / sec",
            categories=["Cumulative", "Window"],
            series=[
                Series(
                    "throughput",
                    [
                        get_path(total_record, "request_throughput_req_s"),
                        get_path(window_record, "interval_delta", "request_throughput_req_s"),
                    ],
                    PALETTE[0],
                )
            ],
        ),
        BarPanel(
            title="Output Throughput",
            ylabel="tokens / sec",
            categories=["Cumulative", "Window"],
            series=[
                Series(
                    "throughput",
                    [
                        get_path(total_record, "output_throughput_tok_s"),
                        get_path(window_record, "interval_delta", "output_throughput_tok_s"),
                    ],
                    PALETTE[2],
                )
            ],
        ),
        BarPanel(
            title="Cache Hit Rate",
            ylabel="percent",
            categories=["Cumulative", "Window"],
            series=[
                Series(
                    "hit rate",
                    [
                        percent_value(total_record, "cache_hit_rate"),
                        percent_value(window_record, "interval_delta", "cache_hit_rate"),
                    ],
                    PALETTE[3],
                )
            ],
        ),
    ]
    render_bar_chart(
        "Latest Throughput and Cache",
        latest_subtitle(total_record, window_record),
        panels,
        out_dir / "00_latest_throughput_cache.svg",
    )

    latency_categories = ["avg", "p50", "p90", "p99"]
    render_bar_chart(
        "Latest Latency Percentiles",
        latest_subtitle(total_record, window_record),
        [
            BarPanel(
                title="TTFT",
                ylabel="seconds",
                categories=latency_categories,
                series=[
                    Series(
                        "Cumulative",
                        [get_path(total_record, "ttft", k) for k in latency_categories],
                        PALETTE[0],
                    ),
                    Series(
                        "Window",
                        [get_path(window_record, "interval_delta", "ttft", k) for k in latency_categories],
                        PALETTE[1],
                    ),
                ],
            ),
            BarPanel(
                title="Round Latency",
                ylabel="seconds",
                categories=latency_categories,
                series=[
                    Series(
                        "Cumulative",
                        [get_path(total_record, "round_latency", k) for k in latency_categories],
                        PALETTE[0],
                    ),
                    Series(
                        "Window",
                        [get_path(window_record, "interval_delta", "round_latency", k) for k in latency_categories],
                        PALETTE[1],
                    ),
                ],
            ),
        ],
        out_dir / "01_latest_latency.svg",
    )
    render_token_mix_chart(total_record, window_record, out_dir / "02_latest_token_mix.svg")


def write_timeseries(records: list[dict[str, Any]], out_dir: Path) -> None:
    x = elapsed_minutes(records)
    subtitle = f"{len(records)} interval summaries; interval series use interval_delta values."

    render_line_chart(
        "Throughput Over Time",
        subtitle,
        x,
        "elapsed time (min)",
        [
            LinePanel(
                "Request Throughput",
                "req/s",
                [
                    Series("Interval", values(records, "interval_delta", "request_throughput_req_s"), PALETTE[0]),
                    Series("Cumulative", values(records, "request_throughput_req_s"), PALETTE[1]),
                ],
            ),
            LinePanel(
                "Output Throughput",
                "tok/s",
                [
                    Series("Interval", values(records, "interval_delta", "output_throughput_tok_s"), PALETTE[2]),
                    Series("Cumulative", values(records, "output_throughput_tok_s"), PALETTE[3]),
                ],
            ),
        ],
        out_dir / "03_throughput_timeseries.svg",
    )

    render_line_chart(
        "Cache Hit Rate Over Time",
        subtitle,
        x,
        "elapsed time (min)",
        [
            LinePanel(
                "Cache Hit Rate",
                "percent",
                [
                    Series("Interval", values(records, "interval_delta", "cache_hit_rate", scale=100), PALETTE[3]),
                    Series("Cumulative", values(records, "cache_hit_rate", scale=100), PALETTE[0]),
                ],
            )
        ],
        out_dir / "04_cache_hit_rate_timeseries.svg",
    )

    latency_keys = ["avg", "p50", "p90", "p99"]
    render_line_chart(
        "TTFT Over Time",
        subtitle,
        x,
        "elapsed time (min)",
        [
            LinePanel(
                "Interval TTFT",
                "seconds",
                [
                    Series(key, values(records, "interval_delta", "ttft", key), PALETTE[idx])
                    for idx, key in enumerate(latency_keys)
                ],
            ),
            LinePanel(
                "Cumulative TTFT",
                "seconds",
                [
                    Series(key, values(records, "ttft", key), PALETTE[idx])
                    for idx, key in enumerate(latency_keys)
                ],
            ),
        ],
        out_dir / "05_ttft_timeseries.svg",
    )

    render_line_chart(
        "Round Latency Over Time",
        subtitle,
        x,
        "elapsed time (min)",
        [
            LinePanel(
                "Interval Round Latency",
                "seconds",
                [
                    Series(key, values(records, "interval_delta", "round_latency", key), PALETTE[idx])
                    for idx, key in enumerate(latency_keys)
                ],
            ),
            LinePanel(
                "Cumulative Round Latency",
                "seconds",
                [
                    Series(key, values(records, "round_latency", key), PALETTE[idx])
                    for idx, key in enumerate(latency_keys)
                ],
            ),
        ],
        out_dir / "06_round_latency_timeseries.svg",
    )

    render_line_chart(
        "Token Volume Over Time",
        subtitle,
        x,
        "elapsed time (min)",
        [
            LinePanel(
                "Prompt Tokens Per Window",
                "tokens",
                [
                    Series("Cached prompt", values(records, "interval_delta", "total_cached_tokens"), PALETTE[2]),
                    Series(
                        "Uncached prompt",
                        values(records, "interval_delta", "total_uncached_prompt_tokens"),
                        PALETTE[3],
                    ),
                ],
            ),
            LinePanel(
                "Output Tokens Per Window",
                "tokens",
                [
                    Series("Output", values(records, "interval_delta", "total_output_tokens"), PALETTE[0]),
                ],
            ),
        ],
        out_dir / "07_token_volume_timeseries.svg",
    )


def main() -> int:
    args = parse_args()
    try:
        records = load_records(args.input)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_latest_comparison(records, args.out_dir)
    write_timeseries(records, args.out_dir)

    print(f"wrote {len(list(args.out_dir.glob('*.svg')))} SVG files to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
