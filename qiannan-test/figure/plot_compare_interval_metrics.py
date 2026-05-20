#!/usr/bin/env python3
"""Plot side-by-side interval metric comparisons for two benchmark runs.

Default usage from the repository root:

    python3 qiannan-test/figure/plot_compare_interval_metrics.py

The x-axis is clipped to the shorter run. For example, if one run has 1000
seconds of records and the other has 500 seconds, only the first 500 seconds are
plotted for both runs.
By default, generated figures are written under qiannan-test/figure-runs/.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from plot_interval_metrics import (
    AXIS,
    BG,
    GRID,
    MUTED,
    PALETTE,
    TEXT,
    compact,
    get_path,
    line,
    load_records,
    make_scaler,
    nice_range,
    rect,
    svg_text,
    tick_values,
)


SCRIPT_DIR = Path(__file__).resolve().parent
RUNS_DIR = SCRIPT_DIR.parent / "runs"
FIGURE_RUNS_DIR = SCRIPT_DIR.parent / "figure-runs"
DEFAULT_LEFT = RUNS_DIR / "dp2tp8_hicache_interval_summaries.jsonl"
DEFAULT_RIGHT = RUNS_DIR / "02_admission64_dp2tp8_interval_summary.jsonl"


@dataclass(frozen=True)
class PointSeries:
    label: str
    points: list[tuple[float, float]]
    color: str


@dataclass(frozen=True)
class ComparePanel:
    title: str
    ylabel: str
    series: list[PointSeries]


def safe_path_name(value: str) -> str:
    chars = [c if c.isalnum() or c in "._-" else "_" for c in value.strip()]
    name = "".join(chars).strip("._-")
    return name or "compare"


def default_out_dir(left_label: str, right_label: str) -> Path:
    return FIGURE_RUNS_DIR / (
        f"{safe_path_name(left_label)}_vs_{safe_path_name(right_label)}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two SGLang interval_summary JSON/JSONL runs as SVG figures."
    )
    parser.add_argument(
        "--left",
        type=Path,
        default=DEFAULT_LEFT,
        help=f"left/baseline input JSONL (default: {DEFAULT_LEFT})",
    )
    parser.add_argument(
        "--right",
        type=Path,
        default=DEFAULT_RIGHT,
        help=f"right/comparison input JSONL (default: {DEFAULT_RIGHT})",
    )
    parser.add_argument(
        "--left-label",
        default="01_dp2tp8_hicache",
        help="legend label for --left",
    )
    parser.add_argument(
        "--right-label",
        default="02_admission64_dp2tp8",
        help="legend label for --right",
    )
    parser.add_argument(
        "-o",
        "--out-dir",
        type=Path,
        default=None,
        help="directory for generated SVG files (default: qiannan-test/figure-runs/<left-label>_vs_<right-label>)",
    )
    args = parser.parse_args()
    if args.out_dir is None:
        args.out_dir = default_out_dir(args.left_label, args.right_label)
    return args


def record_end_secs(record: dict[str, Any], fallback_idx: int) -> float:
    raw = (
        get_path(record, "interval_delta", "window_end_secs")
        or get_path(record, "elapsed_secs")
        or get_path(record, "wall_time")
        or float(fallback_idx)
    )
    return raw


def max_end_secs(records: Sequence[dict[str, Any]]) -> float:
    return max(record_end_secs(record, idx) for idx, record in enumerate(records, 1))


def clip_records(records: Sequence[dict[str, Any]], end_secs: float) -> list[dict[str, Any]]:
    clipped: list[dict[str, Any]] = []
    # Keep a small tolerance so 4561.999999 and 4562.0 land on the same boundary.
    tolerance = max(1e-6, end_secs * 1e-12)
    for idx, record in enumerate(records, 1):
        if record_end_secs(record, idx) <= end_secs + tolerance:
            clipped.append(record)
    return clipped


def points(
    records: Sequence[dict[str, Any]],
    *keys: str,
    scale: float = 1.0,
) -> list[tuple[float, float]]:
    result: list[tuple[float, float]] = []
    for idx, record in enumerate(records, 1):
        x = record_end_secs(record, idx) / 60.0
        value = get_path(record, *keys)
        if value is None or not math.isfinite(value):
            continue
        result.append((x, value * scale))
    return result


def render_compare_line_chart(
    title: str,
    subtitle: str,
    x_max_minutes: float,
    panels: list[ComparePanel],
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
    xmin = 0.0
    xmax = max(x_max_minutes, 1.0)
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
        all_values: list[float | None] = [
            value for series in panel.series for _, value in series.points
        ]
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

        legend_x = left + 260
        legend_y = y0 + 18
        for legend_idx, series in enumerate(panel.series):
            lx = legend_x + legend_idx * 270
            parts.append(line(lx, legend_y - 4, lx + 24, legend_y - 4, series.color, 3))
            parts.append(svg_text(lx + 32, legend_y, series.label, size=12, fill=MUTED))

        for series in panel.series:
            commands: list[str] = []
            pen_down = False
            for x_raw, y_raw in series.points:
                if x_raw < xmin or x_raw > xmax:
                    pen_down = False
                    continue
                x = x_scale(x_raw)
                y = y_scale(y_raw)
                commands.append(("M" if not pen_down else "L") + f"{x:.1f},{y:.1f}")
                pen_down = True
            if commands:
                parts.append(
                    f'<path d="{" ".join(commands)}" fill="none" stroke="{series.color}" '
                    f'stroke-width="2.4" stroke-linejoin="round" stroke-linecap="round"/>'
                )
                if len(series.points) <= 160:
                    for x_raw, y_raw in series.points:
                        if xmin <= x_raw <= xmax:
                            parts.append(
                                f'<circle cx="{x_scale(x_raw):.1f}" cy="{y_scale(y_raw):.1f}" '
                                f'r="2.4" fill="{series.color}"/>'
                            )

        parts.append(
            svg_text(
                left + plot_width / 2,
                plot_bottom + 40,
                "elapsed time (min)",
                size=12,
                fill=MUTED,
                anchor="middle",
            )
        )

    parts.append("</svg>")
    output_path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def pair_series(
    left_records: Sequence[dict[str, Any]],
    right_records: Sequence[dict[str, Any]],
    left_label: str,
    right_label: str,
    *keys: str,
    scale: float = 1.0,
) -> list[PointSeries]:
    return [
        PointSeries(left_label, points(left_records, *keys, scale=scale), PALETTE[0]),
        PointSeries(right_label, points(right_records, *keys, scale=scale), PALETTE[1]),
    ]


def write_throughput_charts(
    left_records: Sequence[dict[str, Any]],
    right_records: Sequence[dict[str, Any]],
    left_label: str,
    right_label: str,
    subtitle: str,
    x_max_minutes: float,
    out_dir: Path,
) -> None:
    render_compare_line_chart(
        "Throughput Comparison",
        subtitle,
        x_max_minutes,
        [
            ComparePanel(
                "Interval Request Throughput",
                "req/s",
                pair_series(
                    left_records,
                    right_records,
                    left_label,
                    right_label,
                    "interval_delta",
                    "request_throughput_req_s",
                ),
            ),
            ComparePanel(
                "Cumulative Request Throughput",
                "req/s",
                pair_series(left_records, right_records, left_label, right_label, "request_throughput_req_s"),
            ),
            ComparePanel(
                "Interval Output Throughput",
                "tok/s",
                pair_series(
                    left_records,
                    right_records,
                    left_label,
                    right_label,
                    "interval_delta",
                    "output_throughput_tok_s",
                ),
            ),
            ComparePanel(
                "Cumulative Output Throughput",
                "tok/s",
                pair_series(left_records, right_records, left_label, right_label, "output_throughput_tok_s"),
            ),
        ],
        out_dir / "00_throughput_compare.svg",
    )


def write_cache_chart(
    left_records: Sequence[dict[str, Any]],
    right_records: Sequence[dict[str, Any]],
    left_label: str,
    right_label: str,
    subtitle: str,
    x_max_minutes: float,
    out_dir: Path,
) -> None:
    render_compare_line_chart(
        "Cache Hit Rate Comparison",
        subtitle,
        x_max_minutes,
        [
            ComparePanel(
                "Interval Cache Hit Rate",
                "percent",
                pair_series(
                    left_records,
                    right_records,
                    left_label,
                    right_label,
                    "interval_delta",
                    "cache_hit_rate",
                    scale=100.0,
                ),
            ),
            ComparePanel(
                "Cumulative Cache Hit Rate",
                "percent",
                pair_series(left_records, right_records, left_label, right_label, "cache_hit_rate", scale=100.0),
            ),
        ],
        out_dir / "01_cache_hit_rate_compare.svg",
    )


def write_latency_chart(
    title: str,
    output_name: str,
    metric_key: str,
    left_records: Sequence[dict[str, Any]],
    right_records: Sequence[dict[str, Any]],
    left_label: str,
    right_label: str,
    subtitle: str,
    x_max_minutes: float,
    out_dir: Path,
) -> None:
    panels: list[ComparePanel] = []
    metric_name = "TTFT" if metric_key == "ttft" else "Round Latency"
    for percentile in ["avg", "p50", "p90", "p99"]:
        panels.append(
            ComparePanel(
                f"Interval {metric_name} {percentile}",
                "seconds",
                pair_series(
                    left_records,
                    right_records,
                    left_label,
                    right_label,
                    "interval_delta",
                    metric_key,
                    percentile,
                ),
            )
        )
    render_compare_line_chart(title, subtitle, x_max_minutes, panels, out_dir / output_name)


def write_token_chart(
    left_records: Sequence[dict[str, Any]],
    right_records: Sequence[dict[str, Any]],
    left_label: str,
    right_label: str,
    subtitle: str,
    x_max_minutes: float,
    out_dir: Path,
) -> None:
    render_compare_line_chart(
        "Token Volume Comparison",
        subtitle,
        x_max_minutes,
        [
            ComparePanel(
                "Cached Prompt Tokens Per Window",
                "tokens",
                pair_series(
                    left_records,
                    right_records,
                    left_label,
                    right_label,
                    "interval_delta",
                    "total_cached_tokens",
                ),
            ),
            ComparePanel(
                "Uncached Prompt Tokens Per Window",
                "tokens",
                pair_series(
                    left_records,
                    right_records,
                    left_label,
                    right_label,
                    "interval_delta",
                    "total_uncached_prompt_tokens",
                ),
            ),
            ComparePanel(
                "Output Tokens Per Window",
                "tokens",
                pair_series(
                    left_records,
                    right_records,
                    left_label,
                    right_label,
                    "interval_delta",
                    "total_output_tokens",
                ),
            ),
        ],
        out_dir / "04_token_volume_compare.svg",
    )


def main() -> int:
    args = parse_args()
    left_all = load_records(args.left)
    right_all = load_records(args.right)

    common_end_secs = min(max_end_secs(left_all), max_end_secs(right_all))
    left_records = clip_records(left_all, common_end_secs)
    right_records = clip_records(right_all, common_end_secs)
    if not left_records or not right_records:
        raise ValueError("no records remain after clipping to the common time range")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    x_max_minutes = common_end_secs / 60.0
    subtitle = (
        f"Clipped to the shorter run: {compact(common_end_secs)} sec "
        f"({compact(x_max_minutes)} min). "
        f"Records shown: {args.left_label} {len(left_records)}/{len(left_all)}, "
        f"{args.right_label} {len(right_records)}/{len(right_all)}."
    )

    write_throughput_charts(
        left_records,
        right_records,
        args.left_label,
        args.right_label,
        subtitle,
        x_max_minutes,
        args.out_dir,
    )
    write_cache_chart(
        left_records,
        right_records,
        args.left_label,
        args.right_label,
        subtitle,
        x_max_minutes,
        args.out_dir,
    )
    write_latency_chart(
        "TTFT Comparison",
        "02_ttft_compare.svg",
        "ttft",
        left_records,
        right_records,
        args.left_label,
        args.right_label,
        subtitle,
        x_max_minutes,
        args.out_dir,
    )
    write_latency_chart(
        "Round Latency Comparison",
        "03_round_latency_compare.svg",
        "round_latency",
        left_records,
        right_records,
        args.left_label,
        args.right_label,
        subtitle,
        x_max_minutes,
        args.out_dir,
    )
    write_token_chart(
        left_records,
        right_records,
        args.left_label,
        args.right_label,
        subtitle,
        x_max_minutes,
        args.out_dir,
    )

    print(
        f"wrote {len(list(args.out_dir.glob('*.svg')))} SVG files to {args.out_dir}; "
        f"common time range is {common_end_secs:.2f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
