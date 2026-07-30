"""Console performance report."""

from __future__ import annotations

from typing import Any

from metrics.performance import PerformanceMetrics


def print_performance_report(
    metrics: PerformanceMetrics,
    *,
    title: str = "Institutional Performance Report",
    extra: dict[str, Any] | None = None,
) -> str:
    text = metrics.format_console(title=title)
    if extra:
        lines = [text.rstrip(), "  ---"]
        for k, v in extra.items():
            lines.append(f"  {k:<22}: {v}")
        lines.append("=" * 68)
        lines.append("")
        text = "\n".join(lines)
    print(text)
    return text
