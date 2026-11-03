"""Run summary: the stages' manifests merged into one record, checked against thresholds, and a metric diff of two
runs."""

from collections.abc import Mapping, Sequence
from typing import Any

LOWER_BOUNDS = {"frames_valid_ratio", "ref_jpeg_min_psnr_db", "registered_ratio"}  # the others are upper bounds


def summarize(run: Mapping[str, Any], stages: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The summary record: run info (`run`) and one entry per stage (`stage`, `sequence_id`, `status`, `duration_s`,
    `metrics`, `warnings`, `backends`, `tools`, ...), in pipeline order."""
    stages = [dict(entry) for entry in stages]
    tools = {name: version for entry in stages for name, version in (entry.get("tools") or {}).items()}
    total = sum(entry.get("duration_s") or 0.0 for entry in stages)
    return {**run, "tools": tools, "total_s": total, "stages": stages}


def _op(metric: str, bound: float | bool) -> str:
    if isinstance(bound, bool):
        return "=="
    return ">=" if metric in LOWER_BOUNDS else "<="


def _passes(value: Any, op: str, bound: float | bool) -> bool | None:
    if value is None or (op != "==" and isinstance(value, bool)):
        return None
    if op == "==":
        return value == bound
    return value >= bound if op == ">=" else value <= bound


def check(summary: Mapping[str, Any], thresholds: Mapping[str, Mapping[str, float | bool]]) -> list[dict[str, Any]]:
    """One row per threshold and stage entry: `{stage, sequence_id, metric, value, op, threshold, pass}`; `pass` is
    None when the stage or the metric is missing."""
    rows = []
    for stage, bounds in thresholds.items():
        entries = [e for e in summary["stages"] if e["stage"] == stage] or [{"stage": stage, "sequence_id": None}]
        for entry in entries:
            for metric, bound in bounds.items():
                value = (entry.get("metrics") or {}).get(metric)
                op = _op(metric, bound)
                rows.append(
                    {
                        "stage": stage,
                        "sequence_id": entry.get("sequence_id"),
                        "metric": metric,
                        "value": value,
                        "op": op,
                        "threshold": bound,
                        "pass": _passes(value, op, bound),
                    }
                )
    return rows


def _label(entry: Mapping[str, Any]) -> str:
    return entry["stage"] if entry.get("sequence_id") is None else f"{entry['stage']}[{entry['sequence_id']}]"


def _numbers(entry: Mapping[str, Any]) -> dict[str, float]:
    values = {"duration_s": entry.get("duration_s"), **(entry.get("metrics") or {})}
    return {k: v for k, v in values.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}


def compare(a: Mapping[str, Any], b: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Rows `{stage, metric, a, b, diff}` for every numeric metric (and duration) of the stages in `a` or `b`; a value
    missing on one side is None and so is its diff."""
    entries_a = {_label(e): _numbers(e) for e in a["stages"]}
    entries_b = {_label(e): _numbers(e) for e in b["stages"]}
    rows = []
    for stage in list(entries_a) + [s for s in entries_b if s not in entries_a]:
        values_a, values_b = entries_a.get(stage, {}), entries_b.get(stage, {})
        for metric in list(values_a) + [m for m in values_b if m not in values_a]:
            va, vb = values_a.get(metric), values_b.get(metric)
            diff = vb - va if va is not None and vb is not None else None
            rows.append({"stage": stage, "metric": metric, "a": va, "b": vb, "diff": diff})
    return rows


def _cell(value: Any) -> str:
    if value is None:
        return "-"
    return f"{value:.4g}" if isinstance(value, float) else str(value)


def format_table(rows: Sequence[Mapping[str, Any]], a_name: str = "a", b_name: str = "b") -> str:
    """The `compare` rows as an aligned text table."""
    header = ("stage", "metric", a_name, b_name, "diff")
    cells = [header] + [(r["stage"], r["metric"], _cell(r["a"]), _cell(r["b"]), _cell(r["diff"])) for r in rows]
    widths = [max(len(row[i]) for row in cells) for i in range(len(header))]
    return "\n".join("  ".join(c.ljust(w) for c, w in zip(row, widths)).rstrip() for row in cells)
