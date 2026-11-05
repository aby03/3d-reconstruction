"""`index.html` of a run summary: one self-contained page (inline CSS, images as relative links)."""

from collections.abc import Mapping, Sequence
from html import escape
from pathlib import Path
from typing import Any

from recon.io.tables import write_bytes

STYLE = """
body { font-family: system-ui, sans-serif; margin: 2em auto; max-width: 1100px; color: #222; }
h1 { margin-bottom: 0.2em; } h2 { border-bottom: 1px solid #ccc; padding-bottom: 0.2em; margin-top: 1.6em; }
table { border-collapse: collapse; margin: 0.6em 0; font-size: 0.9em; }
td, th { border: 1px solid #ddd; padding: 0.2em 0.6em; text-align: left; vertical-align: top; }
th { background: #f4f4f4; } .pass { color: #187a2f; font-weight: 600; } .fail { color: #b3261e; font-weight: 600; }
.na { color: #888; } .warn { color: #a15c00; } img { max-width: 520px; margin: 0.3em; border: 1px solid #ddd; }
"""


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, list) and value and all(isinstance(v, float) for v in value):
        return "[" + ", ".join(f"{v:.3g}" for v in value) + "]"
    return str(value)


def _table(rows: Sequence[Sequence[Any]], header: Sequence[str] | None = None, raw: bool = False) -> str:
    head = "<tr>" + "".join(f"<th>{escape(h)}</th>" for h in header) + "</tr>" if header else ""
    body = "".join(
        "<tr>" + "".join(f"<td>{c if raw else escape(_fmt(c))}</td>" for c in row) + "</tr>" for row in rows
    )
    return f"<table>{head}{body}</table>"


def _verdict(passed: bool | None) -> str:
    if passed is None:
        return '<span class="na">n/a</span>'
    return '<span class="pass">pass</span>' if passed else '<span class="fail">FAIL</span>'


def _stage_section(entry: Mapping[str, Any], checks: Sequence[Mapping[str, Any]]) -> str:
    title = entry["stage"] if entry.get("sequence_id") is None else f"{entry['stage']} ({entry['sequence_id']})"
    info = [
        ("status", entry["status"]),
        ("duration", f"{entry['duration_s']:.1f} s" if entry.get("duration_s") is not None else None),
        ("folder", entry.get("dir")),
        ("backends", ", ".join(f"{k}: {v}" for k, v in (entry.get("backends") or {}).items()) or None),
        ("tools", ", ".join(f"{k} {v}" for k, v in (entry.get("tools") or {}).items()) or None),
    ]
    parts = [f"<h2>{escape(title)}</h2>", _table(info)]
    if metrics := entry.get("metrics"):
        mine = {c["metric"]: c for c in checks if c["sequence_id"] == entry.get("sequence_id")}
        rows = []
        for key, value in metrics.items():
            check = mine.get(key)
            bound = f"{check['op']} {_fmt(check['threshold'])} {_verdict(check['pass'])}" if check else ""
            rows.append((escape(key), escape(_fmt(value)), bound))
        parts.append(_table(rows, ("metric", "value", "threshold"), raw=True))
    if warnings := entry.get("warnings"):
        parts.append("<ul>" + "".join(f'<li class="warn">{escape(w)}</li>' for w in warnings) + "</ul>")
    if planes := entry.get("planes"):
        rows = [
            (p["id"], p["label"], p["normal"], p["d"], f"{100 * p['area_frac']:.1f}%", p["faces"], p["rms_before"],
             p["rms_after"], p["axis_angle_deg"])
            for p in planes
        ]  # fmt: skip
        header = ("id", "label", "normal", "d", "area", "faces", "rms before", "rms after", "axis angle (deg)")
        parts.append("<h3>planes</h3>" + _table(rows, header))
    if qa := entry.get("qa"):
        d = qa["dense_to_mesh"]
        rows = [("tau", qa["tau"])] + [(f"dense → mesh {k}", d[k]) for k in ("mean", "rms", "p50", "p95", "max")]
        rows += [("p95 × diag", qa["p95_frac"]), ("unsupported area", qa["unsupported_area_frac"])]
        parts.append("<h3>geometry check</h3>" + _table(rows))
    if files := entry.get("files"):
        rows = [(name, f.get("up"), f.get("faces"), f"{f['bytes'] / 2**20:.2f} MB") for name, f in files.items()]
        parts.append("<h3>files</h3>" + _table(rows, ("file", "up", "faces", "size")))
    for image in map(escape, entry.get("images", [])):
        parts.append(f'<a href="{image}"><img src="{image}" alt="{image}"></a>')
    return "\n".join(parts)


def render(summary: Mapping[str, Any], out: Path, timings_plot: str | None = None, figures: list[str] | None = None) -> Path:
    """Write `out` (index.html) for `summary`; image paths in the summary are relative to `out`'s folder."""
    checks = summary.get("checks", [])
    failed = [c for c in checks if c["pass"] is False]
    run = [
        ("scene", summary.get("scene")),
        ("run", summary.get("run_id")),
        ("sequences", ", ".join(summary.get("sequence_ids", []))),
        ("hardware preset", summary.get("hardware")),
        ("pipeline version", summary.get("pipeline_version")),
        ("created", summary.get("created")),
        ("total stage time", f"{summary.get('total_s', 0.0):.0f} s"),
    ]
    overview = [
        (escape(c["stage"] + (f" [{c['sequence_id']}]" if c["sequence_id"] else "")), escape(c["metric"]),
         escape(_fmt(c["value"])), escape(f"{c['op']} {_fmt(c['threshold'])}"), _verdict(c["pass"]))
        for c in checks
    ]  # fmt: skip
    diff = [(k, v["default"], v["value"]) for k, v in (summary.get("config_diff") or {}).items()]
    tools = sorted((summary.get("tools") or {}).items())
    sections = [
        f"<h1>{escape(str(summary.get('scene')))} · {escape(str(summary.get('run_id')))}</h1>",
        f"<p>Thresholds: {_verdict(summary.get('pass'))} ({len(checks) - len(failed)} of {len(checks)} pass)</p>",
        "<h2>run</h2>" + _table(run),
        "<h3>thresholds</h3>" + _table(overview, ("stage", "metric", "value", "threshold", ""), raw=True),
        "<h3>config different from the defaults</h3>"
        + (_table(diff, ("key", "default", "run")) if diff else "<p>none</p>"),
        "<h3>tools</h3>" + _table(tools, ("tool", "version")),
    ]
    if timings_plot:
        sections.append(f'<img src="{escape(timings_plot)}" alt="stage durations">')
    if figures:
        sections.append("<h2>figures</h2>")
        sections += [f'<a href="{escape(f)}"><img src="{escape(f)}" alt="{escape(f)}"></a>' for f in figures]
    for entry in summary["stages"]:
        sections.append(_stage_section(entry, [c for c in checks if c["stage"] == entry["stage"]]))
    page = (
        f"<!DOCTYPE html>\n<html><head><meta charset=\"utf-8\"><title>{escape(str(summary.get('run_id')))}</title>"
        f"<style>{STYLE}</style></head>\n<body>\n" + "\n".join(sections) + "\n</body></html>\n"
    )
    write_bytes(page.encode(), out)
    return out
