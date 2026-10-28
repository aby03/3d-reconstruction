from dataclasses import replace

import numpy as np

from recon.core.types import SensorStream

LONG_GAP_FACTOR = 5.0


def clean_stream(stream: SensorStream) -> tuple[SensorStream, dict[str, int]]:
    """Sort samples by time and drop repeated timestamps, keeping the first sample of each."""
    order = np.argsort(stream.t_ns, kind="stable")
    t_ns = stream.t_ns[order]
    keep = np.ones(len(t_ns), dtype=bool)
    keep[1:] = np.diff(t_ns) > 0

    counts = {
        "out_of_order": int(np.count_nonzero(np.diff(stream.t_ns) < 0)),
        "duplicates": int(np.count_nonzero(~keep)),
    }
    cleaned = replace(stream, t_ns=t_ns[keep], values=stream.values[order][keep])
    return cleaned, counts


def stream_stats(stream: SensorStream) -> dict[str, float | int]:
    """Sample count, median rate, largest gap and number of gaps longer than 5x the median period."""
    n = len(stream.t_ns)
    if n < 2:
        return {"n": n, "rate_hz": 0.0, "max_gap_ms": 0.0, "long_gaps": 0}
    dt = np.diff(stream.t_ns)
    median_dt = float(np.median(dt))
    return {
        "n": n,
        "rate_hz": 1e9 / median_dt,
        "max_gap_ms": float(dt.max()) * 1e-6,
        "long_gaps": int(np.count_nonzero(dt > LONG_GAP_FACTOR * median_dt)),
    }
