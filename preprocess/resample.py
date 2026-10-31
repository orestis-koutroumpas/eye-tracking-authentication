"""
Downsample the 200 Hz sample streams of one recording (T3.2).

`resample_stream` takes a processed ``gaze.csv`` or ``3d_eye_states.csv``
table and returns the rows a tracker running at ``target_hz`` would have
recorded. The stream is split into contiguous segments at gaps (credential
note filter, tracking loss); nothing is filtered or bridged across a gap.

  - method "aa"   : zero-phase 4th-order Butterworth low-pass (cut-off at 80%
                    of the new Nyquist frequency) on the continuous columns of
                    each segment, then the samples on the target grid are kept.
  - method "pick" : the samples on the target grid are kept, no filter.

Which rows are kept is decided on the device clock, not on wall time: the
tracker's real rate is ~199.8 Hz, so each sample gets its position in the
uninterrupted stream, ``round((t - t0) / mean step)``, and a ``target_hz``
device keeps every (200 / target_hz)-th position (every 2nd at 100 Hz, every
4th at 50 Hz, positions round(k * 6.67) at 30 Hz). Positions that fall in a gap
simply have no row.

Label columns (``fixation id``, ``blink id``) are never filtered; each kept
row keeps the labels of the original sample. At 200 Hz the input is returned
unchanged.

Importable helper (driven by resample_data.py):
    from preprocess.resample import resample_stream

    gaze_50 = resample_stream(gaze, 50, method="aa")
"""

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

SOURCE_HZ = 200
TS = "timestamp [ns]"
GAP_NS = 7_500_000        # a step > 1.5 source periods (7.5 ms) is a gap
MIN_FILTER_ROWS = 30      # shorter segments are picked without filtering
LABEL_COLS = ["fixation id", "blink id"]


def segments(ts):
    """(start, stop) row indices of the contiguous segments of a timestamp array."""
    breaks = np.flatnonzero(np.diff(ts) > GAP_NS) + 1
    edges = np.r_[0, breaks, len(ts)]
    return list(zip(edges[:-1], edges[1:]))


def device_clock(ts):
    """Device clock of a recording, from its 200 Hz timestamps.

    Returns {"t0", "step", "positions", "ts"}: first timestamp, mean sample
    step within contiguous segments (~5.0047 ms, i.e. ~199.8 Hz), and the
    position of every recorded sample in the uninterrupted stream.
    """
    ts = np.asarray(ts, dtype=np.int64)
    dt = np.diff(ts)
    ok = dt <= GAP_NS
    clock = {"t0": int(ts[0]), "step": float(dt[ok].mean()) if ok.any() else 1e9 / SOURCE_HZ}
    clock["positions"] = positions(ts, clock)
    clock["ts"] = ts
    return clock


def positions(times, clock):
    """Position of each time in the uninterrupted 200 Hz stream."""
    return np.rint((np.asarray(times, dtype=float) - clock["t0"]) / clock["step"]).astype(np.int64)


def on_grid(position, target_hz):
    """True where a stream position is recorded by a ``target_hz`` device.

    The device keeps positions round(m * 200 / target_hz), m = 0, 1, 2, ...
    (every 2nd at 100 Hz, every 4th at 50 Hz, round(m * 6.67) at 30 Hz).
    """
    factor = SOURCE_HZ / target_hz
    if float(factor).is_integer():
        return position % int(factor) == 0
    return np.rint(np.rint(position / factor) * factor) == position


def snap_up(position, target_hz):
    """Index m of the first device sample at or after each stream position."""
    factor = SOURCE_HZ / target_hz
    m = np.floor(np.asarray(position) / factor).astype(np.int64)
    for _ in range(3):                       # at most two steps forward
        behind = np.rint(m * factor) < position
        m = np.where(behind, m + 1, m)
    return m


def grid_time(m, target_hz, clock):
    """Timestamp of device sample m: the recorded timestamp at that stream
    position, or the clock's estimate where the position falls in a gap."""
    pos = np.rint(np.asarray(m) * SOURCE_HZ / target_hz).astype(np.int64)
    t = clock["t0"] + pos * clock["step"]
    i = np.clip(np.searchsorted(clock["positions"], pos), 0, len(clock["positions"]) - 1)
    recorded = clock["positions"][i] == pos
    return np.where(recorded, clock["ts"][i], t).astype(np.int64)


def keep_mask(ts, target_hz):
    """Boolean mask of the rows a ``target_hz`` device would have recorded."""
    return on_grid(device_clock(ts)["positions"], target_hz)


def resample_stream(df, target_hz, method="aa"):
    """Return the rows of ``df`` a ``target_hz`` tracker would have recorded."""
    if target_hz == SOURCE_HZ:
        return df.reset_index(drop=True).copy()
    if method not in ("aa", "pick"):
        raise ValueError(f"unknown method {method!r}")

    continuous = [c for c in df.columns
                  if c != TS and c not in LABEL_COLS and pd.api.types.is_float_dtype(df[c])]
    sos = butter(4, 0.8 * target_hz / 2, fs=SOURCE_HZ, output="sos") if method == "aa" else None

    out = df.reset_index(drop=True).copy()
    ts = out[TS].to_numpy(dtype=np.int64)
    if sos is not None:
        for a, b in segments(ts):
            if b - a < MIN_FILTER_ROWS:
                continue
            for col in continuous:
                values = out[col].to_numpy(dtype=float)[a:b]
                if np.isfinite(values).all():      # skip columns with missing values
                    out.loc[a:b - 1, col] = sosfiltfilt(sos, values)
    return out[keep_mask(ts, target_hz)].reset_index(drop=True)