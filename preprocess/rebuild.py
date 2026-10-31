"""
Rebuild fixations, saccades and blinks at a lower sampling rate (T3.3, Option B).

The events themselves come from the eye tracker's 200 Hz detection (the
recording's processed fixations.csv, saccades.csv, blinks.csv); what a
lower-rate device would measure of them is recomputed:

  timing     : each event's start and end are moved to the first device sample
               at or after them (device grid of preprocess/resample.py), so
               durations become multiples of the sampling period. At 200 Hz
               the times are unchanged.
  fixations  : survive if they still span >= ``min_fixation_ms`` and contain
               >= ``min_fixation_samples`` device samples; x, y, azimuth,
               elevation = mean of their kept gaze samples (vendor value if
               none is left). Labels of dropped fixations are cleared in the
               gaze table, so their samples no longer count as fixation.
  saccades   : the vendor's saccades, re-timed; two saccades separated only by
               a dropped fixation are merged into one. Amplitude = gaze
               distance between the first device sample at the saccade start
               and the first sample at its end; mean / peak velocity = mean /
               max of the sample velocity inside, smoothed with a centred
               moving average of ``peak_velocity_window_ms``; amplitude /
               duration if no valid sample is left (data gap).
  blinks     : the vendor's blinks, re-timed.

Only events listed in the processed CSVs are used: the original preprocessing
(filters.drop_rows) removed events touching the excluded credential-note
samples but left their labels on the gaze samples.

Importable helper (driven by resample_data.py and preprocess/validate.py):
    from preprocess.rebuild import rebuild_events

    gaze_f, fixations, saccades, blinks = rebuild_events(gaze_f, vendor, hz, cfg, clock)
"""

import numpy as np
import pandas as pd

from preprocess.resample import GAP_NS, TS, grid_time, positions, snap_up

GX, GY, AZ, EL = "gaze x [px]", "gaze y [px]", "azimuth [deg]", "elevation [deg]"
START, END, DUR = "start timestamp [ns]", "end timestamp [ns]", "duration [ms]"


def _retime(events, hz, clock):
    """Move start / end of each event to the device grid; adds n_samples."""
    ev = events.copy()
    m_start = snap_up(positions(ev[START], clock), hz)
    m_end = snap_up(positions(ev[END], clock), hz)
    ev[START] = grid_time(m_start, hz, clock)
    ev[END] = grid_time(m_end, hz, clock)
    ev[DUR] = ((ev[END] - ev[START]) // 1_000_000).astype(int)
    ev["n_samples"] = m_end - m_start
    return ev


def rebuild_fixations(gaze, vendor_fix, hz, clock, min_fixation_ms=60, min_fixation_samples=2):
    """Returns (gaze with cleared labels, fixations, ids of dropped fixations)."""
    fix = _retime(vendor_fix, hz, clock)
    keep = (fix["n_samples"] >= min_fixation_samples) & (fix[DUR] >= min_fixation_ms)
    dropped = fix.loc[~keep, "fixation id"]
    fix = fix[keep].drop(columns="n_samples")

    gaze = gaze.copy()
    gaze.loc[~gaze["fixation id"].isin(fix["fixation id"]), "fixation id"] = np.nan
    means = gaze.dropna(subset=["fixation id"]).groupby("fixation id")[[GX, GY, AZ, EL]].mean()
    for src, dst in ((GX, "fixation x [px]"), (GY, "fixation y [px]"), (AZ, AZ), (EL, EL)):
        fix[dst] = fix["fixation id"].map(means[src]).fillna(fix[dst])
    cols = ["fixation id", START, END, DUR, "fixation x [px]", "fixation y [px]", AZ, EL]
    return gaze, fix[cols].sort_values(START).reset_index(drop=True), set(dropped)


def sample_velocity(gaze, hz, window_ms):
    """Gaze speed [px/s] arriving at each kept sample, smoothed with a centred
    moving average of ``window_ms`` (>= 1 sample); NaN right after a gap."""
    ts = gaze[TS].to_numpy(dtype=np.int64)
    dt = np.diff(ts)
    step = np.hypot(np.diff(gaze[GX].to_numpy()), np.diff(gaze[GY].to_numpy()))
    v = np.r_[np.nan, step / (dt / 1e9)]
    v[1:][dt > max(GAP_NS, 1.5e9 / hz)] = np.nan
    width = max(1, int(round(window_ms * hz / 1000)))
    if width > 1:
        v = pd.Series(v).rolling(width, center=True, min_periods=1).mean().to_numpy()
    return v


def _merge_across_dropped(sac, vendor_fix, dropped):
    """Merge consecutive vendor saccades separated only by a dropped fixation."""
    if not dropped or sac.empty:
        return sac[[START, END]].to_numpy()
    gone = vendor_fix[vendor_fix["fixation id"].isin(dropped)][[START, END]].to_numpy()
    tol = 1_000_000                                       # 1 ms
    out = [list(sac.iloc[0][[START, END]])]
    for s, e in sac[[START, END]].to_numpy()[1:]:
        prev_end = out[-1][1]
        bridged = ((np.abs(gone[:, 0] - prev_end) <= tol) & (np.abs(gone[:, 1] - s) <= tol)).any()
        if bridged:
            out[-1][1] = e
        else:
            out.append([s, e])
    return np.array(out, dtype=np.int64)


def rebuild_saccades(gaze, vendor_sac, vendor_fix, dropped, hz, clock, window_ms):
    sac = vendor_sac.sort_values(START)
    spans = _merge_across_dropped(sac, vendor_fix, dropped)
    cols = ["saccade id", START, END, DUR, "amplitude [px]", "amplitude [deg]",
            "mean velocity [px/s]", "peak velocity [px/s]"]
    if len(spans) == 0:
        return pd.DataFrame(columns=cols)

    ev = _retime(pd.DataFrame(spans, columns=[START, END]), hz, clock)
    period = int(round(1e9 / hz))
    ev[END] = np.maximum(ev[END], ev[START] + period)     # at least one period long
    ev[DUR] = ((ev[END] - ev[START]) // 1_000_000).astype(int)

    ts = gaze[TS].to_numpy(dtype=np.int64)
    x, y = gaze[GX].to_numpy(), gaze[GY].to_numpy()
    az, el = gaze[AZ].to_numpy(), gaze[EL].to_numpy()
    v = sample_velocity(gaze, hz, window_ms)
    half = period // 2
    i0 = np.searchsorted(ts, ev[START].to_numpy() - half)  # first sample at the start
    i1 = np.searchsorted(ts, ev[END].to_numpy() - half)    # first sample at the end
    i0 = np.minimum(i0, i1 - 1)                             # no sample inside: use the one before

    rows = []
    for (_, r), a, b in zip(ev.iterrows(), i0, i1):
        if a < 0 or b >= len(ts):
            continue
        inside = v[a + 1:b + 1]
        amp = float(np.hypot(x[b] - x[a], y[b] - y[a]))
        fallback = amp / ((r[END] - r[START]) / 1e9)          # no valid sample inside
        finite = np.isfinite(inside).any()
        rows.append({
            START: r[START], END: r[END], DUR: r[DUR],
            "amplitude [px]": amp,
            "amplitude [deg]": float(np.hypot(az[b] - az[a], el[b] - el[a])),
            "mean velocity [px/s]": fallback,
            "peak velocity [px/s]": float(np.nanmax(inside)) if finite else fallback,
        })
    out = pd.DataFrame(rows, columns=cols[1:])
    out.insert(0, "saccade id", np.arange(1, len(out) + 1))
    return out


def rebuild_blinks(vendor_blk, hz, clock):
    if vendor_blk.empty:
        return vendor_blk[["blink id", START, END, DUR]].copy()
    blk = _retime(vendor_blk, hz, clock)
    return blk[blk["n_samples"] >= 1][["blink id", START, END, DUR]].reset_index(drop=True)


def rebuild_events(gaze, vendor, hz, cfg, clock):
    """All three event tables for one recording at ``hz``.

    gaze   : downsampled gaze table (with vendor labels)
    vendor : dict with the recording's processed "fixations", "saccades", "blinks"
    cfg    : config/resample.yml
    clock  : preprocess.resample.device_clock(original 200 Hz gaze timestamps)
    """
    gaze, fixations, dropped = rebuild_fixations(
        gaze, vendor["fixations"], hz, clock,
        cfg["min_fixation_ms"], cfg["min_fixation_samples"])
    saccades = rebuild_saccades(gaze, vendor["saccades"], vendor["fixations"], dropped,
                                hz, clock, cfg["peak_velocity_window_ms"])
    blinks = rebuild_blinks(vendor["blinks"], hz, clock)
    return gaze, fixations, saccades, blinks