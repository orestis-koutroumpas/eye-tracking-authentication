"""
Sanity checks of the downsampled datasets (T3.6).

Per frequency: effective sampling rate, events per second, share of vendor
fixations dropped and saccades merged, mean fixation duration, median saccade
peak velocity, and features that became constant. Writes
results/tables/resample_checks.md.

Usage:
    python utils/resample_checks.py --src data --resampled data_resampled --freqs 200 100 50 30
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd

TS = "timestamp [ns]"


def _recordings(root):
    found = glob.glob(os.path.join(root, "genuine", "*", "")) + \
        glob.glob(os.path.join(root, "impostors", "*", "*", ""))
    return sorted(os.path.relpath(p, root) for p in found)


def _stats(root, recs):
    rows, feats = [], []
    for rel in recs:
        d = os.path.join(root, rel)
        gaze = pd.read_csv(os.path.join(d, "gaze.csv"), usecols=[TS])[TS].to_numpy()
        fix = pd.read_csv(os.path.join(d, "fixations.csv"))
        sac = pd.read_csv(os.path.join(d, "saccades.csv"))
        blk = pd.read_csv(os.path.join(d, "blinks.csv"))
        dur_s = (gaze[-1] - gaze[0]) / 1e9
        dt = np.diff(gaze)
        rows.append({
            "step_ms": np.median(dt) / 1e6,
            "rate_hz": 1e9 / dt[dt < 2 * np.median(dt)].mean(),
            "fixations": len(fix), "saccades": len(sac), "blinks": len(blk),
            "fix_per_s": len(fix) / dur_s, "sac_per_s": len(sac) / dur_s,
            "blink_per_s": len(blk) / dur_s,
            "fix_dur_ms": fix["duration [ms]"].mean(),
            "sac_peak_px_s": sac["peak velocity [px/s]"].median() if len(sac) else np.nan,
        })
        feats.append(pd.read_csv(os.path.join(d, "agg_session.csv")))
    return pd.DataFrame(rows), pd.concat(feats, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", default="data")
    parser.add_argument("--resampled", default="data_resampled")
    parser.add_argument("--freqs", type=int, nargs="+", default=[200, 100, 50, 30])
    parser.add_argument("--out", default="results/tables/resample_checks.md")
    args = parser.parse_args()

    recs = _recordings(args.src)
    vendor, _ = _stats(args.src, recs)
    table, constant = [], {}
    for hz in args.freqs:
        root = os.path.join(args.resampled, f"{hz}hz")
        have = [r for r in recs if os.path.exists(os.path.join(root, r, "agg_session.csv"))]
        s, feats = _stats(root, have)
        table.append({
            "Hz": hz, "recordings": len(have),
            "median step (ms)": f"{s['step_ms'].median():.1f}",
            "effective rate (Hz)": f"{s['rate_hz'].median():.1f}",
            "fixations/s": f"{s['fix_per_s'].mean():.2f}",
            "saccades/s": f"{s['sac_per_s'].mean():.2f}",
            "blinks/s": f"{s['blink_per_s'].mean():.3f}",
            "fixations dropped": f"{1 - s['fixations'].sum() / vendor['fixations'].sum():.1%}",
            "saccades vs vendor": f"{s['saccades'].sum() / vendor['saccades'].sum():.1%}",
            "mean fixation dur (ms)": f"{s['fix_dur_ms'].mean():.0f}",
            "median peak vel (px/s)": f"{s['sac_peak_px_s'].median():.0f}",
        })
        constant[hz] = [c for c in feats.columns if c != "label" and feats[c].nunique() <= 1]

    t = pd.DataFrame(table)
    lines = ["# Downsampling sanity checks\n",
             f"Vendor 200 Hz reference: {vendor['fix_per_s'].mean():.2f} fixations/s, "
             f"{vendor['sac_per_s'].mean():.2f} saccades/s, mean fixation "
             f"{vendor['fix_dur_ms'].mean():.0f} ms\n",
             "| " + " | ".join(t.columns) + " |", "|" + "---|" * len(t.columns)]
    lines += ["| " + " | ".join(str(v) for v in r) + " |" for r in t.itertuples(index=False)]
    lines += ["", "## Constant features (no variation across recordings)\n"]
    lines += [f"- {hz} Hz: {', '.join(c) if c else 'none'}" for hz, c in constant.items()]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    open(args.out, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()