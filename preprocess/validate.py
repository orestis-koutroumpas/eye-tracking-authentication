"""
Calibrate and validate the Option-B rebuild against the published 200 Hz data (T3.4).

  calibrate   : on a fixed subset of recordings (20 genuine + 20 impostor,
                fixed seed), rebuild the saccades at 200 Hz with several
                peak-velocity smoothing windows and compare each with the
                vendor's peak velocity; with --write the best window is stored
                in config/resample.yml.
  equivalence : compare data_resampled/200hz with the published data, for all
                recordings: event tables (with timing tolerances) and the 113
                features (Spearman rho across recordings). Writes
                results/tables/resample_200hz_equivalence.md.

Neither step uses genuine/impostor labels or any classification result.

Usage:
    python -m preprocess.validate calibrate --src data --write
    python -m preprocess.validate equivalence --src data --resampled data_resampled/200hz
"""

import argparse
import os
import random

import numpy as np
import pandas as pd
import yaml
from scipy.stats import pearsonr, spearmanr

from preprocess.rebuild import START, END, rebuild_events
from preprocess.resample import device_clock
from resample_data import GAZE_COLS, recordings

WINDOWS_MS = [5, 10, 15, 20, 25, 30, 40]   # 5 ms = one 200 Hz sample = raw velocity
TIME_TOL_NS = 100_000                       # 0.1 ms: re-timing moves times by clock jitter (µs)
POS_TOL_PX = 0.05


def _vendor(src_dir):
    return {n: pd.read_csv(os.path.join(src_dir, f"{n}.csv"))
            for n in ("fixations", "saccades", "blinks")}


def calibrate(src, config, n_per_class, write):
    with open(config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    recs = recordings(src)
    rng = random.Random(0)
    subset = (rng.sample([r for r in recs if r.startswith("genuine")], n_per_class)
              + rng.sample([r for r in recs if r.startswith("impostors")], n_per_class))

    rows = []
    for w in WINDOWS_MS:
        pairs = []
        for rel in subset:
            d = os.path.join(src, rel)
            vendor = _vendor(d)
            gaze = pd.read_csv(os.path.join(d, "gaze.csv"))[GAZE_COLS]
            clock = device_clock(gaze["timestamp [ns]"].to_numpy())
            _, _, sac, _ = rebuild_events(gaze, vendor, 200,
                                          {**cfg, "peak_velocity_window_ms": w}, clock)
            m = _match(vendor["saccades"], sac)
            pairs.append(m[["peak velocity [px/s]_v", "peak velocity [px/s]_r"]])
        p = pd.concat(pairs)
        v, r = p.iloc[:, 0].to_numpy(), p.iloc[:, 1].to_numpy()
        rows.append({"window_ms": w, "n_saccades": len(p),
                     "pearson_r": pearsonr(v, r)[0], "spearman_rho": spearmanr(v, r)[0],
                     "median_ratio_vendor_to_rebuilt": float(np.median(v / np.where(r > 0, r, np.nan)))})
    table = pd.DataFrame(rows)
    print(table.round(3).to_string(index=False))
    best = int(table.loc[table["spearman_rho"].idxmax(), "window_ms"])
    print(f"\nbest window (highest Spearman rho): {best} ms")
    if write:
        text = open(config, encoding="utf-8").read().splitlines()
        text = [f"peak_velocity_window_ms: {best}" if line.startswith("peak_velocity_window_ms:")
                else line for line in text]
        open(config, "w", encoding="utf-8").write("\n".join(text) + "\n")
        print(f"written to {config}")


def _match(v, r):
    """Pair vendor and rebuilt events by start time (within TIME_TOL_NS)."""
    v, r = v.astype({START: "int64"}), r.astype({START: "int64"})   # empty files read as object
    v = v.sort_values(START).rename(columns={c: f"{c}_v" for c in v.columns if c != START})
    r = r.sort_values(START).rename(columns={c: f"{c}_r" for c in r.columns if c != START})
    m = pd.merge_asof(v, r, on=START, direction="nearest", tolerance=TIME_TOL_NS)
    return m.dropna(subset=[f"{END}_r"])


def _event_checks(v, r, with_position):
    """Per-criterion agreement of a rebuilt event table with the vendor's."""
    out = {"same count": len(v) == len(r)}
    if not out["same count"] or len(v) == 0:
        return out
    out["start ±0.1 ms"] = bool((np.abs(v[START].values - r[START].values) <= TIME_TOL_NS).all())
    out["end ±0.1 ms"] = bool((np.abs(v[END].values - r[END].values) <= TIME_TOL_NS).all())
    out["duration ±1 ms"] = bool((np.abs(v["duration [ms]"].values - r["duration [ms]"].values) <= 1).all())
    if with_position:
        out["x, y ±0.05 px"] = bool(all((np.abs(v[c].values - r[c].values) <= POS_TOL_PX).all()
                                        for c in ("fixation x [px]", "fixation y [px]")))
    return out


def equivalence(src, resampled, out):
    fix_ok, blink_ok, sac_count_ok, sac_pairs, feats_src, feats_new = [], [], [], [], [], []
    # fix_ok / blink_ok: one dict of criteria per recording
    for rel in recordings(src):
        a, b = os.path.join(src, rel), os.path.join(resampled, rel)
        if not os.path.exists(os.path.join(b, "agg_session.csv")):
            continue
        for name, store, pos in (("fixations", fix_ok, True), ("blinks", blink_ok, False)):
            store.append(_event_checks(pd.read_csv(os.path.join(a, f"{name}.csv")),
                                       pd.read_csv(os.path.join(b, f"{name}.csv")), pos))
        vs, rs = (pd.read_csv(os.path.join(p, "saccades.csv")) for p in (a, b))
        sac_count_ok.append(len(vs) == len(rs))
        sac_pairs.append(_match(vs, rs))
        feats_src.append(pd.read_csv(os.path.join(a, "agg_session.csv")))
        feats_new.append(pd.read_csv(os.path.join(b, "agg_session.csv")))

    sp = pd.concat(sac_pairs)
    lines = ["# Option B at 200 Hz vs published data\n",
             f"Recordings compared: {len(fix_ok)}\n",
             "| Check | Result |", "|---|---|",
             f"| Same number of saccades | {np.mean(sac_count_ok):.1%} of recordings |"]
    for label, checks in (("Fixations", fix_ok), ("Blinks", blink_ok)):
        table = pd.DataFrame(checks)
        for crit in table.columns:
            lines.append(f"| {label}: {crit} (all events of the recording) | "
                         f"{table[crit].fillna(False).mean():.1%} of recordings |")
    for col in ("duration [ms]", "amplitude [px]", "amplitude [deg]",
                "mean velocity [px/s]", "peak velocity [px/s]"):
        rho = spearmanr(sp[f"{col}_v"], sp[f"{col}_r"])[0]
        lines.append(f"| Saccade {col}: Spearman rho vendor vs rebuilt ({len(sp)} saccades) | {rho:.3f} |")

    fs, fn = pd.concat(feats_src, ignore_index=True), pd.concat(feats_new, ignore_index=True)
    rhos = {c: spearmanr(fs[c], fn[c])[0] if fs[c].nunique() > 1 else np.nan
            for c in fs.columns if c != "label"}
    rhos = pd.Series(rhos).sort_values()
    low = rhos[rhos < 0.95]
    lines += ["", f"## Features: Spearman rho across recordings (113 features)\n",
              f"Median rho {rhos.median():.3f}; {len(low)} features below 0.95:\n",
              "| Feature | rho |", "|---|---|"] + [f"| {k} | {v:.3f} |" for k, v in low.items()]

    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nSaved -> {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("calibrate")
    c.add_argument("--src", default="data")
    c.add_argument("--config", default="config/resample.yml")
    c.add_argument("--n", type=int, default=20, help="recordings per class")
    c.add_argument("--write", action="store_true", help="store the best window in the config")
    e = sub.add_parser("equivalence")
    e.add_argument("--src", default="data")
    e.add_argument("--resampled", default="data_resampled/200hz")
    e.add_argument("--out", default="results/tables/resample_200hz_equivalence.md")
    args = parser.parse_args()
    if args.cmd == "calibrate":
        calibrate(args.src, args.config, args.n, args.write)
    else:
        equivalence(args.src, args.resampled, args.out)


if __name__ == "__main__":
    main()