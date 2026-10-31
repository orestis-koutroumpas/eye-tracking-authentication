"""
Build one feature dataset per sampling frequency (T3.5, Option B).

For every recording in ``--src`` (genuine/<session>/ and
impostors/<user>/<session>/) and every frequency, the 200 Hz gaze and
eye-state streams are downsampled (preprocess/resample.py), fixations,
saccades and blinks are rebuilt (preprocess/rebuild.py), the CSVs are written
to ``--out/<f>hz/`` with the same folder layout, and the existing feature
code (extract_features + aggregate_recording) produces ``agg_session.csv``.
The source folder is only read, never modified.

Usage:
    python resample_data.py --src data --out data_resampled --freqs 200 100 50 30
"""

import argparse
import glob
import logging
import os

import pandas as pd
import yaml

from preprocess.features import aggregate_recording, extract_features
from preprocess.rebuild import rebuild_events
from preprocess.resample import device_clock, resample_stream

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

GAZE_COLS = ["timestamp [ns]", "gaze x [px]", "gaze y [px]", "fixation id",
             "blink id", "azimuth [deg]", "elevation [deg]"]


def recordings(src):
    """Relative paths of all recordings: genuine/<s> and impostors/<u>/<s>."""
    found = glob.glob(os.path.join(src, "genuine", "*", "")) + \
        glob.glob(os.path.join(src, "impostors", "*", "*", ""))
    return sorted(os.path.relpath(p, src) for p in found)


def build_recording(src_dir, out_dir, hz, cfg):
    gaze = pd.read_csv(os.path.join(src_dir, "gaze.csv"))[GAZE_COLS]
    eye_states = pd.read_csv(os.path.join(src_dir, "3d_eye_states.csv"))
    vendor = {name: pd.read_csv(os.path.join(src_dir, f"{name}.csv"))
              for name in ("fixations", "saccades", "blinks")}

    clock = device_clock(gaze["timestamp [ns]"].to_numpy())
    gaze_f = resample_stream(gaze, hz, cfg["method"])
    eye_f = resample_stream(eye_states, hz, cfg["method"])
    gaze_f, fixations, saccades, blinks = rebuild_events(gaze_f, vendor, hz, cfg, clock)

    os.makedirs(out_dir, exist_ok=True)
    gaze_f.to_csv(os.path.join(out_dir, "gaze.csv"), index=False)
    eye_f.to_csv(os.path.join(out_dir, "3d_eye_states.csv"), index=False)
    fixations.to_csv(os.path.join(out_dir, "fixations.csv"), index=False)
    saccades.to_csv(os.path.join(out_dir, "saccades.csv"), index=False)
    blinks.to_csv(os.path.join(out_dir, "blinks.csv"), index=False)

    extract_features(out_dir)
    aggregate_recording(out_dir, "agg_session")
    return {"fixations": len(fixations), "saccades": len(saccades), "blinks": len(blinks)}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", default="data")
    parser.add_argument("--out", default="data_resampled")
    parser.add_argument("--config", default="config/resample.yml")
    parser.add_argument("--freqs", type=int, nargs="+", default=None,
                        help="frequencies in Hz (default: from the config)")
    parser.add_argument("--limit", type=int, default=None,
                        help="process only the first N recordings (for tests)")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    freqs = args.freqs or cfg["frequencies"]
    recs = recordings(args.src)[: args.limit]
    logger.info(f"{len(recs)} recordings | freqs={freqs} | method={cfg['method']}")

    for hz in freqs:
        counts, failed = [], []
        for rel in recs:
            try:
                c = build_recording(os.path.join(args.src, rel),
                                    os.path.join(args.out, f"{hz}hz", rel), hz, cfg)
                counts.append({"recording": rel, **c})
            except Exception as e:  # keep going; report at the end
                failed.append(rel)
                logger.error(f"{hz} Hz {rel}: {e}")
        summary = pd.DataFrame(counts)
        summary.to_csv(os.path.join(args.out, f"{hz}hz", "event_counts.csv"), index=False)
        logger.info(f"== {hz} Hz: {len(counts)} ok, {len(failed)} failed | "
                    f"median per recording: {summary.drop(columns='recording').median().to_dict()}")
        if failed:
            logger.error(f"   failed: {failed}")


if __name__ == "__main__":
    main()