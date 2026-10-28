"""
Compare two per-seed result files on their common (seed, model[, train_pct]) rows (T1.2).

Prints the largest absolute EER difference per model; 0.00 everywhere means
the two runs are identical (same code, same data, same splits).

Usage:
    python utils/compare_runs.py results/metrics/setting_a_per_seed.csv results/metrics/setting_a_repro_per_seed.csv
"""

import sys

import pandas as pd


def main(path_a, path_b):
    a, b = pd.read_csv(path_a), pd.read_csv(path_b)
    keys = [k for k in ("train_pct", "seed", "model") if k in a and k in b]
    m = a.merge(b, on=keys, suffixes=("_a", "_b"))
    if m.empty:
        raise SystemExit("no common rows")
    m["diff"] = (m["EER_a"] - m["EER_b"]).abs()
    print(f"{len(m)} common rows on {keys}")
    print(m.groupby("model")["diff"].max().round(4).rename("max |EER diff| (pp)").to_string())
    print("IDENTICAL" if m["diff"].max() < 1e-9 else "DIFFERENT")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])