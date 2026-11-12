#!/usr/bin/env python3
"""
Compare two summary-metric CSVs (A vs B) and write a Markdown report in the same
layout as the single-setting report: one table per metric, rows = train %,
columns = models. For every model there are three columns: A, B and the
difference (B - A).

Expected CSV columns: train_pct, model, <METRIC>_mean, <METRIC>_ci_lo, <METRIC>_ci_hi, ...

Usage:
    python utils/compare_metrics.py results\\metrics\\setting_a_summary.csv results\\metrics\\setting_b_summary.csv
    python utils/compare_metrics.py a.csv b.csv --labels "Setting A" "Setting B"
    python utils/compare_metrics.py a.csv b.csv --no-ci          # means only (narrower tables)
    python utils/compare_metrics.py a.csv b.csv --out-dir results/tables --out-name my_report.md

Output (default): results/tables/<A>_vs_<B>.md
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import pandas as pd

KEYS = ["train_pct", "model"]
HIGHER_IS_BETTER = {"AUC"}                      # all other metrics: lower is better
METRIC_ORDER = ["EER", "FAR", "FRR", "AUC", "FAR_at_FRR1"]
METRIC_TITLES = {"FAR_at_FRR1": "FAR @ FRR=1%"}  # pretty section titles
DEC = 2


def norm_path(p: str) -> Path:
    """Accept Windows-style paths (results\\metrics\\x.csv) on any OS."""
    if os.name != "nt":
        p = p.replace("\\", "/")
    return Path(p)


def load(path: Path) -> pd.DataFrame:
    if not path.is_file():
        sys.exit(f"ERROR: file not found: {path}")
    df = pd.read_csv(path)
    missing = [k for k in KEYS if k not in df.columns]
    if missing:
        sys.exit(f"ERROR: {path} is missing required columns: {missing}")
    return df


def detect_metrics(df: pd.DataFrame) -> list[str]:
    found = [c[:-5] for c in df.columns if c.endswith("_mean")]
    return [m for m in METRIC_ORDER if m in found] + [m for m in found if m not in METRIC_ORDER]


def pick_labels(pa: Path, pb: Path, user: list[str] | None) -> tuple[str, str]:
    if user:
        return user[0], user[1]
    if pa.stem != pb.stem:
        return pa.stem, pb.stem
    la, lb = pa.resolve().parent.name or "A", pb.resolve().parent.name or "B"
    return (la, lb) if la != lb else ("A", "B")


def slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")


def cell(row: pd.Series | None, metric: str, with_ci: bool) -> str:
    if row is None or pd.isna(row.get(f"{metric}_mean")):
        return "–"
    s = f"{row[f'{metric}_mean']:.{DEC}f}"
    if with_ci and not (pd.isna(row.get(f"{metric}_ci_lo")) or pd.isna(row.get(f"{metric}_ci_hi"))):
        s += f" [{row[f'{metric}_ci_lo']:.{DEC}f}-{row[f'{metric}_ci_hi']:.{DEC}f}]"
    return s


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def build_report(a, b, metrics, la, lb, with_ci) -> str:
    # index rows by (train_pct, model)
    ia = a.set_index(KEYS)
    ib = b.set_index(KEYS)
    pcts = sorted(set(a["train_pct"]) | set(b["train_pct"]), reverse=True)  # 70 -> 10, like the source report
    models = sorted(set(a["model"]) | set(b["model"]))

    L = [f"## {la} vs {lb} - performance vs training size\n"]
    ci_txt = "mean [95% bootstrap CI]" if with_ci else "mean"
    L.append(f"_Values in %, {ci_txt}. Rows = train % of sessions; AUC higher is better._  ")
    L.append(f"_A = {la}, B = {lb}. Diff = B − A (negative = B lower)._\n")

    for met in metrics:
        L.append(f"### {METRIC_TITLES.get(met, met)}\n")
        headers = ["train %"]
        for mo in models:
            headers += [f"{mo} (A)", f"{mo} (B)", f"{mo} Diff"]
        rows = []
        for p in pcts:
            r = [str(p)]
            for mo in models:
                ra = ia.loc[(p, mo)] if (p, mo) in ia.index else None
                rb = ib.loc[(p, mo)] if (p, mo) in ib.index else None
                if ra is not None and rb is not None:
                    d = f"{rb[f'{met}_mean'] - ra[f'{met}_mean']:+.{DEC}f}"
                else:
                    d = "–"
                r += [cell(ra, met, with_ci), cell(rb, met, with_ci), d]
            rows.append(r)
        L.append(md_table(headers, rows))
        L.append("")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv_a")
    ap.add_argument("csv_b")
    ap.add_argument("--labels", nargs=2, metavar=("A", "B"), help="display names for the two files")
    ap.add_argument("--out-dir", default="results/tables")
    ap.add_argument("--out-name", default=None)
    ap.add_argument("--no-ci", action="store_true", help="show only means (no confidence intervals)")
    args = ap.parse_args()

    pa, pb = norm_path(args.csv_a), norm_path(args.csv_b)
    a, b = load(pa), load(pb)
    metrics = [m for m in detect_metrics(a) if m in detect_metrics(b)]
    if not metrics:
        sys.exit("ERROR: no common '<metric>_mean' columns in both files.")

    ka, kb = set(map(tuple, a[KEYS].values)), set(map(tuple, b[KEYS].values))
    if ka != kb:
        print(f"WARNING: {len(ka ^ kb)} (train_pct, model) pairs exist in only one file; shown as '–'.")

    la, lb = pick_labels(pa, pb, args.labels)
    report = build_report(a, b, metrics, la, lb, with_ci=not args.no_ci)

    out_dir = norm_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / (args.out_name or f"{slug(la)}_vs_{slug(lb)}.md")
    out.write_text(report, encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()