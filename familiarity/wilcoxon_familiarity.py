"""Wilcoxon signed-rank significance: baseline vs familiarity-reduced per-seed results.

    python familiarity/wilcoxon_familiarity.py
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

KEYS = ["seed", "model", "train_pct"]
METRIC_ORDER = ["EER", "FAR", "FRR", "AUC", "FAR_at_FRR1"]
METRIC_TITLES = {"FAR_at_FRR1": "FAR @ FRR=1%"}


def benjamini_hochberg(pvals):
    p = np.array(pvals, float)
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        idx = order[i]
        prev = min(prev, p[idx] * n / (i + 1))
        q[idx] = min(prev, 1.0)
    return q


def bootstrap_median_ci(diff, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    d = np.asarray(diff, float)
    d = d[~np.isnan(d)]
    if d.size == 0:
        return np.nan, np.nan
    meds = np.median(rng.choice(d, (n_boot, d.size), replace=True), axis=1)
    return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def rank_biserial(diff):
    d = np.asarray(diff, float)
    d = d[(~np.isnan(d)) & (d != 0)]
    if d.size == 0:                                   # FIX: was `!= 0`
        return 0.0
    ranks = pd.Series(np.abs(d)).rank().to_numpy()
    return float((ranks[d > 0].sum() - ranks[d < 0].sum()) / ranks.sum())


def compute(baseline_csv, reduced_csv, models, ratios, metrics, alternative, n_boot=10000):
    baseline = pd.read_csv(baseline_csv)
    reduced = pd.read_csv(reduced_csv)
    for name, df in [("baseline", baseline), ("reduced", reduced)]:
        dup = df.duplicated(KEYS).sum()               # FIX: was `.duplicate`
        if dup:
            raise ValueError(f"{name}: {dup} duplicate rows on {KEYS}")
    m = baseline.merge(reduced, on=KEYS, suffixes=("_baseline", "_reduced"), validate="one_to_one")
    n_seeds = m.seed.nunique()

    rows = []
    for model in models:
        g = m[m.model == model]
        for ratio in ratios:
            sub = g[g.train_pct == ratio]
            for metric in metrics:
                b = sub[f"{metric}_baseline"].to_numpy(float)
                r = sub[f"{metric}_reduced"].to_numpy(float)
                ok = ~(np.isnan(b) | np.isnan(r))
                b, r = b[ok], r[ok]
                if len(b) < 2:
                    continue
                d = r - b
                try:
                    W, p = wilcoxon(r, b, alternative=alternative)
                except ValueError:
                    W, p = np.nan, 1.0
                lo, hi = bootstrap_median_ci(d, n_boot=n_boot)
                rows.append(dict(
                    model=model, train_pct=ratio, metric=metric, n_pairs=len(b),
                    mean_base=round(float(b.mean()), 3),
                    mean_red=round(float(r.mean()), 3),   # FIX: was `mead_red`
                    median_diff=round(float(np.median(d)), 3),
                    ci_lo=round(lo, 3), ci_hi=round(hi, 3),
                    rank_biserial=round(rank_biserial(d), 3),
                    W=W, p_value=p,
                ))
    res = pd.DataFrame(rows)
    res["q_bh"] = benjamini_hochberg(res["p_value"].fillna(1.0).values)
    res["significant"] = res["q_bh"] < 0.05

    mo = {mt: i for i, mt in enumerate(METRIC_ORDER)}
    ro = {rt: i for i, rt in enumerate(ratios)}
    res = res.sort_values(by=["train_pct", "metric", "model"],
                          key=lambda s: s.map(ro) if s.name == "train_pct"   # FIX: was `.mao`
                          else (s.map(mo) if s.name == "metric" else s)).reset_index(drop=True)
    return res, n_seeds, sorted(m.model.unique())


# --------------------------------------------------------------------------- #
# Markdown output
# --------------------------------------------------------------------------- #
def _num(x, d=3, signed=False):
    if pd.isna(x):
        return "–"
    return f"{x:+.{d}f}" if signed else f"{x:.{d}f}"


def _pval(p):
    if pd.isna(p):
        return "–"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def _md_table(headers, rows, align):
    sep = {"l": ":---", "r": "---:", "c": ":---:"}
    out = ["| " + " | ".join(headers) + " |",
           "| " + " | ".join(sep[a] for a in align) + " |"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def to_markdown(res: pd.DataFrame, n_seeds: int, title: str, multi_model: bool = False) -> str:
    """Render the results DataFrame as Markdown tables.

    multi_model=True  -> one "### <model>" section (table) per model.
    multi_model=False -> a single table (a Model column is added only if several models are present).
    Row order follows `res` (already sorted by train %, metric, model).
    """
    L = [f"## {title}\n",
         f"_Paired per-seed Wilcoxon signed-rank test, n = {n_seeds} seeds. "
         f"Median Δ = reduced − baseline (with bootstrap 95% CI). "
         f"r_rb = rank-biserial effect size. q = Benjamini–Hochberg FDR-adjusted p; "
         f"✔ = q < 0.05. AUC higher is better; for all other metrics lower is better._\n"]

    if res.empty:
        L.append("No results (not enough paired seeds).")
        return "\n".join(L) + "\n"

    def table(df: pd.DataFrame, with_model: bool) -> str:
        headers = (["Model"] if with_model else []) + [
            "train %", "Metric", "n", "Baseline", "Reduced",
            "Median Δ [95% CI]", "r_rb", "W", "p", "q (BH)", "Sig."]
        align = (["l"] if with_model else []) + ["r", "l", "r", "r", "r", "r", "r", "r", "r", "r", "c"]
        rows = []
        for _, r in df.iterrows():
            row = [r["model"]] if with_model else []
            row += [
                str(r["train_pct"]),
                METRIC_TITLES.get(r["metric"], r["metric"]),
                str(int(r["n_pairs"])),
                _num(r["mean_base"]), _num(r["mean_red"]),
                f"{_num(r['median_diff'], signed=True)} "
                f"[{_num(r['ci_lo'], signed=True)}, {_num(r['ci_hi'], signed=True)}]",
                _num(r["rank_biserial"], signed=True),
                _num(r["W"], 1),
                _pval(r["p_value"]), _pval(r["q_bh"]),
                "✔" if r["significant"] else "",
            ]
            rows.append(row)
        return _md_table(headers, rows, align)

    models = list(dict.fromkeys(res["model"]))          # keep order of appearance
    if multi_model and len(models) > 1:
        for mo in models:
            L.append(f"### {mo}\n")
            L.append(table(res[res["model"] == mo], with_model=False))
            L.append("")
    else:
        L.append(table(res, with_model=len(models) > 1))
        L.append("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description="Wilcoxon signed-rank: baseline vs familiarity-reduced")
    ap.add_argument("--baseline", default="results/metrics/setting_b_per_seed.csv")
    ap.add_argument("--reduced", default="results/metrics/setting_b_fam_effect_per_seed.csv")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--name", default="fam_singed_rank")
    ap.add_argument("--all-models", action="store_true",
                    help="write one Markdown section per model instead of a single table")
    ap.add_argument("--ratios", default="70 60 50 40 35 30 25 20 15 10")
    ap.add_argument("--metrics", default="EER,FAR,FRR,AUC,FAR_at_FRR1")
    ap.add_argument("--alternative", default="two-sided", choices=["two-sided", "greater", "less"])
    args = ap.parse_args()

    ratios = [int(x) for x in re.split(r"[,\s]+", args.ratios.strip()) if x]   # FIX: accepts "70 60" and "70,60"
    metrics = [x.strip() for x in args.metrics.split(",")]
    models = pd.read_csv(args.baseline)["model"].unique().tolist()

    res, n_seeds, all_models = compute(args.baseline, args.reduced, models, ratios, metrics, args.alternative)

    rdir = Path(args.results_dir)
    csv_path = rdir / "sing-rank-test" / f"{args.name}.csv"
    md_path = rdir / "tables" / f"{args.name}.md"
    csv_path.parent.mkdir(parents=True, exist_ok=True)   # FIX: create output dirs
    md_path.parent.mkdir(parents=True, exist_ok=True)

    res.to_csv(csv_path, index=False)
    title = "Signed-rank test - baseline vs familiarity-reduced"
    md_path.write_text(to_markdown(res, n_seeds, title, multi_model=args.all_models),
                       encoding="utf-8")                  # FIX: utf-8 (✔, Δ) – Windows default would fail
    pd.set_option("display.width", 200)
    print(res.to_string(index=False))
    print(f"Saved -> {csv_path} \n Saved -> {md_path}")


if __name__ == "__main__":
    main()