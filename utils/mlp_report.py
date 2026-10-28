"""
Collect every MLP number the paper needs into one Markdown file (T2.7).

Reads the merged CSVs (after utils/merge_results.py) and writes
``results/tables/mlp_report.md`` with:
  1. Setting A: MLP vs RBF SVM, all metrics, mean [95% CI];
  2. Setting B: MLP per ratio, all metrics, plus the RBF SVM EER;
  3. augmentation effect on the MLP EER at ratios <= 30%;
  4. selected configuration per ratio (most frequent), how often, ties, iterations;
  5. paired Wilcoxon signed-rank test MLP vs RBF SVM on per-seed EER.

Usage:
    python utils/mlp_report.py
"""

import argparse
import ast
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

METRICS = ["EER", "FAR", "FRR", "AUC", "FAR_at_FRR1"]
LABELS = {"EER": "EER", "FAR": "FAR", "FRR": "FRR", "AUC": "AUC", "FAR_at_FRR1": "FAR@FRR=1%"}


def _cell(mean, lo, hi):
    return f"{mean:.2f} [{lo:.2f}-{hi:.2f}]"


def _table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _wilcoxon(a, b):
    """Paired two-sided Wilcoxon on aligned per-seed values; (median diff, p)."""
    d = np.asarray(a) - np.asarray(b)
    if np.allclose(d, 0):
        return 0.0, 1.0
    return float(np.median(d)), float(wilcoxon(a, b).pvalue)


def _paired(per_seed, model_a, model_b, keys):
    a = per_seed[per_seed["model"] == model_a].set_index(keys)["EER"]
    b = per_seed[per_seed["model"] == model_b].set_index(keys)["EER"]
    common = a.index.intersection(b.index)
    return a.loc[common], b.loc[common]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metrics-dir", default="results/metrics")
    parser.add_argument("--out", default="results/tables/mlp_report.md")
    parser.add_argument("--ref", default="RBF SVM", help="model to compare the MLP with")
    args = parser.parse_args()
    d = args.metrics_dir
    md = ["# MLP results\n"]

    # 1. Setting A
    sa = pd.read_csv(os.path.join(d, "setting_a_summary.csv"), header=[0, 1], index_col=0)
    rows = [[m] + [_cell(sa.loc[m, (k, "mean")], sa.loc[m, (k, "ci_lo")], sa.loc[m, (k, "ci_hi")])
                   for k in METRICS] for m in ["MLP", args.ref] if m in sa.index]
    md += ["## Setting A (70/30)\n", _table(["Model"] + [LABELS[k] for k in METRICS], rows), ""]
    pa = pd.read_csv(os.path.join(d, "setting_a_per_seed.csv"))
    a, b = _paired(pa, "MLP", args.ref, ["seed"])
    med, p = _wilcoxon(a, b)
    md += [f"Paired Wilcoxon, MLP - {args.ref} EER over {len(a)} seeds: median diff {med:+.2f} pp, p = {p:.3g}\n"]

    # 2. Setting B
    sb = pd.read_csv(os.path.join(d, "setting_b_summary.csv"))
    mlp = sb[sb["model"] == "MLP"].set_index("train_pct").sort_index(ascending=False)
    ref = sb[sb["model"] == args.ref].set_index("train_pct")
    rows = [[pct] + [_cell(r[f"{k}_mean"], r[f"{k}_ci_lo"], r[f"{k}_ci_hi"]) for k in METRICS]
            + [f"{ref.loc[pct, 'EER_mean']:.2f}" if pct in ref.index else "-"]
            for pct, r in mlp.iterrows()]
    md += ["## Setting B: MLP per training ratio\n",
           _table(["Train %"] + [LABELS[k] for k in METRICS] + [f"{args.ref} EER"], rows), ""]

    # 3. Augmentation effect (ratios where augmentation is applied)
    no_aug_path = os.path.join(d, "setting_b_no_augment_summary.csv")
    if os.path.exists(no_aug_path):
        na = pd.read_csv(no_aug_path)
        na = na[na["model"] == "MLP"].set_index("train_pct")
        rows = [[pct, f"{mlp.loc[pct, 'EER_mean']:.2f}", f"{na.loc[pct, 'EER_mean']:.2f}",
                 f"{mlp.loc[pct, 'EER_mean'] - na.loc[pct, 'EER_mean']:+.2f}"]
                for pct in sorted(mlp.index, reverse=True) if pct <= 30 and pct in na.index]
        md += ["## Augmentation effect on MLP EER (ratios <= 30%)\n",
               _table(["Train %", "With aug.", "No aug.", "Delta"], rows), ""]

    # 4. Selected configurations + 5. paired test, per ratio
    pb = pd.read_csv(os.path.join(d, "setting_b_per_seed.csv"))
    pm = pb[pb["model"] == "MLP"].copy()
    pm["config"] = pm["best_params"].apply(
        lambda s: ", ".join(f"{k.split('__')[1]}={v}" for k, v in sorted(ast.literal_eval(s).items())))
    rows = []
    for pct, g in sorted(pm.groupby("train_pct"), key=lambda x: -x[0]):
        top = g["config"].value_counts()
        a, b = _paired(pb[pb["train_pct"] == pct], "MLP", args.ref, ["seed"])
        med, p = _wilcoxon(a, b)
        tied = (g["n_tied_best"] > 1).mean() if "n_tied_best" in g else np.nan
        n_iter = g["mlp_n_iter"].median() if "mlp_n_iter" in g else np.nan
        rows.append([pct, top.index[0], f"{top.iloc[0] / len(g):.0%}", f"{tied:.0%}",
                     f"{n_iter:.0f}", f"{med:+.2f}", f"{p:.3g}"])
    md += ["## Setting B: selected MLP configuration and paired test vs " + args.ref + "\n",
           _table(["Train %", "Most frequent configuration", "Share of seeds", "Seeds with tied best",
                   "Median iterations", f"Median EER diff vs {args.ref} (pp)", "Wilcoxon p"], rows), ""]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print("\n".join(md))
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()