"""
Compare the results across sampling frequencies (T3.8).

Reads the tagged per-frequency outputs of setting_a.py / setting_b.py
(``setting_a_<f>hz_*.csv``, ``setting_b_<f>hz_*.csv``) and the published
200 Hz results (``setting_a_summary.csv``, ``setting_b_summary.csv``), and
writes:
  results/tables/frequency_comparison.md   tables + paired tests
  results/plots/frequency/eer_vs_frequency_<ratio>.png

Paired tests: Wilcoxon signed-rank on per-seed EER, each frequency vs the
200 Hz Option-B baseline, per model (same seeds = same splits).

Usage:
    python utils/frequency_tables.py --freqs 200 100 50 30 --ratios 70 30 15 10
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

METRICS = {"EER": "EER", "FAR_at_FRR1": "FAR@FRR=1%", "AUC": "AUC"}


def _cell(mean, lo, hi):
    return f"{mean:.2f} [{lo:.2f}-{hi:.2f}]"


def _table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    return "\n".join(out + ["| " + " | ".join(map(str, r)) + " |" for r in rows])


def _p(a, b):
    d = np.asarray(a) - np.asarray(b)
    if len(d) < 2 or np.allclose(d, 0):
        return np.nan
    return wilcoxon(a, b).pvalue


def _load_a(d, tag):
    name = f"setting_a_{tag}_summary.csv" if tag else "setting_a_summary.csv"
    path = os.path.join(d, name)
    return pd.read_csv(path, header=[0, 1], index_col=0) if os.path.exists(path) else None


def _load_b(d, tag, kind="summary"):
    name = f"setting_b_{tag}_{kind}.csv" if tag else f"setting_b_{kind}.csv"
    path = os.path.join(d, name)
    return pd.read_csv(path) if os.path.exists(path) else None


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metrics-dir", default="results/metrics")
    parser.add_argument("--freqs", type=int, nargs="+", default=[200, 100, 50, 30])
    parser.add_argument("--ratios", type=int, nargs="+", default=[70, 30, 15, 10])
    parser.add_argument("--out", default="results/tables/frequency_comparison.md")
    parser.add_argument("--plots", default="results/plots/frequency")
    args = parser.parse_args()
    d, base = args.metrics_dir, f"{args.freqs[0]}hz"
    cols = [("200 Hz published", "")] + [(f"{f} Hz", f"full_run{f}hz") for f in args.freqs]
    md = ["# Results across sampling frequencies (Option B)\n",
          f"Paired tests compare each frequency with the {args.freqs[0]} Hz Option-B baseline "
          "(Wilcoxon signed-rank on per-seed EER). The published column uses the vendor events.\n"]

    # ---- Setting A ----
    sa = {label: _load_a(d, tag) for label, tag in cols}
    sa = {k: v for k, v in sa.items() if v is not None}
    models = list(next(iter(sa.values())).index)
    for key, label in METRICS.items():
        rows = [[m] + [_cell(t.loc[m, (key, "mean")], t.loc[m, (key, "ci_lo")], t.loc[m, (key, "ci_hi")])
                       if m in t.index else "-" for t in sa.values()] for m in models]
        md += [f"## Setting A (70/30): {label}\n", _table(["Model"] + list(sa), rows), ""]

    pa = {f: pd.read_csv(os.path.join(d, f"setting_a_{f}hz_per_seed.csv"))
          for f in args.freqs if os.path.exists(os.path.join(d, f"setting_a_{f}hz_per_seed.csv"))}
    if base[:-2].isdigit() and int(base[:-2]) in pa:
        ref = pa[int(base[:-2])]
        rows = []
        for m in models:
            r = ref[ref["model"] == m].set_index("seed")["EER"]
            cells = []
            for f in args.freqs[1:]:
                if f not in pa:
                    cells.append("-")
                    continue
                x = pa[f][pa[f]["model"] == m].set_index("seed")["EER"]
                common = r.index.intersection(x.index)
                if common.empty:
                    cells.append("-")
                    continue
                cells.append(f"{(x.loc[common] - r.loc[common]).median():+.2f} (p={_p(x.loc[common], r.loc[common]):.3g})")
            rows.append([m] + cells)
        md += [f"### Setting A: median EER change vs {args.freqs[0]} Hz, pp (Wilcoxon p)\n",
               _table(["Model"] + [f"{f} Hz" for f in args.freqs[1:]], rows), ""]

    # ---- Setting B ----
    sb = {label: _load_b(d, tag) for label, tag in cols}
    sb = {k: v for k, v in sb.items() if v is not None}
    pb = {f: _load_b(d, f"{f}hz", "per_seed") for f in args.freqs}
    os.makedirs(args.plots, exist_ok=True)
    for pct in args.ratios:
        rows = []
        for m in models:
            cells = []
            for t in sb.values():
                r = t[(t["train_pct"] == pct) & (t["model"] == m)]
                cells.append(_cell(*r[["EER_mean", "EER_ci_lo", "EER_ci_hi"]].iloc[0]) if len(r) else "-")
            ref = pb.get(args.freqs[0])
            last = pb.get(args.freqs[-1])
            if ref is not None and last is not None:
                a = ref[(ref["train_pct"] == pct) & (ref["model"] == m)].set_index("seed")["EER"]
                b = last[(last["train_pct"] == pct) & (last["model"] == m)].set_index("seed")["EER"]
                common = a.index.intersection(b.index)
                cells.append(f"{_p(b.loc[common], a.loc[common]):.3g}" if len(common) else "-")
            rows.append([m] + cells)
        head = ["Model"] + list(sb) + [f"p ({args.freqs[-1]} vs {args.freqs[0]} Hz)"]
        md += [f"## Setting B, {pct}% training: EER\n", _table(head, rows), ""]

        # figure: EER vs frequency at this ratio
        plt.figure(figsize=(7, 4.5))
        for m in models:
            xs, ys, lo, hi = [], [], [], []
            for f in args.freqs:
                t = sb.get(f"{f} Hz")
                if t is None:
                    continue
                r = t[(t["train_pct"] == pct) & (t["model"] == m)]
                if len(r):
                    xs.append(f)
                    ys.append(r["EER_mean"].iloc[0])
                    lo.append(r["EER_ci_lo"].iloc[0])
                    hi.append(r["EER_ci_hi"].iloc[0])
            if xs:
                plt.plot(xs, ys, marker="o", lw=2, label=m)
                plt.fill_between(xs, lo, hi, alpha=0.15)
        plt.gca().invert_xaxis()
        plt.xticks(args.freqs, [str(f) for f in args.freqs])
        plt.xlabel("Sampling frequency (Hz)")
        plt.ylabel("EER (%) - mean [95% CI]")
        plt.title(f"EER vs sampling frequency, {pct}% training")
        plt.ylim(bottom=0)
        plt.grid(alpha=0.3)
        plt.legend()
        path = f"{args.plots}/eer_vs_frequency_{pct}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        md.append(f"Figure: `{path}`\n")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    open(args.out, "w", encoding="utf-8").write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()