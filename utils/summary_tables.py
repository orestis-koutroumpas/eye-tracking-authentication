"""
Build presentation-ready summary tables from the experiment metric summaries.

Reads the per-setting summary CSVs written by setting_a.py / setting_b.py and
renders compact Markdown tables (mean [95% CI]) suitable for a report or README.

Reads:
    results/metrics/setting_a_summary.csv             per-model, MultiIndex (metric, stat)
    results/metrics/setting_b_summary.csv             per train_pct x model, flat columns
    results/metrics/setting_b_summary_no_augment.csv  same layout, no augmentation

Writes (and prints):
    results/tables/setting_a_summary.md               one row per model
    results/tables/setting_b_summary.md               one table per metric, train_pct x model
    results/tables/setting_b_train_composition.md     train-set composition by ratio
    results/tables/setting_b_augment_comparison.md    EER augment vs no-augment (train < 35%)

Usage:
    python utils/summary_tables.py
"""

import ast
import logging
import os

import pandas as pd

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Repo root = parent of this utils/ directory, so the script works regardless of
# the current working directory.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METRICS_DIR = os.path.join(REPO_ROOT, "results", "metrics")
TABLES_DIR = os.path.join(REPO_ROOT, "results", "tables")

# Metrics shared by both settings, in display order, with human-readable labels.
# All are percentages; for AUC higher is better, for the rest lower is better.
METRICS = ["EER", "FAR", "FRR", "AUC", "FAR_at_FRR1"]
METRIC_LABELS = {
    "EER": "EER",
    "FAR": "FAR",
    "FRR": "FRR",
    "AUC": "AUC",
    "FAR_at_FRR1": "FAR @ FRR=1%",
}

# Only train ratios strictly below this threshold are included in the
# augment vs no-augment comparison table.
LOW_DATA_THRESHOLD = 35


def _md_table(headers, rows):
    """Render a list of header strings and row lists as a Markdown table."""
    line = lambda cells: "| " + " | ".join(str(c) for c in cells) + " |"
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    return "\n".join([line(headers), sep, *(line(r) for r in rows)])


def _ci_cell(mean, lo, hi):
    """Format a 'mean [lo-hi]' cell at two decimals."""
    return f"{mean:.2f} [{lo:.2f}-{hi:.2f}]"


def _fmt_params(raw):
    """Render a stored best_params dict as 'key=val, ...', dropping pipeline
    step prefixes (clf__ / selector__) for readability."""
    try:
        params = ast.literal_eval(str(raw))
    except (ValueError, SyntaxError):
        return str(raw)
    return ", ".join(f"{k.split('__')[-1]}={v}" for k, v in params.items())


def build_setting_a_table():
    """Per-model headline table: mean [95% CI] for each metric, plus n_features.

    setting_a_summary.csv has a two-row (metric, stat) column header and the
    model name as the index; rows are pre-sorted by EER mean (best first).
    """
    path = os.path.join(METRICS_DIR, "setting_a_summary.csv")
    df = pd.read_csv(path, header=[0, 1], index_col=0)
    has_params = ("best_params", "mode") in df.columns

    headers = ["Model"] + [METRIC_LABELS[m] for m in METRICS] + ["n_features"]
    if has_params:
        headers.append("hyperparameters")
    rows = []
    for model, row in df.iterrows():
        cells = [model]
        for m in METRICS:
            cells.append(_ci_cell(row[(m, "mean")], row[(m, "ci_lo")], row[(m, "ci_hi")]))
        cells.append(f"{row[('n_features', 'median')]:.0f}")
        if has_params:
            cells.append(_fmt_params(row[("best_params", "mode")]))
        rows.append(cells)

    title = "## Setting A - idealized 70/30 split"
    note = "_Values in %, mean [95% bootstrap CI]. FAR/FRR at the EER threshold; AUC higher is better._"
    return "\n\n".join([title, note, _md_table(headers, rows)])


def build_setting_b_tables():
    """One table per metric: rows = train %, columns = model, cells = mean [95% CI].

    setting_b_summary.csv is flat (one row per train_pct x model) with
    <METRIC>_mean / _ci_lo / _ci_hi columns. Train fractions are shown largest
    first so the tables read from most to least training data.
    """
    path = os.path.join(METRICS_DIR, "setting_b_summary.csv")
    df = pd.read_csv(path)

    models = list(dict.fromkeys(df["model"]))  # order of first appearance
    train_pcts = sorted(df["train_pct"].unique(), reverse=True)

    sections = [
        "## Setting B - performance vs training size",
        "_Values in %, mean [95% bootstrap CI]. Rows = train % of sessions; AUC higher is better._",
    ]
    for m in METRICS:
        headers = ["train %"] + models
        rows = []
        for pct in train_pcts:
            cells = [f"{pct}"]
            for model in models:
                r = df[(df["train_pct"] == pct) & (df["model"] == model)]
                if r.empty:
                    cells.append("-")
                else:
                    r = r.iloc[0]
                    cells.append(_ci_cell(r[f"{m}_mean"], r[f"{m}_ci_lo"], r[f"{m}_ci_hi"]))
            rows.append(cells)
        sections.append(f"### {METRIC_LABELS[m]}")
        sections.append(_md_table(headers, rows))

    # Most frequent hyperparameter set per ratio x model (if recorded).
    if "best_params" in df.columns:
        headers = ["train %"] + models
        rows = []
        for pct in train_pcts:
            cells = [f"{pct}"]
            for model in models:
                r = df[(df["train_pct"] == pct) & (df["model"] == model)]
                cells.append("-" if r.empty else _fmt_params(r.iloc[0]["best_params"]))
            rows.append(cells)
        sections.append("### Hyperparameters (most common over seeds)")
        sections.append(_md_table(headers, rows))
    return "\n\n".join(sections)


def build_setting_b_composition_table():
    """Training-set composition per train fraction (genuine/impostor counts).

    Mirrors setting_b_train_composition.csv: how many genuine sessions, impostor
    sessions and impostor users go into training at each ratio, and whether
    Gaussian-noise augmentation was applied (with the resulting sample count).
    """
    path = os.path.join(METRICS_DIR, "setting_b_train_composition.csv")
    df = pd.read_csv(path).sort_values("train_pct", ascending=False)

    headers = [
        "train %", "genuine", "impostor sessions", "impostor users",
        "total train", "augmented", "total after aug",
    ]
    rows = [
        [
            r["train_pct"], r["genuine"], r["impostor_sessions"],
            r["impostor_users"], r["total_train"],
            "yes" if r["augmented"] else "no", r["total_after_aug"],
        ]
        for _, r in df.iterrows()
    ]

    title = "## Setting B - training set composition by ratio"
    note = ("_Counts of training samples per train %. Ratios of 50% and below "
            "(train no larger than test) are augmented with 3 Gaussian-noise "
            "copies per class (4x)._")
    return "\n\n".join([title, note, _md_table(headers, rows)])


def build_augment_comparison_table():
    """EER comparison: augmented vs no-augment, one flat table, train < 35% only.

    Single table with 4 rows per train ratio (one per model). Columns:
    train %, model, EER (aug), EER (no aug), Δ EER.
    Δ EER = aug − no_aug: negative means augmentation helped (lower EER),
    positive means it hurt.
    """
    aug_path = os.path.join(METRICS_DIR, "setting_b_summary.csv")
    no_aug_path = os.path.join(METRICS_DIR, "setting_b_summary_no_augment.csv")

    if not os.path.exists(aug_path) or not os.path.exists(no_aug_path):
        logger.warning(
            "Skipping augment comparison: one or both summary CSVs not found "
            f"({aug_path}, {no_aug_path})"
        )
        return None

    aug = pd.read_csv(aug_path)
    no_aug = pd.read_csv(no_aug_path)

    # Restrict to ratios strictly below the threshold.
    aug = aug[aug["train_pct"] < LOW_DATA_THRESHOLD]
    no_aug = no_aug[no_aug["train_pct"] < LOW_DATA_THRESHOLD]

    models = list(dict.fromkeys(aug["model"]))  # order of first appearance
    train_pcts = sorted(aug["train_pct"].unique(), reverse=True)

    headers = ["train %", "model", "EER (augmented)", "EER (no augment)", "Δ EER"]
    rows = []
    for pct in train_pcts:
        for model in models:
            a = aug[(aug["train_pct"] == pct) & (aug["model"] == model)]
            n = no_aug[(no_aug["train_pct"] == pct) & (no_aug["model"] == model)]

            if a.empty or n.empty:
                rows.append([f"{pct}", model, "-", "-", "-"])
                continue

            a, n = a.iloc[0], n.iloc[0]
            aug_cell = _ci_cell(a["EER_mean"], a["EER_ci_lo"], a["EER_ci_hi"])
            no_aug_cell = _ci_cell(n["EER_mean"], n["EER_ci_lo"], n["EER_ci_hi"])

            diff = a["EER_mean"] - n["EER_mean"]
            sign = "+" if diff >= 0 else "−"
            diff_cell = f"{sign}{abs(diff):.2f}"

            rows.append([f"{pct}", model, aug_cell, no_aug_cell, diff_cell])

    title = "## Setting B - augmentation effect on EER (train < 35%)"
    note = (
        "_EER mean [95% bootstrap CI]. "
        "Δ EER = aug − no_aug: negative (−) means augmentation reduced EER (improved); "
        "positive (+) means it increased EER (hurt)._"
    )
    return "\n\n".join([title, note, _md_table(headers, rows)])


def _write(name, content):
    os.makedirs(TABLES_DIR, exist_ok=True)
    path = os.path.join(TABLES_DIR, name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content + "\n")
    logger.info(f"Saved table -> {path.replace(os.sep, '/')}")
    print("\n" + content + "\n")


def main():
    _write("setting_a_summary.md", build_setting_a_table())
    _write("setting_b_summary.md", build_setting_b_tables())
    _write("setting_b_train_composition.md", build_setting_b_composition_table())

    comparison = build_augment_comparison_table()
    if comparison is not None:
        _write("setting_b_augment_comparison.md", comparison)


if __name__ == "__main__":
    main()