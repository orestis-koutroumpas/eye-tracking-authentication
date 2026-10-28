"""
Pilot check of the MLP search space (T2.3): edge choices, ties, convergence.

Reads a per-seed CSV written by setting_a.py / setting_b.py (with the
``mlp_n_iter`` and ``n_tied_best`` columns) and prints, per training ratio:
how often the best inner-CV score was tied, how many fits hit max_iter, and
how often a boundary value of the grid was chosen in seeds WITHOUT a tie
(a tied choice only reflects the simplest-first tie-break, not a preference).

Usage:
    python utils/check_grid_edges.py results/metrics/setting_b_mlp_pilot_per_seed.csv
"""

import ast
import sys

import pandas as pd

MAX_ITER = 2000
EDGES = {
    "clf__alpha": [10.0, 1e-4],
    "clf__hidden_layer_sizes": [(8,), (64,)],
}


def main(path):
    df = pd.read_csv(path)
    df = df[df["model"] == "MLP"].copy()
    params = df["best_params"].apply(ast.literal_eval)
    for key in EDGES:
        df[key] = params.apply(lambda p, k=key: p[k])
    df["untied"] = df["n_tied_best"] == 1

    groups = df.groupby("train_pct") if "train_pct" in df else [(70, df)]
    for pct, g in groups:
        untied = g[g["untied"]]
        print(
            f"\n== train {pct}% | seeds={len(g)} | best score tied in "
            f"{1 - g['untied'].mean():.0%} of seeds | not converged: "
            f"{int((g['mlp_n_iter'] >= MAX_ITER).sum())}"
        )
        for key, edges in EDGES.items():
            counts = g[key].astype(str).value_counts().to_dict()
            if len(untied):
                share = untied[key].apply(lambda v, e=edges: v in e).mean()
                flag = "  <-- extend the grid" if len(untied) >= 3 and share > 0.3 else ""
                edge_txt = f"{share:.0%} of {len(untied)} untied seeds at an edge{flag}"
            else:
                edge_txt = "no untied seeds (choice set by the tie-break)"
            print(f"  {key}: {counts}\n      {edge_txt}")


if __name__ == "__main__":
    main(sys.argv[1])