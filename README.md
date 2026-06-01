# Eye Tracking Authentication

This repository contains work from my thesis titled
**"Study and Development of a Secure Authentication System Using Eye Tracking."**

## Overview

This thesis investigates the use of **eye movements and physiological eye features** as an additional security layer for authentication systems.

Traditional authentication methods such as passwords and PINs are widely used but remain vulnerable to attacks and credential compromise. Biometric authentication methods offer improved security, as biometric traits are generally more difficult to steal or spoof. Eye movements represent a form of **behavioral biometrics** and are closely linked to perception and cognition, making them difficult to consciously control or imitate.

The core hypothesis of this research is that **genuine users** entering their credentials exhibit eye movement patterns that differ from those of **impostors**, even when both use the same valid credentials.

---

## Methodology

Eye-tracking data were collected during password-entry tasks from both genuine users and impostors. Each recording is preprocessed and reduced to one feature vector per session (`preprocess_data.py` → per-session aggregated features), combining eye-movement statistics and physiological eye characteristics. Authentication is framed as a binary classification problem: **genuine = 1**, **impostor = 0**.

Four classifiers are evaluated:

- Logistic Regression
- Linear Support Vector Machine (Linear SVM)
- Support Vector Machine with RBF kernel (RBF SVM)
- Random Forest

### Model pipeline

Every model is trained as a single scikit-learn `Pipeline`:

```
StandardScaler  →  RFE feature selection  →  classifier
```

- **Scaling** is fit inside each cross-validation fold, so no test-fold statistics leak into selection or training.
- **Feature selection** uses **Recursive Feature Elimination (RFE)** driven by a fast linear ranker (`LinearSVC`, dropping 10% of features per step). The same linear ranker is used regardless of the final classifier, so all models share one consistent feature ranking. The target subset size (`10 / 20 / 40 / 80` features) is itself a tuned hyperparameter.
- **Hyperparameters** (the subset size plus each model's own grid) are selected by a nested `GridSearchCV` that **optimizes the Equal Error Rate (EER)** on continuous scores.
- **Cross-validation** uses `StratifiedGroupKFold`, grouped by impostor user, so no subject appears on both sides of a fold.

### Evaluation metrics

All metrics are derived from each model's continuous scores on the held-out test set and reported in percent:

| Metric | Meaning |
| --- | --- |
| **EER** | Equal Error Rate — operating point where FAR = FRR (primary objective) |
| **FAR** | False Acceptance Rate (impostor accepted), at the EER threshold |
| **FRR** | False Rejection Rate (genuine rejected), at the EER threshold |
| **AUC** | Area under the ROC curve (threshold-free ranking quality) |
| **FAR @ FRR=1%** | Residual security risk when user friction is fixed at 1% rejections |

Each experiment is repeated over **100 random splits (seeds)**; results are summarized as **mean [95% bootstrap CI]** (and median [IQR]) per model.

---

## Experimental Setup and Results

Two experimental settings are examined. The `results/` outputs are **not committed** to the repository — they are produced by running the scripts (see [Reproducing the results](#reproducing-the-results)). The headline tables below are reproduced from a representative run.

### Setting A — idealized condition (70/30 split)

A conventional 70% train / 30% test split. All four models are strong; the linear models and the RBF SVM are statistically tied at the top.

_Values in %, mean [95% bootstrap CI]. FAR/FRR at the EER threshold; AUC higher is better._

| Model | EER | FAR | FRR | AUC | FAR @ FRR=1% | n_features | hyperparameters |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Linear SVM | 1.19 [0.85-1.56] | 1.36 [1.00-1.74] | 1.03 [0.69-1.39] | 99.84 [99.75-99.91] | 1.99 [1.35-2.72] | 40 | C=0.01, n_features_to_select=40 |
| RBF SVM | 1.19 [0.83-1.58] | 1.21 [0.81-1.65] | 1.17 [0.83-1.53] | 99.89 [99.84-99.93] | 2.72 [1.79-3.80] | 20 | C=0.1, gamma=scale, n_features_to_select=10 |
| Logistic Regression | 1.53 [1.14-1.98] | 1.84 [1.37-2.37] | 1.22 [0.86-1.64] | 99.71 [99.56-99.83] | 3.20 [2.23-4.38] | 40 | C=0.01, n_features_to_select=40 |
| Random Forest | 2.77 [2.26-3.30] | 2.76 [2.21-3.34] | 2.78 [2.28-3.31] | 99.64 [99.55-99.73] | 7.67 [6.00-9.52] | 20 | max_depth=None, n_estimators=100, n_features_to_select=10 |

### Setting B — realistic condition (how little training data is enough?)

The same fraction is applied to genuine sessions and impostor users across a sweep from 70% down to 10%. Ratios where the training set is no larger than the test set (**50% and below**) are augmented with Gaussian-noise copies (3 synthetic copies per class, applied to training rows only, after the split). The EER-vs-training-size table locates the smallest training fraction at which each model stays reliable.

**EER (%), mean [95% bootstrap CI]:**

| train % | Linear SVM | Logistic Regression | RBF SVM | Random Forest |
| --- | --- | --- | --- | --- |
| 70 | 1.19 [0.85-1.56] | 1.53 [1.14-1.98] | 1.19 [0.83-1.58] | 2.77 [2.26-3.30] |
| 60 | 1.49 [1.13-1.90] | 1.84 [1.45-2.27] | 1.29 [0.97-1.64] | 3.02 [2.51-3.54] |
| 50 | 2.07 [1.65-2.53] | 2.36 [1.93-2.81] | 1.68 [1.36-2.01] | 3.53 [3.00-4.09] |
| 45 | 2.53 [2.11-2.96] | 2.57 [2.16-3.02] | 1.84 [1.43-2.28] | 3.77 [3.24-4.31] |
| 40 | 3.20 [2.69-3.74] | 3.22 [2.73-3.73] | 2.30 [1.88-2.75] | 4.13 [3.63-4.65] |
| 35 | 3.46 [2.97-3.97] | 3.88 [3.35-4.45] | 2.77 [2.34-3.22] | 4.84 [4.28-5.42] |
| 30 | 3.42 [2.96-3.89] | 3.47 [3.04-3.93] | 2.83 [2.43-3.25] | 5.08 [4.60-5.56] |
| 25 | 5.28 [4.41-6.25] | 5.34 [4.50-6.28] | 3.20 [2.80-3.62] | 5.80 [5.31-6.33] |
| 20 | 6.63 [5.71-7.62] | 6.58 [5.71-7.48] | 3.79 [3.35-4.24] | 6.79 [6.14-7.46] |
| 15 | 7.99 [6.83-9.24] | 7.76 [6.62-8.99] | 3.82 [3.36-4.33] | 8.33 [7.37-9.35] |
| 10 | 16.08 [14.20-18.06] | 16.20 [14.28-18.21] | 6.94 [6.14-7.81] | 14.24 [12.76-15.90] |

The full per-metric breakdown (FAR, FRR, AUC, FAR @ FRR=1%) and the most-common hyperparameters per ratio are written to `results/tables/setting_b_summary.md` when you run the scripts. The **RBF SVM degrades most gracefully**, holding an EER below ~7% even at the smallest 10% training fraction, where the linear models climb above 16%.

**Training-set composition by ratio:**

| train % | genuine | impostor sessions | impostor users | total train | augmented | total after aug |
| --- | --- | --- | --- | --- | --- | --- |
| 70 | 83 | 79 | 12 | 162 | no | 162 |
| 60 | 71 | 66 | 10 | 137 | no | 137 |
| 50 | 59 | 59 | 9 | 118 | yes | 472 |
| 45 | 53 | 52 | 8 | 105 | yes | 420 |
| 40 | 47 | 46 | 7 | 93 | yes | 372 |
| 35 | 41 | 40 | 6 | 81 | yes | 324 |
| 30 | 35 | 33 | 5 | 68 | yes | 272 |
| 25 | 29 | 27 | 4 | 56 | yes | 224 |
| 20 | 23 | 20 | 3 | 43 | yes | 172 |
| 15 | 17 | 14 | 2 | 31 | yes | 124 |
| 10 | 11 | 7 | 1 | 18 | yes | 72 |

---

## Reproducing the results

The `results/` directory is generated, not committed. Starting from the raw recordings, run the preprocessing step first to build the per-session feature dataset, then run the experiments:

```bash
python preprocess_data.py --data_dir data/raw_data   # raw recordings -> per-session features
python setting_a.py              # idealized 70/30 split -> results/metrics/setting_a_*.csv
python setting_b.py              # training-size sweep    -> results/metrics/setting_b_*.csv
python utils/summary_tables.py   # render Markdown tables -> results/tables/*.md
```

`preprocess_data.py` must be run first — the experiment scripts consume the aggregated features it produces. Raw per-seed records and aggregated summaries are then written under `results/metrics/`, and the formatted Markdown tables shown above under `results/tables/`.

---

## Conclusions

The results demonstrate that eye movement features captured during password entry contain **discriminative information** capable of distinguishing genuine users from impostors. Under an idealized split all models reach an EER around 1–3%, and even with as little as 10% of the data for training the RBF SVM keeps the EER below ~7%. These findings support the feasibility of **eye tracking as a complementary authentication factor** to enhance the security of traditional password-based systems.

This work provides a foundation for further research into the use of behavioral biometrics in secure authentication systems.

---

## License

This project is provided for academic and research purposes.
