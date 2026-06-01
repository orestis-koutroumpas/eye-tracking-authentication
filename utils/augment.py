"""
Training-set augmentation for the aggregated eye-tracking sessions.

SMOTE-style augmentation: for each session we keep the original row and add
``n_copies`` synthetic rows by interpolating towards same-class neighbours.
Each synthetic row is anchored to its parent session and inherits the parent's
group id, so GroupKFold keeps all copies of a session in the same fold.

Importable helper (not run directly):
    from utils.augment import smote_augment

    X_aug, y_aug, groups_aug = smote_augment(
        X_train, y_train, groups_train, n_copies=2, random_state=seed
    )
"""

import logging

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger(__name__)


def smote_augment(X, y, groups, n_copies=2, k_neighbors=5, random_state=42):
    """Augment each session into 1 + n_copies rows via SMOTE-style interpolation.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (rows = sessions).
    y : array-like
        Class labels aligned with X.
    groups : array-like
        Group id per row (impostor user / genuine session); synthetic copies
        inherit their parent row's group.
    n_copies : int
        Number of synthetic copies to add per original row (total per row =
        1 original + n_copies).
    k_neighbors : int
        Neighbourhood size for interpolation (clipped to class size - 1).
    random_state : int
        Seed for reproducible interpolation.

    Returns
    -------
    X_aug : pd.DataFrame
    y_aug : pd.Series
    groups_aug : pd.Series
    """
    rng = np.random.default_rng(random_state)

    X = X.reset_index(drop=True)
    y = pd.Series(np.asarray(y)).reset_index(drop=True)
    groups = pd.Series(np.asarray(groups)).reset_index(drop=True)
    columns = X.columns
    X_values = X.to_numpy(dtype=float)

    X_parts = [X_values]
    y_parts = [y.to_numpy()]
    g_parts = [groups.to_numpy()]

    for label in np.unique(y):
        cls_idx = np.where(y.to_numpy() == label)[0]
        if len(cls_idx) < 2:
            # Not enough samples to interpolate; originals already kept.
            logger.warning(
                f"Class {label} has {len(cls_idx)} sample(s); skipping augmentation."
            )
            continue

        cls_X = X_values[cls_idx]
        k = min(k_neighbors, len(cls_idx) - 1)
        nn = NearestNeighbors(n_neighbors=k + 1).fit(cls_X)  # +1 includes self
        neigh = nn.kneighbors(cls_X, return_distance=False)[:, 1:]  # drop self

        for _ in range(n_copies):
            # For each class sample pick one of its neighbours and interpolate.
            chosen = neigh[np.arange(len(cls_idx)), rng.integers(0, k, len(cls_idx))]
            gaps = rng.random((len(cls_idx), 1))
            synth = cls_X + gaps * (cls_X[chosen] - cls_X)
            X_parts.append(synth)
            y_parts.append(np.full(len(cls_idx), label))
            g_parts.append(groups.to_numpy()[cls_idx])  # inherit parent group

    X_aug = pd.DataFrame(np.vstack(X_parts), columns=columns)
    y_aug = pd.Series(np.concatenate(y_parts), name=y.name)
    groups_aug = pd.Series(np.concatenate(g_parts), name="group")

    logger.info(
        f"Augmented training set: {len(X)} -> {len(X_aug)} rows "
        f"(x{1 + n_copies}, SMOTE-style)"
    )
    return X_aug, y_aug, groups_aug
