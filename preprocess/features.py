"""
Aggregate eye-tracking recording CSVs into one feature row per session.

`extract_features` augments the raw CSVs (fixations, saccades, 3d eye states)
with derived per-event columns. `aggregate_recording` then collapses each
recording into a single row by summarizing every feature's distribution with
median, skewness, kurtosis, coefficient of variation and median absolute
deviation, and by expressing event counts as per-second rates.

Importable helpers (not run directly; driven by preprocess_data.py):
    from preprocess.features import extract_features, aggregate_recording

    extract_features(recording_dir)              # augment raw CSVs in place
    aggregate_recording(recording_dir, "agg_session")  # -> agg_session.csv
"""

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# Configure logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

logger = logging.getLogger(__name__)


def summarize_series(series: pd.Series, name: str) -> dict:
    """Compute distribution-shape statistics for one feature series.

    Captures the shape of an individual's distribution (robust center,
    asymmetry, tailedness, scale-invariant spread and robust dispersion)
    rather than just its center and extremes:

    - median: robust center, insensitive to tracker outliers
    - skew: asymmetry of the distribution
    - kurtosis: tailedness / peakedness
    - cv: coefficient of variation (std / mean), scale-invariant spread
    - mad: median absolute deviation, robust dispersion
    """
    s = pd.to_numeric(series, errors="coerce").dropna()
    mean = s.mean()
    median = s.median()
    return {
        f"median_{name}": median,
        f"skew_{name}": s.skew(),
        f"kurtosis_{name}": s.kurt(),
        f"cv_{name}": s.std() / mean if mean else 0.0,
        f"mad_{name}": (s - median).abs().median(),
    }


def extract_features(data_dir: str) -> None:
    gaze = pd.read_csv(os.path.join(data_dir, "gaze.csv"))
    fixations = pd.read_csv(os.path.join(data_dir, "fixations.csv"))
    saccades = pd.read_csv(os.path.join(data_dir, "saccades.csv"))
    eye_states = pd.read_csv(os.path.join(data_dir, "3d_eye_states.csv"))

    ### FIXATIONS ###

    # Dispersion of each fixation (from gaze)
    dispersion_df = (
        gaze.groupby("fixation id")
        .apply(
            lambda g: (g["gaze x [px]"].max() - g["gaze x [px]"].min())
            + (g["gaze y [px]"].max() - g["gaze y [px]"].min())
        )
        .reset_index(name="dispersion [px]")
    )
    fixations = fixations.merge(dispersion_df, on="fixation id", how="left")

    # Euclidian Distance from previous fixation
    fixations["euclidian distance from previous [px]"] = np.sqrt(
        (fixations["fixation x [px]"].diff() ** 2)
        + (fixations["fixation y [px]"].diff() ** 2)
    ).fillna(0)

    # Angle from previous fixation
    fixations["angle from previous [rad]"] = np.arctan2(
        fixations["fixation y [px]"].diff(), fixations["fixation x [px]"].diff()
    ).fillna(0)

    # Time since previous fixation
    fixations["time since previous [ns]"] = (
        fixations["start timestamp [ns]"] - fixations["end timestamp [ns]"].shift(1)
    ).fillna(0)

    # Threshold for Screen/Keyboard
    Y = fixations["fixation y [px]"].values.reshape(-1, 1)
    kmeans = KMeans(n_clusters=2, n_init=10)
    labels = kmeans.fit_predict(Y)

    c1, c2 = kmeans.cluster_centers_.flatten()
    threshold = (c1 + c2) / 2
    fixations["threshold y [px]"] = threshold

    #  Transitions between screen - keyboard
    fixations["region"] = np.where(
        fixations["fixation y [px]"] > threshold, "keyboard", "screen"
    )

    fixations["transition"] = (
        fixations["region"] != fixations["region"].shift(1)
    ).astype(int)

    fixations.loc[0, "transition"] = 0

    ### SACCADES ###

    # Main sequence ratio (peak velocity / amplitude)
    saccades["main sequence ratio"] = (
        (saccades["peak velocity [px/s]"] / saccades["amplitude [px]"])
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    # Q Ratio
    saccades["Q ratio"] = (
        saccades["peak velocity [px/s]"] / saccades["mean velocity [px/s]"]
    ).fillna(0)

    ### 3d eye states ###
    xl, yl, zl = (
        eye_states["eyeball center left x [mm]"],
        eye_states["eyeball center left y [mm]"],
        eye_states["eyeball center left z [mm]"],
    )
    xr, yr, zr = (
        eye_states["eyeball center right x [mm]"],
        eye_states["eyeball center right y [mm]"],
        eye_states["eyeball center right z [mm]"],
    )
    eye_states["distance between pupils center [mm]"] = (
        (xr - xl) ** 2 + (yr - yl) ** 2 + (zr - zl) ** 2
    ) ** 0.5

    # Save updated files
    fixations.to_csv(os.path.join(data_dir, "fixations.csv"), index=False)
    saccades.to_csv(os.path.join(data_dir, "saccades.csv"), index=False)
    eye_states.to_csv(os.path.join(data_dir, "3d_eye_states.csv"), index=False)


def aggregate_recording(data_dir: str, name: str) -> None:
    fixations = pd.read_csv(os.path.join(data_dir, "fixations.csv"))
    saccades = pd.read_csv(os.path.join(data_dir, "saccades.csv"))
    eye_states = pd.read_csv(os.path.join(data_dir, "3d_eye_states.csv"))
    blinks = pd.read_csv(os.path.join(data_dir, "blinks.csv"))

    features = {}

    # Session duration is used only to normalize event counts into
    # per-second rates; it is not emitted as a feature itself.
    session_duration_s = (
        eye_states["timestamp [ns]"].max() - eye_states["timestamp [ns]"].min()
    ) / 1e9

    def rate(count: int) -> float:
        return count / session_duration_s if session_duration_s else 0.0

    ### Fixations ###
    features["screen_fixations_per_s"] = rate(
        len(fixations[fixations["region"] == "screen"])
    )
    features["screen_time_s"] = (
        fixations[fixations["region"] == "screen"]["duration [ms]"].sum() / 1e3
    )

    features["keyboard_fixations_per_s"] = rate(
        len(fixations[fixations["region"] == "keyboard"])
    )
    features["keyboard_time_s"] = (
        fixations[fixations["region"] == "keyboard"]["duration [ms]"].sum() / 1e3
    )

    features["transitions_per_s"] = rate(len(fixations[fixations["transition"] == 1]))

    features["path_length_px"] = fixations[
        "euclidian distance from previous [px]"
    ].sum()

    features.update(
        summarize_series(
            fixations["euclidian distance from previous [px]"],
            "distance_from_previous_px",
        )
    )
    features.update(
        summarize_series(
            fixations["angle from previous [rad]"], "angle_from_previous_rad"
        )
    )
    features.update(
        summarize_series(
            fixations["time since previous [ns]"] / 1e9, "time_since_previous_s"
        )
    )
    features.update(summarize_series(fixations["dispersion [px]"], "dispersion_px"))

    ### Saccades ###
    features["saccades_per_s"] = rate(len(saccades))

    features.update(
        summarize_series(saccades["duration [ms]"] / 1e3, "saccade_duration_s")
    )
    features.update(summarize_series(saccades["amplitude [px]"], "amplitude_px"))
    features.update(summarize_series(saccades["amplitude [deg]"], "amplitude_deg"))
    features.update(
        summarize_series(saccades["mean velocity [px/s]"], "mean_velocity_px_s")
    )
    features.update(
        summarize_series(saccades["peak velocity [px/s]"], "peak_velocity_px_s")
    )
    features.update(
        summarize_series(saccades["main sequence ratio"], "main_sequence_ratio")
    )
    features.update(summarize_series(saccades["Q ratio"], "Q_ratio"))

    ### Blinks ###
    features["blinks_per_s"] = rate(len(blinks))

    features.update(
        summarize_series(blinks["duration [ms]"] / 1e3, "blink_duration_s")
    )

    ### Eye States ###
    features.update(
        summarize_series(
            eye_states["pupil diameter left [mm]"], "pupil_diameter_left_mm"
        )
    )
    features.update(
        summarize_series(
            eye_states["pupil diameter right [mm]"], "pupil_diameter_right_mm"
        )
    )
    features.update(
        summarize_series(
            eye_states["eyelid angle top left [rad]"], "eyelid_angle_top_left"
        )
    )
    features.update(
        summarize_series(
            eye_states["eyelid angle bottom left [rad]"], "eyelid_angle_bottom_left"
        )
    )
    features.update(
        summarize_series(
            eye_states["eyelid angle top right [rad]"], "eyelid_angle_top_right"
        )
    )
    features.update(
        summarize_series(
            eye_states["eyelid angle bottom right [rad]"], "eyelid_angle_bottom_right"
        )
    )
    features.update(
        summarize_series(
            eye_states["eyelid aperture left [mm]"], "eyelid_aperture_left_mm"
        )
    )
    features.update(
        summarize_series(
            eye_states["eyelid aperture right [mm]"], "eyelid_aperture_right_mm"
        )
    )
    features.update(
        summarize_series(
            eye_states["distance between pupils center [mm]"],
            "distance_between_pupils_center_mm",
        )
    )

    df = pd.DataFrame([features]).fillna(0)

    df["label"] = (
        0
        if os.path.basename(os.path.dirname(Path(data_dir).parent)) == "impostors"
        else 1
    )

    filename = name + ".csv"
    out_path = os.path.join(data_dir, filename)
    df.to_csv(out_path, index=False)
    logger.info(f"Saved {name} aggregated features -> {out_path}")
