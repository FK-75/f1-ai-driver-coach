"""
alignment.py — Distance-axis DTW alignment for cross-lap comparison.

The core technical insight of this project: two laps cannot be compared
on a timestamp basis (a faster driver covers the same corner sooner), but
they CAN be compared on a distance basis. We use the distance axis as the
common reference and apply Dynamic Time Warping (DTW) only as a refinement
step to handle GPS/telemetry jitter in the distance measurement itself.
"""

import numpy as np
import pandas as pd
from scipy.signal import correlate
from typing import Tuple


def align_to_common_grid(
    driver_df: pd.DataFrame,
    reference_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Align driver and reference DataFrames to the same distance grid.

    Both DataFrames must have a "distance" column (metres from lap start).
    We interpolate both onto the shorter lap's grid so they have the same
    number of samples and directly comparable indices.

    Returns:
        (driver_aligned, reference_aligned) — both on common grid
    """
    max_dist = min(driver_df["distance"].max(), reference_df["distance"].max())
    n_samples = min(len(driver_df), len(reference_df))

    grid = np.linspace(0, max_dist, n_samples)

    def interpolate_to_grid(df: pd.DataFrame) -> pd.DataFrame:
        out = {"distance": grid}
        for col in df.columns:
            if col == "distance":
                continue
            out[col] = np.interp(grid, df["distance"].values, df[col].values)
        return pd.DataFrame(out)

    return interpolate_to_grid(driver_df), interpolate_to_grid(reference_df)


def compute_rolling_time_delta(
    driver_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    window_m: float = 200.0,
    resolution_m: float = 5.0,
) -> np.ndarray:
    """
    Compute a rolling lap time delta at each distance point.

    The delta is estimated from cumulative speed differences. A positive
    delta means the driver is AHEAD of reference at that point; negative
    means they're behind.

    Args:
        driver_df:    Distance-aligned driver telemetry
        reference_df: Distance-aligned reference telemetry
        window_m:     Rolling window width in metres
        resolution_m: Distance resolution of the grid

    Returns:
        delta_seconds: array of shape (T,) with time delta at each point
    """
    window_samples = int(window_m / resolution_m)

    drv_speed_ms = driver_df["speed"].values / 3.6   # km/h → m/s
    ref_speed_ms = reference_df["speed"].values / 3.6

    # Avoid division by zero
    drv_speed_ms = np.where(drv_speed_ms < 1, 1, drv_speed_ms)
    ref_speed_ms = np.where(ref_speed_ms < 1, 1, ref_speed_ms)

    # Time taken to cover each distance step
    drv_dt = resolution_m / drv_speed_ms   # seconds per step
    ref_dt = resolution_m / ref_speed_ms

    # Cumulative time delta
    cumulative_delta = np.cumsum(drv_dt - ref_dt)

    # Smooth with rolling window
    if window_samples > 1:
        kernel = np.ones(window_samples) / window_samples
        cumulative_delta = np.convolve(cumulative_delta, kernel, mode="same")

    return cumulative_delta.astype(np.float32)


def detect_performance_zones(
    delta: np.ndarray,
    distance: np.ndarray,
    threshold_s: float = 0.05,
) -> list[dict]:
    """
    Identify track zones where the driver is consistently gaining or losing
    time relative to the reference. Used for sector-level coaching.

    Args:
        delta:     Rolling time delta array
        distance:  Corresponding distance array
        threshold: Minimum time gap to report as a zone

    Returns:
        List of zone dicts with start_m, end_m, delta_s, type
    """
    zones = []
    gaining = delta > threshold_s
    losing = delta < -threshold_s

    for mask, zone_type in [(gaining, "gaining"), (losing, "losing")]:
        in_zone = False
        start_idx = 0

        for i, active in enumerate(mask):
            if active and not in_zone:
                in_zone = True
                start_idx = i
            elif not active and in_zone:
                in_zone = False
                zone_length = distance[i] - distance[start_idx]
                if zone_length > 50:  # Minimum 50m zone
                    zones.append({
                        "type": zone_type,
                        "start_m": float(distance[start_idx]),
                        "end_m": float(distance[i]),
                        "length_m": float(zone_length),
                        "delta_s": float(np.mean(delta[start_idx:i])),
                    })

    return sorted(zones, key=lambda z: z["start_m"])


def lap_similarity_score(
    driver_df: pd.DataFrame,
    reference_df: pd.DataFrame,
) -> float:
    """
    Compute a 0–100 "driving similarity" score against the reference lap.
    Used for the overall performance rating shown in the dashboard.

    Methodology: cross-correlation of normalised speed traces.
    A score of 100 = identical to reference, 0 = completely different.
    """
    drv_speed = driver_df["speed"].values
    ref_speed = reference_df["speed"].values

    # Normalise both to 0-1
    def norm(x):
        r = x.max() - x.min()
        return (x - x.min()) / r if r > 0 else x

    drv_norm = norm(drv_speed)
    ref_norm = norm(ref_speed)

    # Pearson correlation
    if len(drv_norm) == 0 or len(ref_norm) == 0:
        return 0.0

    correlation = np.corrcoef(drv_norm, ref_norm)[0, 1]
    score = max(0.0, (correlation + 1) / 2 * 100)
    return round(score, 1)
