"""
labels.py — Automatic mistake label generation from telemetry delta.

No human annotation required. We derive coaching labels algorithmically
by comparing a driver's telemetry against a reference lap on the shared
distance axis. The key insight: if your speed trace diverges negatively
from the reference in a braking zone, that's a detectable late brake event.

Label categories (multi-label, a single corner can trigger multiple):
    0: late_brake   — braking later than reference, lower min speed
    1: oversteer    — high lateral G + steering correction spike
    2: understeer   — running wide (increasing lateral distance from ref line)
    3: missed_apex  — speed trace dip at corner not matching reference profile

Note: early_throttle and late_throttle removed — F1 drivers are flat-out most
of the lap so these labels fire ~0%. Re-add with better feature engineering later.
"""

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from typing import Optional


# ── Thresholds (tuned empirically on F1 telemetry) ──────────────────────────

# Speed delta threshold below which we flag a potential mistake (km/h)
SPEED_DEFICIT_KMH = 2.0

# Minimum braking zone length to consider (metres)
MIN_BRAKE_ZONE_M = 30.0

# Throttle applied while cornering (lateral G > threshold) = early throttle
EARLY_THROTTLE_LAT_G_THRESHOLD = 1.2

# Lateral G spike above reference that indicates oversteer correction
OVERSTEER_G_DELTA = 0.03

# Steering angle absolute delta indicating oversteer correction.
# Previously 15.0 degrees (too coarse, and was dead code — not wired into mask).
# Lowered to 3.0 so subtle corrections on normalised curvature data are caught.
OVERSTEER_STEER_RMS = 0.02  # degrees (absolute delta, not RMS — name kept for compatibility)

# Minimum samples for a region to be labelled (avoids noise)
MIN_REGION_SAMPLES = 5

MISTAKE_NAMES = [
    "late_brake",   # 0
    "oversteer",    # 1
    "understeer",   # 2
    "missed_apex",  # 3
]


def find_braking_zones(df: pd.DataFrame) -> list[tuple[int, int]]:
    """
    Identify braking zones as contiguous regions where brake > 0.
    Returns list of (start_idx, end_idx) index pairs.
    """
    # Smooth brake signal with a small window to merge fragmented spikes
    # (synthetic data produces scattered 1-sample zones; real data is cleaner)
    from scipy.ndimage import uniform_filter1d
    brake_raw = df["brake"].values.astype(float)
    brake = uniform_filter1d(brake_raw, size=5)  # 5-sample (~25m) smoothing

    zones = []
    in_zone = False
    start = 0

    for i, b in enumerate(brake):
        if b > 0.1 and not in_zone:  # use 0.1 threshold to handle float brake values
            in_zone = True
            start = i
        elif b <= 0.1 and in_zone:
            in_zone = False
            zone_dist = df["distance"].iloc[i] - df["distance"].iloc[start]
            if zone_dist >= MIN_BRAKE_ZONE_M:
                zones.append((start, i))

    return zones


def find_corner_apices(df: pd.DataFrame) -> list[int]:
    """
    Locate corner apices as local speed minima. These are the points
    the AI uses to evaluate if the driver hit the right entry/exit speed.
    """
    speed = df["speed"].values
    # Invert speed to find minima as peaks, require minimum prominence
    minima, _ = find_peaks(-speed, prominence=15, distance=20)
    return minima.tolist()


def generate_labels(
    driver_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    window: int = 10,
) -> np.ndarray:
    """
    Generate per-sample multi-label annotations by comparing driver telemetry
    to the reference lap. Both DataFrames must be on the same distance grid.

    Args:
        driver_df:    Distance-resampled telemetry for the driver being coached
        reference_df: Reference lap telemetry (same distance grid)
        window:       Smoothing window for delta signals

    Returns:
        labels: np.ndarray of shape (T, 6), dtype float32, values 0 or 1
    """
    T = min(len(driver_df), len(reference_df))
    labels = np.zeros((T, 4), dtype=np.float32)

    d = driver_df.iloc[:T]
    r = reference_df.iloc[:T]

    # ── Speed delta ────────────────────────────────────────────────────────
    speed_delta = d["speed"].values - r["speed"].values  # negative = slower

    # ── 0: Late brake ──────────────────────────────────────────────────────
    # Driver is slower than reference through a corner entry where reference
    # is also braking. Classic sign of late braking then scrubbing speed.
    brake_zones = find_braking_zones(r)  # use reference zones as anchors
    for start, end in brake_zones:
        zone_speed_delta = speed_delta[start:end]
        # If significantly slower through braking zone, flag late brake
        if np.mean(zone_speed_delta) < -SPEED_DEFICIT_KMH:
            labels[start:end, 0] = 1.0

    apices = find_corner_apices(r)

    # ── 1: Oversteer ──────────────────────────────────────────────────────
    # Lateral G spike above reference AND steering correction in opposite direction.
    if "lateral_g" in d.columns and "steer" in d.columns:
        lat_g_delta = d["lateral_g"].values - r["lateral_g"].values
        steer_delta = np.abs(d["steer"].values) - np.abs(r["steer"].values)

        oversteer_mask = (lat_g_delta > OVERSTEER_G_DELTA) & (steer_delta > OVERSTEER_STEER_RMS)
        labels[:, 1] = oversteer_mask.astype(np.float32)

    # ── 2: Understeer ──────────────────────────────────────────────────────
    # Driver's speed through a corner is higher than reference BUT they carry
    # less lateral G, indicating they're running wide rather than rotating.
    if "lateral_g" in d.columns:
        # line ~130 in labels.py
        understeer_mask = (
            (speed_delta > 2) &                                      # was 5
            (d["lateral_g"].values < r["lateral_g"].values - 0.15) & # was 0.3 — too large
            (r["lateral_g"].values > 0.8)
        )
        labels[:, 2] = understeer_mask.astype(np.float32)

    # ── 3: Missed apex ─────────────────────────────────────────────────────
    # The driver's speed dip at the apex is significantly higher than reference,
    # meaning they carried too much or too little speed to hit the apex.
    for apex_idx in apices:
        apex_window = slice(max(0, apex_idx - 5), min(T, apex_idx + 5))
        drv_apex_speed = d["speed"].values[apex_window].min()
        ref_apex_speed = r["speed"].values[apex_window].min()
        if abs(drv_apex_speed - ref_apex_speed) > SPEED_DEFICIT_KMH * 0.4:
            labels[apex_window, 3] = 1.0

    # ── Smooth labels to avoid single-sample flickers ─────────────────────
    for i in range(4):
        kernel = np.ones(window) / window
        labels[:, i] = np.convolve(labels[:, i], kernel, mode="same")
        labels[:, i] = (labels[:, i] > 0.3).astype(np.float32)

    return labels


def labels_to_coaching_cues(
    labels: np.ndarray,
    distance: float,
    speed: float,
    min_gap_seconds: float = 3.0,
    last_cue_time: Optional[float] = None,
    current_time: float = 0.0,
) -> Optional[dict]:
    """
    Convert a label vector at a specific distance into a human-readable
    coaching cue. Returns None if no cue, or if too soon after last cue.

    Cues are rate-limited to avoid overwhelming the driver.
    """
    if last_cue_time is not None and (current_time - last_cue_time) < min_gap_seconds:
        return None

    # Priority order — only return the highest-priority active label
    cue_templates = [
        (0, "late_brake",  "⚠️  Brake earlier here — you're losing {delta:.0f} km/h through the zone"),
        (1, "oversteer",   "🔴 Oversteer — ease steering inputs on entry"),
        (2, "understeer",  "🟡 Understeer — reduce entry speed, hit the apex tighter"),
        (3, "missed_apex", "📍 Missed apex — adjust your reference point"),
    ]

    for idx, name, template in cue_templates:
        if idx < labels.shape[0] and labels[idx] > 0.5:
            return {
                "type": name,
                "message": template.format(delta=abs(speed - 250)),  # placeholder delta
                "distance": distance,
                "severity": "high" if idx in (0, 1) else "medium",
                "timestamp": current_time,
            }

    return None


def compute_sector_report(
    driver_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    labels: np.ndarray,
    n_sectors: int = 3,
) -> list[dict]:
    """
    Summarise performance into sector-level coaching report.
    Returns a list of dicts, one per sector.
    """
    T = min(len(driver_df), len(reference_df))
    sector_size = T // n_sectors
    report = []

    for s in range(n_sectors):
        start = s * sector_size
        end = (s + 1) * sector_size if s < n_sectors - 1 else T

        drv_speed = driver_df["speed"].values[start:end]
        ref_speed = reference_df["speed"].values[start:end]
        sector_labels = labels[start:end]

        sector_dist_start = driver_df["distance"].iloc[start]
        sector_dist_end = driver_df["distance"].iloc[end - 1]

        # Estimate time delta using average speed difference
        speed_delta_kmh = np.mean(drv_speed - ref_speed)
        sector_length_km = (sector_dist_end - sector_dist_start) / 1000.0
        avg_ref_speed = np.mean(ref_speed)
        if avg_ref_speed > 0:
            time_delta_s = -speed_delta_kmh / avg_ref_speed * sector_length_km * 3600
        else:
            time_delta_s = 0.0

        mistake_counts = {MISTAKE_NAMES[i]: int(sector_labels[:, i].sum()) for i in range(4)}
        primary_mistake = max(mistake_counts, key=mistake_counts.get)
        total_mistakes = sum(mistake_counts.values())

        report.append({
            "sector": s + 1,
            "time_delta_s": round(time_delta_s, 3),
            "avg_speed_delta_kmh": round(speed_delta_kmh, 1),
            "mistakes": mistake_counts,
            "primary_issue": primary_mistake if total_mistakes > 5 else None,
            "dist_start_m": round(sector_dist_start, 0),
            "dist_end_m": round(sector_dist_end, 0),
        })

    return report


if __name__ == "__main__":
    # Smoke test with synthetic data
    n = 500
    dist = np.linspace(0, 2500, n)

    ref = pd.DataFrame({
        "distance": dist,
        "speed": 200 + 80 * np.sin(dist / 400) + np.random.randn(n) * 2,
        "throttle": np.clip(50 + 50 * np.sin(dist / 400 + 1), 0, 100),
        "brake": ((np.sin(dist / 400) < -0.5)).astype(float),
        "gear": np.clip(4 + 3 * np.sin(dist / 400), 1, 8),
        "steer": 20 * np.sin(dist / 400),
        "lateral_g": np.abs(1.5 * np.sin(dist / 400)),
        "x": np.cumsum(np.cos(dist / 400)),
        "y": np.cumsum(np.sin(dist / 400)),
    })

    # Driver is slightly slower in braking zones
    drv = ref.copy()
    drv["speed"] = ref["speed"] - 10 * (ref["brake"] > 0).astype(float)

    labels = generate_labels(drv, ref)
    print(f"Labels shape: {labels.shape}")
    print(f"Mistake counts: { {MISTAKE_NAMES[i]: int(labels[:,i].sum()) for i in range(4)} }")

    report = compute_sector_report(drv, ref, labels)
    for s in report:
        print(f"\nSector {s['sector']}: Δt={s['time_delta_s']:+.3f}s | Primary: {s['primary_issue']}")