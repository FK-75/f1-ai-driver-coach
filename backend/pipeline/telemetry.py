"""
telemetry.py — FastF1 data fetching and preprocessing pipeline.

Fetches real F1 telemetry (speed, throttle, brake, gear, steering, lateral G)
and resamples everything onto a common distance axis so laps can be compared
across drivers and sessions without timestamp misalignment.
"""

import os
import numpy as np
import pandas as pd
import fastf1
from pathlib import Path
from typing import Optional

# FastF1 caches data locally so we don't re-download
CACHE_DIR = Path(__file__).parent.parent.parent / "data" / "fastf1_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
fastf1.Cache.enable_cache(str(CACHE_DIR))

# Telemetry channels we care about
CHANNELS = ["Speed", "Throttle", "Brake", "nGear", "X", "Y"]
# Note: SteeringAngle is NOT provided by the F1 timing API (FastF1 car_data).
# Steering is derived from X/Y curvature via compute_lateral_g instead.

# Distance resolution: resample every N metres along the lap
DISTANCE_RESOLUTION_M = 5.0


def fetch_session(year: int, gp: str, session_type: str = "Q") -> fastf1.core.Session:
    """Load a FastF1 session (uses local cache after first download)."""
    session = fastf1.get_session(year, gp, session_type)
    session.load(telemetry=True, laps=True, weather=False, messages=False)
    return session


def get_fastest_lap(session: fastf1.core.Session, driver: str) -> fastf1.core.Lap:
    """Return the fastest timed lap for a driver in a session."""
    laps = session.laps.pick_driver(driver).pick_quicklaps()
    return laps.pick_fastest()


def lap_to_distance_frame(lap: fastf1.core.Lap) -> pd.DataFrame:
    """
    Extract telemetry for a lap and resample onto a uniform distance axis.

    FastF1 telemetry is timestamped at irregular intervals (~10–100ms).
    We convert to distance (metres from lap start) and resample at fixed
    DISTANCE_RESOLUTION_M intervals so two laps can be directly compared
    channel-by-channel without DTW alignment artifacts from timestamp jitter.

    Returns a DataFrame with columns:
        distance, speed, throttle, brake, gear, steer, lateral_g, x, y
    and one row per distance sample.
    """
    tel = lap.get_car_data().add_distance()

    # Merge with positional data for track map
    pos = lap.get_pos_data()
    # Interpolate position to car data timestamps
    tel = tel.merge_channels(pos)

    # Rename for cleaner downstream use
    df = pd.DataFrame({
        "distance":  tel["Distance"].values,
        "speed":     tel["Speed"].values,         # km/h
        "throttle":  tel["Throttle"].values,      # 0–100
        "brake":     tel["Brake"].astype(float).values,   # 0 or 1 (binary)
        "gear":      tel["nGear"].values.astype(float),
        "x":         tel["X"].values,             # track coordinates
        "y":         tel["Y"].values,
        # SteeringAngle is not available from the F1 timing API.
        # A surrogate steer signal is computed from X/Y curvature below.
    })

    # Drop NaNs that sometimes appear at lap boundaries
    df = df.dropna().reset_index(drop=True)

    # Deduplicate distance (can get duplicate values at start/end)
    df = df.drop_duplicates(subset="distance").sort_values("distance").reset_index(drop=True)

    # Resample onto uniform grid
    max_dist = df["distance"].max()
    grid = np.arange(0, max_dist, DISTANCE_RESOLUTION_M)

    resampled = {}
    for col in df.columns:
        if col == "distance":
            resampled[col] = grid
        else:
            resampled[col] = np.interp(grid, df["distance"].values, df[col].values)

    return pd.DataFrame(resampled)


def compute_lateral_g(df: pd.DataFrame, dt_seconds: float = 0.05) -> pd.Series:
    """
    Estimate lateral G from X/Y track coordinates.

    We don't always get lateral G directly from FastF1, so we derive it from
    the curvature of the path and instantaneous speed. This is an approximation
    but close enough for coaching purposes.
    """
    # Speed in m/s
    speed_ms = df["speed"].values / 3.6

    # Get the actual distance spacing for gradient calculation
    dist = df["distance"].values
    
    # Derivative of heading angle = curvature (with proper spacing)
    dx = np.gradient(df["x"].values, dist)
    dy = np.gradient(df["y"].values, dist)
    ddx = np.gradient(dx, dist)
    ddy = np.gradient(dy, dist)

    # Curvature κ = (x'y'' - y'x'') / (x'^2 + y'^2)^(3/2)
    denom = (dx**2 + dy**2) ** 1.5
    denom = np.where(denom < 1e-6, 1e-6, denom)  # avoid div/0
    curvature = (dx * ddy - dy * ddx) / denom

    # Lateral acceleration = v² × κ, convert to G
    lat_accel_ms2 = speed_ms**2 * np.abs(curvature)
    lat_g = lat_accel_ms2 / 9.81

    return pd.Series(lat_g, name="lateral_g")


def compute_steer(df: pd.DataFrame) -> pd.Series:
    """
    Derive a signed steering surrogate from X/Y curvature.

    Since SteeringAngle is not transmitted by the F1 timing API, we approximate
    it from the signed curvature of the X/Y path. Positive = right, negative = left.
    The value is dimensionless curvature scaled to a ±1 range (clipped).
    This is sufficient for coaching comparisons — absolute degree values aren't needed.
    """
    # Get the actual distance spacing for gradient calculation
    dist = df["distance"].values
    
    dx = np.gradient(df["x"].values, dist)
    dy = np.gradient(df["y"].values, dist)
    ddx = np.gradient(dx, dist)
    ddy = np.gradient(dy, dist)

    denom = (dx**2 + dy**2) ** 1.5
    denom = np.where(denom < 1e-6, 1e-6, denom)
    # Signed curvature: positive = curving right, negative = curving left
    curvature = (dx * ddy - dy * ddx) / denom

    # Scale: typical F1 curvature peaks ~0.02–0.05 m⁻¹ in tight corners
    steer = np.clip(curvature / 0.02, -1, 1)
    return pd.Series(steer, name="steer")


def build_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """
    Normalise channels and stack into (T, C) feature matrix for the TCN.

    Channels (in order):
        0: speed       — normalised 0–1 (0 = 0 km/h, 1 = 350 km/h)
        1: throttle    — already 0–100, divide by 100
        2: brake       — binary 0/1
        3: gear        — divide by 8 (max gear)
        4: steer       — signed curvature surrogate, clipped ±1
                         (SteeringAngle not available from F1 API; derived from X/Y path)
        5: lateral_g   — clip at 5G, divide by 5
        6: dist_norm   — distance / lap_length (0→1)
    """
    T = len(df)
    features = np.zeros((T, 7), dtype=np.float32)

    features[:, 0] = np.clip(df["speed"].values / 350.0, 0, 1)
    features[:, 1] = np.clip(df["throttle"].values / 100.0, 0, 1)
    features[:, 2] = df["brake"].values.astype(np.float32)
    features[:, 3] = np.clip(df["gear"].values / 8.0, 0, 1)
    features[:, 4] = df["steer"].values if "steer" in df.columns else np.zeros(T, dtype=np.float32)
    features[:, 5] = np.clip(df["lateral_g"].values / 5.0, 0, 1) if "lateral_g" in df.columns else np.zeros(T, dtype=np.float32)
    features[:, 6] = df["distance"].values / df["distance"].max()

    return features


def fetch_training_data(
    events: Optional[list] = None,
    drivers: Optional[list] = None,
    save_dir: Optional[Path] = None,
) -> list[dict]:
    """
    Fetch multiple laps for training. Returns list of dicts with:
        { "driver", "gp", "year", "session", "features": np.ndarray (T,7),
          "lap_time_s": float, "df": pd.DataFrame }

    Default: 2023 Silverstone + Monza qualifying, Verstappen + Hamilton.
    """
    if events is None:
        events = [
            (2023, "British Grand Prix", "Q"),
            (2023, "Italian Grand Prix", "Q"),
            (2022, "British Grand Prix", "Q"),
        ]
    if drivers is None:
        drivers = ["VER", "HAM", "LEC", "NOR"]

    records = []
    for year, gp, ses_type in events:
        print(f"\n📡 Loading {year} {gp} {ses_type}...")
        try:
            session = fetch_session(year, gp, ses_type)
        except Exception as e:
            print(f"  ⚠️  Could not load session: {e}")
            continue

        for driver in drivers:
            try:
                lap = get_fastest_lap(session, driver)
                df = lap_to_distance_frame(lap)
                df["lateral_g"] = compute_lateral_g(df)
                df["steer"] = compute_steer(df)
                features = build_feature_matrix(df)
                lap_time_s = lap["LapTime"].total_seconds()

                record = {
                    "driver": driver,
                    "gp": gp,
                    "year": year,
                    "session": ses_type,
                    "features": features,
                    "lap_time_s": lap_time_s,
                    "df": df,
                }
                records.append(record)
                print(f"  ✅ {driver}: {lap_time_s:.3f}s, {len(df)} samples")

            except Exception as e:
                print(f"  ⚠️  {driver}: {e}")

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        import pickle
        with open(save_dir / "training_laps.pkl", "wb") as f:
            pickle.dump(records, f)
        print(f"\n💾 Saved {len(records)} laps to {save_dir}/training_laps.pkl")

    return records


def load_reference_lap(gp: str = "British Grand Prix", year: int = 2023, driver: str = "VER") -> dict:
    """
    Load the reference lap used for live coaching comparisons.
    This is the lap the AI compares the user's (or demo) driving against.
    """
    print(f"🏁 Loading reference lap: {year} {gp} — {driver}")
    session = fetch_session(year, gp, "Q")
    lap = get_fastest_lap(session, driver)
    df = lap_to_distance_frame(lap)
    df["lateral_g"] = compute_lateral_g(df)
    df["steer"] = compute_steer(df)
    features = build_feature_matrix(df)

    return {
        "driver": driver,
        "gp": gp,
        "year": year,
        "features": features,
        "lap_time_s": lap["LapTime"].total_seconds(),
        "df": df,
    }


if __name__ == "__main__":
    # Quick test
    ref = load_reference_lap()
    print(f"\nReference lap: {ref['lap_time_s']:.3f}s, {len(ref['df'])} distance samples")
    print(f"Feature matrix shape: {ref['features'].shape}")
    print(f"Columns: {list(ref['df'].columns)}")