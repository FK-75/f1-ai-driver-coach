"""
calibrate_thresholds.py — Threshold sensitivity analysis for label generation.

Sweeps SPEED_DEFICIT_KMH and OVERSTEER_STEER_RMS across a range of values
and plots how label frequency changes with each threshold. Helps you tune
thresholds empirically rather than guessing.

Usage:
    python scripts/calibrate_thresholds.py

Output:
    - Console table of label frequencies per threshold value
    - data/threshold_calibration.png — sensitivity plot

The goal: find thresholds where label frequency is stable (flat region of the
curve). A threshold in the steep part of the curve is sensitive to noise;
one in the flat part is robust.
"""

import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data"
PKL_PATH = DATA_DIR / "training_laps.pkl"


def load_records():
    if not PKL_PATH.exists():
        raise FileNotFoundError(f"No training data at {PKL_PATH}. Run download_data.py first.")
    with open(PKL_PATH, "rb") as f:
        return pickle.load(f)


def compute_label_freqs(records, speed_deficit, oversteer_steer, sample_n=20):
    """
    Run label generation on a sample of laps with given thresholds.
    Returns dict of {mistake_name: mean_frequency_pct}.
    """
    import importlib
    import backend.pipeline.labels as labels_mod

    # Temporarily patch thresholds
    orig_speed = labels_mod.SPEED_DEFICIT_KMH
    orig_steer = labels_mod.OVERSTEER_STEER_RMS
    labels_mod.SPEED_DEFICIT_KMH = speed_deficit
    labels_mod.OVERSTEER_STEER_RMS = oversteer_steer

    try:
        # Sample laps for speed
        sample = records[:sample_n] if len(records) >= sample_n else records
        freqs = defaultdict(list)

        # Pre-build list of DFs so we can pick a different lap as reference
        dfs = [rec["df"] for rec in sample]

        for i, rec in enumerate(sample):
            drv_df = rec["df"]
            # Use the next lap (wraps around) as reference — never self-compare,
            # since identical signals produce zero deltas and no labels fire.
            ref_df = dfs[(i + 1) % len(dfs)]

            # Align to the shorter of the two laps
            T = min(len(drv_df), len(ref_df))
            drv_df = drv_df.iloc[:T].reset_index(drop=True)
            ref_df = ref_df.iloc[:T].reset_index(drop=True)

            try:
                lab = labels_mod.generate_labels(drv_df, ref_df)
                for j, name in enumerate(labels_mod.MISTAKE_NAMES):
                    freq = float(lab[:, j].mean()) * 100
                    freqs[name].append(freq)
            except Exception as e:
                print(f"    [warn] lap {i} skipped: {e}")
                continue

        return {name: np.mean(vals) if vals else 0.0 for name, vals in freqs.items()}

    finally:
        labels_mod.SPEED_DEFICIT_KMH = orig_speed
        labels_mod.OVERSTEER_STEER_RMS = orig_steer


def run_sweep():
    print("Loading training laps...")
    records = load_records()
    print(f"  Loaded {len(records)} laps. Using first 20 for calibration.\n")

    from backend.pipeline.labels import MISTAKE_NAMES

    # ── Sweep 1: SPEED_DEFICIT_KMH ───────────────────────────────────────────
    speed_thresholds = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0]
    print("=" * 70)
    print("SPEED_DEFICIT_KMH sweep (OVERSTEER_STEER_RMS fixed at 0.05)")
    print("Lower threshold = more late_brake / missed_apex labels fired")
    print("=" * 70)

    speed_results = []
    for thresh in speed_thresholds:
        freqs = compute_label_freqs(records, speed_deficit=thresh, oversteer_steer=0.05)
        speed_results.append((thresh, freqs))
        lb = freqs.get("late_brake", 0)
        ma = freqs.get("missed_apex", 0)
        print(f"  speed_deficit={thresh:5.1f} km/h  |  late_brake={lb:5.1f}%  missed_apex={ma:5.1f}%")

    # ── Sweep 2: OVERSTEER_STEER_RMS ─────────────────────────────────────────
    steer_thresholds = [0.01, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50]
    print()
    print("=" * 70)
    print("OVERSTEER_STEER_RMS sweep (SPEED_DEFICIT_KMH fixed at 2.0)")
    print("Lower threshold = more oversteer labels fired")
    print("=" * 70)

    steer_results = []
    for thresh in steer_thresholds:
        freqs = compute_label_freqs(records, speed_deficit=2.0, oversteer_steer=thresh)
        steer_results.append((thresh, freqs))
        os_freq = freqs.get("oversteer", 0)
        us_freq = freqs.get("understeer", 0)
        print(f"  steer_rms={thresh:5.1f}°  |  oversteer={os_freq:5.1f}%  understeer={us_freq:5.1f}%")

    # ── Recommendation ────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)
    print()
    print("Target label frequencies for a well-balanced dataset:")
    print("  late_brake:     5–20%   (should fire at every significant braking zone)")
    print("  missed_apex:    5–15%   (should fire at corners with speed mismatch)")
    print("  oversteer:      2–10%   (should fire at high-speed corners)")
    print("  understeer:     2–10%   (should fire at slow entry corners)")
    print()

    # Find best speed threshold (closest to 10% late_brake)
    best_speed = min(speed_results, key=lambda x: abs(x[1].get("late_brake", 0) - 10.0))
    print(f"  Best SPEED_DEFICIT_KMH:    {best_speed[0]} km/h  "
          f"(late_brake={best_speed[1].get('late_brake', 0):.1f}%)")

    best_steer = min(steer_results, key=lambda x: abs(x[1].get("oversteer", 0) - 5.0))
    print(f"  Best OVERSTEER_STEER_RMS:  {best_steer[0]}°  "
          f"(oversteer={best_steer[1].get('oversteer', 0):.1f}%)")

    print()
    print("To apply: update constants at the top of backend/pipeline/labels.py")

    # ── Plot if matplotlib available ──────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor("#080b10")
        for ax in axes:
            ax.set_facecolor("#0d1117")
            ax.tick_params(colors="#888")
            ax.spines[:].set_color("#333")

        # Speed deficit plot
        ax = axes[0]
        ax.set_title("SPEED_DEFICIT_KMH sensitivity", color="#e8eaed", fontsize=11)
        ax.set_xlabel("Threshold (km/h)", color="#888")
        ax.set_ylabel("Label frequency (%)", color="#888")
        colors = {"late_brake": "#ff6b35", "missed_apex": "#ab47bc",
                  "early_throttle": "#ffa726", "late_throttle": "#66bb6a"}
        for name, color in colors.items():
            vals = [freq_dict.get(name, 0) for _, freq_dict in speed_results]
            ax.plot(speed_thresholds, vals, marker="o", label=name, color=color, linewidth=2)
        ax.axvline(x=best_speed[0], color="#ffd700", linestyle="--", alpha=0.6, label=f"recommended ({best_speed[0]})")
        ax.legend(fontsize=8, facecolor="#0d1117", labelcolor="#888")
        ax.grid(alpha=0.1, color="#333")

        # Steer RMS plot
        ax = axes[1]
        ax.set_title("OVERSTEER_STEER_RMS sensitivity", color="#e8eaed", fontsize=11)
        ax.set_xlabel("Threshold (degrees)", color="#888")
        ax.set_ylabel("Label frequency (%)", color="#888")
        steer_colors = {"oversteer": "#ef5350", "understeer": "#ffee58"}
        for name, color in steer_colors.items():
            vals = [freq_dict.get(name, 0) for _, freq_dict in steer_results]
            ax.plot(steer_thresholds, vals, marker="o", label=name, color=color, linewidth=2)
        ax.axvline(x=best_steer[0], color="#ffd700", linestyle="--", alpha=0.6, label=f"recommended ({best_steer[0]})")
        ax.legend(fontsize=8, facecolor="#0d1117", labelcolor="#888")
        ax.grid(alpha=0.1, color="#333")

        plt.tight_layout()
        out_path = DATA_DIR / "threshold_calibration.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#080b10")
        print(f"\n📊 Plot saved to {out_path}")

    except ImportError:
        print("\n  (Install matplotlib to generate plot: pip install matplotlib)")


if __name__ == "__main__":
    run_sweep()