"""
train.py (scripts entrypoint) — Trains the TCN model and exports to ONNX.

Usage:
    python scripts/train.py [--epochs N] [--fast]

Options:
    --epochs N   Number of training epochs (default: 100)
    --fast       Quick training with synthetic data (no FastF1 required)
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data"
MODEL_DIR = DATA_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def make_synthetic_records(n_laps: int = 20) -> list[dict]:
    """
    Generate synthetic training records for fast/offline training.
    Mimics FastF1 data structure. Use when real F1 data isn't available.
    """
    import pandas as pd

    print("Generating synthetic training data...")
    np.random.seed(42)
    records = []

    lap_length_m = 5891.0
    n_samples = int(lap_length_m / 5.0)
    distance = np.linspace(0, lap_length_m, n_samples)

    for i in range(n_laps):
        noise = np.random.randn(n_samples)
        skill = np.random.uniform(0.88, 1.0)

        speed = np.clip(
            220 + 100 * (0.4 * np.sin(distance / 400) + 0.3 * np.sin(distance / 180 + 1.2)) * skill
            + noise * 4,
            80,
            330,
        )
        speed_grad = np.gradient(speed)
        throttle = np.clip(50 + 50 * speed_grad / (np.abs(speed_grad).max() + 1e-6) + noise * 5, 0, 100)
        brake = (speed_grad < -3).astype(float)
        gear = np.clip((speed / 60).astype(int), 1, 8).astype(float)
        steer = 30 * np.sin(distance / 300 + 0.8) + noise * 5

        theta = distance / lap_length_m * 2 * np.pi
        r = 800 + 300 * np.sin(3 * theta + 0.5)
        x = r * np.cos(theta)
        y = r * np.sin(theta)

        lat_g = np.abs(2.5 * np.sin(distance / 300 + 0.8)) * np.clip(1 - speed / 330, 0, 1) * 3

        df = pd.DataFrame({
            "distance": distance,
            "speed": speed,
            "throttle": throttle,
            "brake": brake,
            "gear": gear,
            "steer": steer,
            "lateral_g": lat_g,
            "x": x,
            "y": y,
        })

        dt = 5.0 / np.clip(speed / 3.6, 1, 100)
        lap_time_s = dt.sum()

        drivers = ["VER", "HAM", "LEC", "NOR", "PER", "SAI", "ALO", "RUS"]
        gps = ["British Grand Prix", "Italian Grand Prix", "Belgian Grand Prix"]

        records.append({
            "driver": drivers[i % len(drivers)],
            "gp": gps[i % len(gps)],
            "year": 2023,
            "session": "Q",
            "features": None,
            "lap_time_s": float(lap_time_s),
            "df": df,
        })

    print(f"  Saved {len(records)} synthetic laps in memory")
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--fast", action="store_true", help="Use synthetic data, skip FastF1")
    args = parser.parse_args()

    print("=" * 60)
    print("  F1 AI Driver Coach — Training")
    print("=" * 60)

    data_path = DATA_DIR / "training_laps.pkl"

    if args.fast:
        print("\nFast mode: using synthetic data")
        records = make_synthetic_records(n_laps=24)
    elif data_path.exists():
        print(f"\nLoading training data from {data_path}")
        with open(data_path, "rb") as f:
            records = pickle.load(f)
        print(f"  Loaded {len(records)} laps")
    else:
        print(f"\nNo training data found at {data_path}")
        print("  Falling back to synthetic data (run download_data.py for real F1 data)")
        records = make_synthetic_records(n_laps=24)

    if not records:
        print("No training records available.")
        sys.exit(1)

    from backend.models.export import ONNXInferenceEngine, export_to_onnx
    from backend.models.train import CONFIG, train

    config = CONFIG.copy()
    config["n_epochs"] = args.epochs

    model, model_path = train(records, config)

    print("\nExporting to ONNX...")
    onnx_path = MODEL_DIR / "tcn.onnx"
    export_to_onnx(model, onnx_path)

    engine = ONNXInferenceEngine(onnx_path)
    bench = engine.benchmark(100)
    print("\nInference benchmark:")
    print(f"   Mean: {bench['mean_ms']:.2f}ms | P95: {bench['p95_ms']:.2f}ms | P99: {bench['p99_ms']:.2f}ms")

    print("\n" + "=" * 60)
    print("  Training complete")
    print(f"  PyTorch model: {model_path}")
    print(f"  ONNX model:    {onnx_path}")
    print("\n  Next: python backend/api/main.py")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
