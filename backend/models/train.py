"""
train.py — Training loop for the TelemetryTCN model.

Training strategy:
    - Load FastF1 laps for multiple drivers/events
    - Use the fastest lap from each event as the reference
    - Generate labels by comparing each driver lap against the reference
    - Train with sliding windows (512 samples) to handle variable-length laps
    - Export to ONNX after training

Usage:
    python backend/models/train.py
    or via:
    python scripts/train.py
"""

import os
import sys
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional
from tqdm import tqdm

# Add project root to path
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from backend.models.tcn import TelemetryTCN, TelemetryLoss
from backend.pipeline.labels import generate_labels, compute_sector_report
from backend.pipeline.alignment import align_to_common_grid, compute_rolling_time_delta

# ── Device selection ──────────────────────────────────────────────────────────

def get_device() -> torch.device:
    """
    Pick the best available device in priority order: CUDA → MPS → CPU.
    CUDA covers the RTX 4080 Super (Windows/Linux via CUDA, or eGPU).
    MPS covers Apple Silicon on the MacBook.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
        print(f"🖥️  Using CUDA: {name}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🍎 Using MPS (Apple Silicon GPU)")
    else:
        device = torch.device("cpu")
        print("💻 Using CPU (no GPU detected)")
    return device

# ── Config ───────────────────────────────────────────────────────────────────

CONFIG = {
    # ── Window config ─────────────────────────────────────────────────────
    # window_size: how many distance samples per training window (512 = 2,560m)
    # stride: step between windows — smaller = more windows per lap.
    #   At stride=20, a 1,160-sample Silverstone lap gives ~32 windows vs 10 at stride=64.
    #   Monaco (650 samples) gives ~7 windows vs 2 at stride=64.
    #   Total windows for 48 laps goes from ~120 → ~900+.
    "window_size": 512,
    "stride": 20,

    # ── Model capacity ────────────────────────────────────────────────────
    # Scaled up now that we have 48 laps (was 64ch/4blocks for 12 laps).
    # 128 channels + 6 blocks gives ~4× more parameters while keeping
    # inference well under 5ms on CPU.
    "hidden_channels": 128,
    "kernel_size": 3,
    "n_blocks": 6,
    "dropout": 0.2,              # slightly higher dropout to match larger model

    # ── Training ──────────────────────────────────────────────────────────
    "batch_size": 64,            # larger batch fits in memory with more windows
    "learning_rate": 2e-4,       # slightly lower LR for larger model stability
    "n_epochs": 100,
    "val_split": 0.15,
    "seed": 42,
}

DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "data" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ── Dataset ──────────────────────────────────────────────────────────────────

class TelemetryDataset(Dataset):
    """
    Sliding-window dataset over multiple F1 laps.

    Each item is a dict with:
        "features":  (7, window_size) float32 — telemetry channels
        "delta":     (window_size,)   float32 — time delta vs reference
        "labels":    (6, window_size) float32 — mistake labels
    """

    def __init__(
        self,
        records: list[dict],
        reference_record: dict,
        window_size: int = 512,
        stride: int = 64,
    ):
        self.window_size = window_size
        self.stride = stride
        self.samples = []

        ref_df = reference_record["df"]

        for record in records:
            drv_df = record["df"]

            # Align to common distance grid
            try:
                drv_aligned, ref_aligned = align_to_common_grid(drv_df, ref_df)
            except Exception:
                continue

            T = min(len(drv_aligned), len(ref_aligned))
            if T < window_size:
                continue

            # Generate labels and time delta
            labels = generate_labels(drv_aligned, ref_aligned)  # (T, 6)
            delta = compute_rolling_time_delta(drv_aligned, ref_aligned)  # (T,)

            # Features (T, 7) — already normalised
            from backend.pipeline.telemetry import build_feature_matrix
            drv_aligned_full = drv_aligned.copy()
            if "lateral_g" not in drv_aligned_full.columns:
                drv_aligned_full["lateral_g"] = 0.0
            features = build_feature_matrix(drv_aligned_full)  # (T, 7)

            # Sliding windows
            for start in range(0, T - window_size, stride):
                end = start + window_size
                self.samples.append({
                    "features": features[start:end].T.astype(np.float32),   # (7, W)
                    "delta": delta[start:end].astype(np.float32),            # (W,)
                    "labels": labels[start:end].T.astype(np.float32),        # (6, W)
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return (
            torch.from_numpy(s["features"]),
            torch.from_numpy(s["delta"]),
            torch.from_numpy(s["labels"]),
        )


# ── Training loop ─────────────────────────────────────────────────────────────

def train(records: list[dict], config: dict = CONFIG):
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    device = get_device()

    # Use the fastest lap overall as reference
    reference = min(records, key=lambda r: r["lap_time_s"])
    print(f"📍 Reference lap: {reference['driver']} @ {reference['gp']} {reference['year']} — {reference['lap_time_s']:.3f}s")

    # Exclude reference from training data to avoid trivial learning
    train_records = [r for r in records if r is not reference]

    # Split train/val
    n_val = max(1, int(len(train_records) * config["val_split"]))
    val_records = train_records[:n_val]
    train_records = train_records[n_val:]

    print(f"\n📊 Dataset: {len(train_records)} train laps, {len(val_records)} val laps")

    train_ds = TelemetryDataset(train_records, reference, config["window_size"], config["stride"])
    val_ds = TelemetryDataset(val_records, reference, config["window_size"], config["window_size"])

    print(f"   Train windows: {len(train_ds)}, Val windows: {len(val_ds)}")

    # num_workers > 0 can cause issues with MPS/CUDA on some setups; keep at 0 for safety
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=0, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))

    # Model — move to device
    model = TelemetryTCN(
        hidden_channels=config["hidden_channels"],
        kernel_size=config["kernel_size"],
        n_blocks=config["n_blocks"],
        dropout=config["dropout"],
    ).to(device)

    criterion = TelemetryLoss(delta_weight=1.0, mistake_weight=2.0)
    optimiser = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=config["n_epochs"])

    best_val_loss = float("inf")
    best_model_path = MODEL_DIR / "tcn_best.pt"

    print(f"\n🚀 Training for {config['n_epochs']} epochs...\n")

    epoch_bar = tqdm(range(1, config["n_epochs"] + 1), desc="Training", unit="epoch")

    for epoch in epoch_bar:
        # ── Train ────────────────────────────────────────────────────────
        model.train()
        train_losses = []

        batch_bar = tqdm(train_loader, desc=f"Epoch {epoch:02d} train", leave=False)
        for features, delta, labels in batch_bar:
            features = features.to(device, non_blocking=True)
            delta    = delta.to(device, non_blocking=True)
            labels   = labels.to(device, non_blocking=True)

            optimiser.zero_grad()
            pred_delta, pred_mistakes = model(features)
            loss, _ = criterion(pred_delta, pred_mistakes, delta, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            train_losses.append(loss.item())
            batch_bar.set_postfix(loss=f"{loss.item():.4f}")

        # ── Validate ─────────────────────────────────────────────────────
        model.eval()
        val_losses = []
        val_details = {"total": [], "delta": [], "mistake": []}

        with torch.no_grad():
            for features, delta, labels in tqdm(val_loader, desc=f"Epoch {epoch:02d} val", leave=False):
                features = features.to(device, non_blocking=True)
                delta    = delta.to(device, non_blocking=True)
                labels   = labels.to(device, non_blocking=True)

                pred_delta, pred_mistakes = model(features)
                loss, detail = criterion(pred_delta, pred_mistakes, delta, labels)
                val_losses.append(loss.item())
                for k, v in detail.items():
                    val_details[k].append(v)

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        val_delta = np.mean(val_details["delta"])
        val_mistake = np.mean(val_details["mistake"])

        scheduler.step()

        epoch_bar.set_postfix(
            train=f"{train_loss:.4f}",
            val=f"{val_loss:.4f}",
            lr=f"{scheduler.get_last_lr()[0]:.1e}",
        )

        tqdm.write(
            f"Epoch {epoch:02d}/{config['n_epochs']} | "
            f"Train: {train_loss:.4f} | "
            f"Val: {val_loss:.4f} (Δt={val_delta:.4f}, cls={val_mistake:.4f}) | "
            f"LR: {scheduler.get_last_lr()[0]:.2e}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": config,
                "val_loss": val_loss,
                "reference_driver": reference["driver"],
                "reference_gp": reference["gp"],
                "reference_year": reference["year"],
            }, best_model_path)
            tqdm.write(f"  💾 Saved best model (val_loss={val_loss:.4f})")

    print(f"\n✅ Training complete. Best val loss: {best_val_loss:.4f}")
    print(f"   Model saved to: {best_model_path}")

    # Return model on CPU so ONNX export doesn't need a device-aware path
    return model.cpu(), best_model_path


def load_model(path: Optional[Path] = None) -> TelemetryTCN:
    """Load a trained model from checkpoint."""
    if path is None:
        path = MODEL_DIR / "tcn_best.pt"

    checkpoint = torch.load(path, map_location="cpu")
    cfg = checkpoint.get("config", CONFIG)

    model = TelemetryTCN(
        hidden_channels=cfg["hidden_channels"],
        kernel_size=cfg["kernel_size"],
        n_blocks=cfg["n_blocks"],
        dropout=0.0,  # No dropout at inference
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


if __name__ == "__main__":
    # Load pre-fetched training data
    data_path = DATA_DIR / "training_laps.pkl"
    if not data_path.exists():
        print("❌ No training data found. Run scripts/download_data.py first.")
        sys.exit(1)

    with open(data_path, "rb") as f:
        records = pickle.load(f)

    print(f"Loaded {len(records)} laps from {data_path}")
    model, model_path = train(records)

    # Export to ONNX
    from backend.models.export import export_to_onnx
    export_to_onnx(model, MODEL_DIR / "tcn.onnx")
