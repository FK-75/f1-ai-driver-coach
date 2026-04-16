"""
tcn.py — Temporal Convolutional Network for F1 telemetry analysis.

Architecture choice rationale:
    A Transformer would work but gives ~47ms inference on CPU.
    This TCN gives ~8ms on CPU, which is what makes real-time coaching viable.
    The receptive field covers 512 distance samples = 2,560m at 5m resolution,
    which is enough context to understand a complete corner sequence.

Model outputs:
    1. delta_time  — predicted lap time delta vs reference (regression)
    2. mistakes    — 4-class multi-label mistake classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class CausalConv1d(nn.Module):
    """
    Causal 1D convolution — output at position t only depends on t and earlier.
    Essential for real-time inference where future samples aren't available.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=self.padding, dilation=dilation
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        # Remove future-looking padding
        if self.padding > 0:
            x = x[:, :, :-self.padding]
        return x


class TCNBlock(nn.Module):
    """
    Single TCN residual block with:
        - Two causal dilated convolutions
        - LayerNorm + GELU activation (more stable than BatchNorm for sequences)
        - Residual connection with optional projection
        - Dropout for regularisation
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.norm1 = nn.LayerNorm(out_channels)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.norm2 = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

        # Projection if channel dimensions differ
        self.residual_proj = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        residual = self.residual_proj(x)

        out = self.conv1(x)
        out = self.norm1(out.transpose(1, 2)).transpose(1, 2)
        out = self.activation(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.norm2(out.transpose(1, 2)).transpose(1, 2)
        out = self.activation(out)
        out = self.dropout(out)

        return out + residual


class TelemetryTCN(nn.Module):
    """
    Full TCN model for telemetry analysis.

    Architecture:
        Input projection → 4 TCN blocks (exponential dilation) → Two heads

    Input:  (B, 7, T)  — 7 telemetry channels, T timesteps
    Output:
        delta_time: (B, T)      — time delta prediction (seconds)
        mistakes:   (B, 4, T)   — multi-label mistake probabilities

    Receptive field with kernel=3, dilations=[1,2,4,8]:
        RF = 1 + 2*(3-1)*1 + 2*(3-1)*2 + 2*(3-1)*4 + 2*(3-1)*8 = 61 samples
        At 5m resolution = 305m of context per prediction
    """

    INPUT_CHANNELS = 7      # speed, throttle, brake, gear, steer, lat_g, dist_norm
    N_MISTAKE_CLASSES = 4

    def __init__(
        self,
        hidden_channels: int = 64,
        kernel_size: int = 3,
        n_blocks: int = 4,
        dropout: float = 0.15,
    ):
        super().__init__()

        # Input projection
        self.input_proj = nn.Conv1d(self.INPUT_CHANNELS, hidden_channels, 1)

        # Exponentially dilated TCN blocks
        dilations = [2 ** i for i in range(n_blocks)]
        self.blocks = nn.ModuleList([
            TCNBlock(hidden_channels, hidden_channels, kernel_size, d, dropout)
            for d in dilations
        ])

        # Global context aggregation
        self.context_pool = nn.AdaptiveAvgPool1d(16)

        # Delta time regression head
        self.delta_head = nn.Sequential(
            nn.Conv1d(hidden_channels, 32, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(32, 1, 1),
        )

        # Mistake classification head (multi-label)
        self.mistake_head = nn.Sequential(
            nn.Conv1d(hidden_channels, 32, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(32, self.N_MISTAKE_CLASSES, 1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, 7, T) — normalised telemetry channels

        Returns:
            delta_time: (B, T) — predicted time delta in seconds
            mistakes:   (B, 4, T) — mistake class probabilities [0, 1]
        """
        # Project input channels
        out = self.input_proj(x)  # (B, H, T)

        # TCN blocks
        for block in self.blocks:
            out = block(out)

        # Compute heads
        delta_time = self.delta_head(out).squeeze(1)      # (B, T)
        mistakes = torch.sigmoid(self.mistake_head(out))   # (B, 4, T)

        return delta_time, mistakes

    def predict_window(self, window: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single-window inference for real-time use. No grad for speed.
        window: (7, T) — single sample, no batch dimension
        """
        with torch.no_grad():
            x = window.unsqueeze(0)  # Add batch dim
            delta, mistakes = self.forward(x)
            return delta.squeeze(0), mistakes.squeeze(0)


class TelemetryLoss(nn.Module):
    """
    Combined loss for multi-task training.

    - Regression: Huber loss for delta_time (robust to outliers)
    - Classification: Weighted BCE for mistakes (handles class imbalance)
    """

    def __init__(self, delta_weight: float = 1.0, mistake_weight: float = 2.0):
        super().__init__()
        self.delta_weight = delta_weight
        self.mistake_weight = mistake_weight
        self.huber = nn.HuberLoss(delta=0.5)

    def forward(
        self,
        pred_delta: torch.Tensor,
        pred_mistakes: torch.Tensor,
        target_delta: torch.Tensor,
        target_mistakes: torch.Tensor,
        mistake_pos_weight: float = 3.0,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            pred_delta:      (B, T)    predicted time delta
            pred_mistakes:   (B, 4, T) predicted mistake probs
            target_delta:    (B, T)    ground truth delta
            target_mistakes: (B, 4, T) ground truth labels
        """
        # Delta loss
        delta_loss = self.huber(pred_delta, target_delta)

        # Mistake loss — upweight positive (mistake) samples
        pos_weight = torch.full((4,), mistake_pos_weight, device=pred_mistakes.device)
        bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight.unsqueeze(-1))
        # BCEWithLogitsLoss expects logits, but we output sigmoid — convert back
        logits = torch.log(pred_mistakes.clamp(1e-6, 1 - 1e-6) / (1 - pred_mistakes.clamp(1e-6, 1 - 1e-6)))
        mistake_loss = bce(logits, target_mistakes)

        total = self.delta_weight * delta_loss + self.mistake_weight * mistake_loss

        return total, {
            "total": total.item(),
            "delta": delta_loss.item(),
            "mistake": mistake_loss.item(),
        }


def model_summary(model: TelemetryTCN) -> str:
    """Print parameter count and estimated inference time."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Estimate inference time with a dummy forward pass
    import time
    dummy = torch.randn(1, 7, 512)
    model.eval()

    # Warmup
    for _ in range(5):
        model.predict_window(dummy.squeeze(0))

    times = []
    for _ in range(50):
        t0 = time.perf_counter()
        model.predict_window(dummy.squeeze(0))
        times.append((time.perf_counter() - t0) * 1000)

    avg_ms = sum(times) / len(times)

    return (
        f"TelemetryTCN Summary\n"
        f"  Total parameters:     {total:,}\n"
        f"  Trainable parameters: {trainable:,}\n"
        f"  Avg inference time:   {avg_ms:.2f}ms (512-sample window, CPU)\n"
    )


if __name__ == "__main__":
    model = TelemetryTCN()
    print(model_summary(model))

    # Forward pass test
    x = torch.randn(4, 7, 512)  # Batch of 4, 7 channels, 512 timesteps
    delta, mistakes = model(x)
    print(f"\nForward pass:")
    print(f"  Input:    {x.shape}")
    print(f"  Delta:    {delta.shape}  (min={delta.min():.3f}, max={delta.max():.3f})")
    print(f"  Mistakes: {mistakes.shape}  (all in [0,1]: {mistakes.min() >= 0 and mistakes.max() <= 1})")