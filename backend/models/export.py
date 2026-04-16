"""
export.py — Export trained TelemetryTCN to ONNX for fast CPU inference.

ONNX Runtime on CPU is ~2-3x faster than PyTorch for inference-only workloads,
which is what makes sub-10ms real-time coaching viable.
"""

import numpy as np
import torch
from pathlib import Path
from typing import Optional
import time


def export_to_onnx(model, output_path: Path, window_size: int = 512) -> Path:
    """
    Export TelemetryTCN to ONNX with dynamic sequence length.

    The model supports variable-length sequences, but we fix the window size
    at export time for maximum ONNX Runtime optimisation.
    """
    model.eval()
    dummy_input = torch.randn(1, 7, window_size)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["telemetry"],
        output_names=["delta_time", "mistakes"],
        dynamic_axes={
            "telemetry": {0: "batch_size"},
            "delta_time": {0: "batch_size"},
            "mistakes": {0: "batch_size"},
        },
        verbose=False,
    )

    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"✅ Exported ONNX model to: {output_path} ({size_mb:.2f} MB)")
    return output_path


class ONNXInferenceEngine:
    """
    ONNX Runtime inference engine for real-time telemetry coaching.

    Wraps the exported ONNX model with:
        - Session initialisation with CPU optimisation flags
        - Window management for streaming inference
        - Latency tracking
    """

    def __init__(self, model_path: Path, window_size: int = 512):
        import onnxruntime as ort

        self.window_size = window_size
        self.model_path = Path(model_path)

        # CPU optimisation options
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )

        self.input_name = self.session.get_inputs()[0].name
        self._buffer = np.zeros((7, window_size), dtype=np.float32)
        self._latencies_ms = []

        print(f"🔧 ONNX Runtime session loaded: {self.model_path.name}")
        print(f"   Input: {self.input_name} {self.session.get_inputs()[0].shape}")

    def update_buffer(self, new_sample: np.ndarray) -> None:
        """
        Slide the window buffer by one sample.
        new_sample: (7,) array of normalised telemetry channels
        """
        self._buffer = np.roll(self._buffer, -1, axis=1)
        self._buffer[:, -1] = new_sample

    def infer(self) -> tuple[float, np.ndarray]:
        """
        Run inference on the current window buffer.

        Returns:
            delta_time: float — predicted time delta in seconds
            mistakes:   np.ndarray (6,) — mistake class probabilities
        """
        t0 = time.perf_counter()

        # Add batch dimension: (1, 7, W)
        feed = {self.input_name: self._buffer[np.newaxis, :, :]}
        delta_time_arr, mistakes_arr = self.session.run(None, feed)

        latency_ms = (time.perf_counter() - t0) * 1000
        self._latencies_ms.append(latency_ms)

        # Return last timestep prediction
        delta_time = float(delta_time_arr[0, -1])
        mistakes = mistakes_arr[0, :, -1]  # (6,)

        return delta_time, mistakes

    def infer_from_features(self, features: np.ndarray) -> tuple[float, np.ndarray]:
        """
        Run inference on a pre-computed (7, W) feature window.
        """
        t0 = time.perf_counter()
        feed = {self.input_name: features[np.newaxis, :, :].astype(np.float32)}
        delta_time_arr, mistakes_arr = self.session.run(None, feed)
        self._latencies_ms.append((time.perf_counter() - t0) * 1000)
        return float(delta_time_arr[0, -1]), mistakes_arr[0, :, -1]

    @property
    def avg_latency_ms(self) -> float:
        if not self._latencies_ms:
            return 0.0
        return sum(self._latencies_ms[-50:]) / min(50, len(self._latencies_ms))

    def benchmark(self, n_runs: int = 100) -> dict:
        """Benchmark inference latency."""
        dummy = np.random.randn(7, self.window_size).astype(np.float32)

        # Warmup
        for _ in range(10):
            self.infer_from_features(dummy)

        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            self.infer_from_features(dummy)
            times.append((time.perf_counter() - t0) * 1000)

        return {
            "mean_ms": np.mean(times),
            "p50_ms": np.percentile(times, 50),
            "p95_ms": np.percentile(times, 95),
            "p99_ms": np.percentile(times, 99),
            "min_ms": np.min(times),
            "max_ms": np.max(times),
        }


def load_or_create_engine(
    onnx_path: Optional[Path] = None,
    pt_path: Optional[Path] = None,
    window_size: int = 512,
) -> ONNXInferenceEngine:
    """
    Load ONNX model if it exists, otherwise load PyTorch and export.
    """
    from pathlib import Path as P
    ROOT = P(__file__).parent.parent.parent
    default_onnx = ROOT / "data" / "models" / "tcn.onnx"
    default_pt = ROOT / "data" / "models" / "tcn_best.pt"

    onnx_path = Path(onnx_path) if onnx_path else default_onnx
    pt_path = Path(pt_path) if pt_path else default_pt

    if not onnx_path.exists():
        if pt_path.exists():
            print("⚡ ONNX model not found — exporting from PyTorch checkpoint...")
            from backend.models.train import load_model
            model = load_model(pt_path)
            export_to_onnx(model, onnx_path, window_size)
        else:
            raise FileNotFoundError(
                f"Neither ONNX ({onnx_path}) nor PyTorch ({pt_path}) model found. "
                "Run python scripts/train.py first."
            )

    return ONNXInferenceEngine(onnx_path, window_size)


if __name__ == "__main__":
    # Benchmark test
    import sys
    from pathlib import Path
    ROOT = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(ROOT))

    onnx_path = ROOT / "data" / "models" / "tcn.onnx"
    if not onnx_path.exists():
        print("No ONNX model found. Creating one from scratch for benchmarking...")
        from backend.models.tcn import TelemetryTCN
        model = TelemetryTCN()
        export_to_onnx(model, onnx_path)

    engine = ONNXInferenceEngine(onnx_path)
    results = engine.benchmark(200)
    print(f"\n⚡ Inference Benchmark (n=200):")
    for k, v in results.items():
        print(f"   {k}: {v:.2f}ms")
