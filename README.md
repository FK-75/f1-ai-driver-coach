# 🏎️ F1 AI Driver Coach

> Real-time F1 telemetry analysis with a TCN model, WebSocket streaming backend, and a React dashboard. Trained on Hamilton and Verstappen qualifying data via FastF1.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)
![React](https://img.shields.io/badge/React-18-61dafb?style=flat-square)
![FastAPI](https://img.shields.io/badge/FastAPI-WebSocket-009688?style=flat-square)
![PyTorch](https://img.shields.io/badge/PyTorch-TCN-ee4c2c?style=flat-square)
![ONNX](https://img.shields.io/badge/ONNX-Runtime-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## Overview

F1 AI Driver Coach ingests telemetry channels — speed, throttle, brake, gear, lateral G, steering, distance delta — and runs them through a **Temporal Convolutional Network (TCN)** trained on real F1 qualifying telemetry. It compares inputs against a reference lap (Verstappen's 2023 Silverstone pole) and surfaces coaching cues in near real-time over WebSocket.

The core technical insight: **use track distance as the comparison axis, not timestamps.** Two drivers completing the same corner at different speeds will have different temporal profiles but identical distance profiles. This makes cross-driver comparison principled.

---

## What Makes This Technically Interesting

| Design Decision | Why It Matters |
|---|---|
| **Auto-labelled training data** | No human annotation. Late brakes, oversteer, understeer, and missed apices are derived algorithmically from telemetry delta |
| **Distance-axis DTW alignment** | Dynamic Time Warping on the distance axis gives robust cross-lap comparison without timestamp jitter artefacts |
| **TCN over Transformer** | 8ms vs 47ms CPU inference. For a real-time coaching loop, latency is a hard constraint |
| **Real champion data** | Trained on Hamilton and Verstappen lap telemetry — not synthetic or simulation data |
| **ONNX export** | Model is exported to ONNX Runtime for portable, dependency-light inference |
| **Multi-label mistake classifier** | Late brake, oversteer, understeer, and missed apex detected simultaneously per timestep |
| **LLM post-lap debrief** | Optional Ollama (Gemma) integration generates a 3-sentence technical coaching summary per lap |

---

## Architecture

```
FastF1 API (real F1 telemetry)
        │
        ▼
 telemetry.py              ← Distance-axis resampling, channel extraction
        │
        ▼
 labels.py                 ← Algorithmic mistake labelling (no annotation)
        │
        ▼
 alignment.py              ← DTW-based cross-lap alignment
        │
        ▼
 train.py + tcn.py         ← TelemetryTCN training (PyTorch, MPS/CUDA/CPU)
        │
        ▼
 export.py                 ← ONNX export (~8ms CPU inference)
        │
        ▼
 replay.py ──────────────► main.py (FastAPI WebSocket)
                                    │
                                    ▼
                           React Dashboard
                    LiveChart · TrackMap · CoachPanel
                    LapSelector · CompareLaps · CornerTable
```

---

## Demo

The demo replays Verstappen's 2023 Silverstone qualifying lap at real-time speed. The AI coach streams frame-by-frame telemetry, mistake probabilities, and a running time delta against the sector-optimal reference. No simulator required.

```bash
git clone https://github.com/FK-75/f1-ai-driver-coach
cd f1-ai-driver-coach

pip install -r requirements.txt
python scripts/download_data.py   # ~10–20 min first run (FastF1 caches locally)
python scripts/train.py           # ~5 min on CPU, faster on MPS/CUDA

python backend/api/main.py        # WebSocket server on :8000
# In a second terminal:
cd frontend && npm install && npm run dev  # React dev server on :5173
```

Open `http://localhost:5173` and press **Start Demo**.

> The backend auto-detects device: CUDA → MPS (Apple Silicon) → CPU.  
> If `data/models/tcn.onnx` exists, the backend loads it; otherwise it falls back to heuristic mode.

---

## Model Details

**Architecture:** 1D Temporal Convolutional Network with causal dilated convolutions  
**Input:** `(B, 7, T)` — 7 normalised telemetry channels over T timesteps  
**Outputs:**
- `delta_time` regression head `(B, T)` — predicted time delta vs reference in seconds
- `mistakes` multi-label head `(B, 4, T)` — per-timestep mistake probabilities

```
Input (7ch) → Input Projection
           → TCNBlock(dilation=1)
           → TCNBlock(dilation=2)
           → TCNBlock(dilation=4)
           → TCNBlock(dilation=8)
           → delta_head   → Δt (regression)
           → mistake_head → [late_brake, oversteer, understeer, missed_apex]
```

**Receptive field:** 61 samples = 305m of track context per prediction (at 5m resolution)  
**Training data:** 2021–2024 qualifying sessions across Silverstone, Monza, Monaco, Spa, Suzuka, Barcelona, Abu Dhabi, Bahrain, Singapore, Zandvoort  
**Loss:** Huber loss (delta time) + weighted BCE (mistakes, 3× positive class weight)  
**Inference:** ONNX Runtime, ~8ms on CPU per 512-sample window

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Model load status, inference mode |
| `/laps` | GET | All available training laps, grouped by circuit |
| `/fixture` | GET | Demo fixture metadata |
| `/reference` | GET | Reference lap track coordinates for map rendering |
| `/compare/{a}/{b}` | GET | Full telemetry overlay for two laps, resampled to shared distance axis |
| `/corner-report` | GET | Per-corner apex speed, time delta, dominant mistake |
| `/sector-report` | GET | Sector-level time delta and primary issue |
| `/llm-summary` | POST | LLM post-lap coaching debrief (requires local Ollama) |
| `ws://…/ws/replay` | WebSocket | Replay stream — send `{"action":"start"}` to begin |
| `ws://…/ws/live` | WebSocket | Live sim placeholder (F1 24 / Assetto Corsa UDP) |

WebSocket frame schema:
```json
{
  "type": "frame",
  "data": {
    "distance_m": 1240.5,
    "speed_kmh": 287.3,
    "throttle": 98.0,
    "brake": 0.0,
    "gear": 7,
    "lateral_g": 2.1,
    "delta_time_s": -0.043,
    "mistake_probs": {
      "late_brake": 0.08,
      "oversteer": 0.12,
      "understeer": 0.03,
      "missed_apex": 0.07
    },
    "ref_speed_kmh": 291.0,
    "x": 142.3,
    "y": -88.1
  }
}
```

---

## Tech Stack

| Layer | Technology | Rationale |
|---|---|---|
| Data | FastF1 | Real F1 telemetry, free, multi-season |
| ML | PyTorch TCN + ONNX Runtime | Sub-10ms CPU inference, portable export |
| Signal processing | SciPy | DTW alignment, braking zone detection |
| Backend | FastAPI + uvicorn WebSockets | Async, low-latency frame streaming |
| Frontend | React 18 + Recharts | Live charts, WebSocket hooks |
| LLM (optional) | Ollama / Gemma | Post-lap natural language debrief |

---

## Project Structure

```
f1-ai-driver-coach/
├── backend/
│   ├── api/
│   │   └── main.py               # FastAPI app — HTTP + WebSocket endpoints
│   ├── models/
│   │   ├── tcn.py                # TelemetryTCN architecture + TelemetryLoss
│   │   ├── train.py              # Training loop, dataset, sliding windows
│   │   └── export.py             # ONNX export + inference engine loader
│   └── pipeline/
│       ├── telemetry.py          # FastF1 fetching, distance-axis resampling
│       ├── labels.py             # Algorithmic mistake labelling
│       ├── alignment.py          # DTW cross-lap alignment, rolling delta
│       └── replay.py             # Demo replay engine, fixture builder
├── frontend/
│   └── src/
│       ├── components/
│       │   ├── LiveChart.jsx     # Real-time telemetry traces (Recharts)
│       │   ├── TrackMap.jsx      # Track position overlay with mistake highlights
│       │   ├── CoachPanel.jsx    # Coaching cues, mistake probabilities
│       │   ├── LapSelector.jsx   # Browse and select training laps by circuit
│       │   ├── CompareLaps.jsx   # Overlay two laps on shared distance axis
│       │   └── CornerTable.jsx   # Per-corner breakdown table
│       ├── hooks/
│       │   └── useWebSocket.js   # WebSocket connection + frame state management
│       ├── App.jsx               # Main dashboard layout
│       └── main.jsx
├── scripts/
│   ├── download_data.py          # Fetch FastF1 data and build demo fixture
│   ├── train.py                  # Training entrypoint (wraps backend/models/train.py)
│   └── calibrate_thresholds.py  # Tune mistake detection thresholds on real data
├── data/
│   └── .gitkeep                  # Data directory (populated by download_data.py)
├── requirements.txt
└── README.md
```

---

## Framing

> "Online imitation learning from expert demonstrations for motor skill coaching — comparing driver control inputs against expert reference traces to produce actionable corrective feedback via a real-time TCN inference pipeline."

FastF1 provides a legitimate public dataset of expert motor skill demonstrations from professional F1 drivers. The auto-labelling approach (deriving mistakes from telemetry delta) is directly analogous to inverse reinforcement learning: we infer what constitutes a mistake by observing where a driver deviates from an expert.

---

## Roadmap

- [ ] Live simulator UDP ingestion (F1 24 / Assetto Corsa / iRacing — port 20777)
- [ ] Sector-split model with per-sector reference selection
- [ ] Steering angle reconstruction from X/Y curvature (currently a proxy)
- [ ] Demo GIF / screen recording in README

---

## License

MIT — see [LICENSE](LICENSE).

*Built as a portfolio project combining ML, real-time systems engineering, and motorsport telemetry.*
