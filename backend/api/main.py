"""
main.py — FastAPI WebSocket backend for F1 AI Coach.

Endpoints:
    GET  /health              — Health check
    GET  /fixture             — Load demo fixture metadata
    GET  /reference           — Reference lap track data (for map rendering)
    WS   /ws/replay           — Demo replay stream
    WS   /ws/live             — Live sim data stream (future: F1 24 UDP)

The WebSocket streams JSON frames at real-time pace. The React frontend
connects, renders the live charts and track map, and displays coaching cues.
"""

import asyncio
import json
import pickle
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from backend.pipeline.replay import load_fixture, fixture_to_dataframes, replay_stream

# ── Corner name lookup ────────────────────────────────────────────────────────
CORNER_NAMES = {
    "Silverstone": {200:"Abbey",700:"Farm",950:"Village",1200:"The Loop",
        1550:"Aintree",1900:"Wellington",2200:"Brooklands",2500:"Luffield",
        2900:"Woodcote",3200:"Copse",3700:"Maggotts",3900:"Becketts",
        4200:"Chapel",4900:"Stowe",5200:"Vale",5500:"Club"},
    "Monaco": {150:"Ste Devote",500:"Massenet",700:"Casino",1000:"Mirabeau",
        1100:"Grand Hotel",1200:"Portier",1700:"Tunnel",1900:"Chicane",
        2100:"Tabac",2400:"Piscine",2700:"Rascasse",2900:"Noghes"},
    "Spa": {250:"La Source",750:"Eau Rouge",900:"Raidillon",2000:"Les Combes",
        2500:"Malmedy",3500:"Rivage",4200:"Pouhon",5500:"Campus",
        6100:"Stavelot",6600:"Blanchimont",6900:"Bus Stop"},
    "Monza": {200:"Rettifilo",700:"Variante 1",1500:"Curva Grande",
        2000:"Variante 2",3000:"Lesmo 1",3400:"Lesmo 2",4200:"Ascari",5100:"Parabolica"},
    "Suzuka": {200:"T1",600:"Dunlop",900:"Degner 1",1100:"Degner 2",
        1500:"Hairpin",2000:"Spoon S1",2400:"Spoon S2",3200:"130R",3600:"Casio"},
    "Abu Dhabi": {300:"T1",800:"T5",1500:"T9",2500:"T11",3500:"T14",4000:"T17",4500:"T19",5000:"T21"},
    "Singapore": {200:"T1",600:"T3",1000:"T5",1500:"Anderson",2000:"T10",2800:"T13",3500:"T16",4200:"T18",4800:"T20"},
    "Bahrain": {400:"T1",700:"T3",1200:"T4",2000:"T8",2800:"T10",3300:"T11",4000:"T13",4700:"T14"},
    "Zandvoort": {200:"Tarzan",900:"Hugenholtzbocht",1500:"Scheivlak",
        2000:"Mastersbocht",2600:"Luyendyk",3200:"Renaultbocht",4100:"Financierbocht"},
    "Barcelona": {400:"T1",800:"T3",1500:"T5",2200:"T7",2800:"T9",3500:"T10",4000:"T12",4600:"T14"},
    "Dutch": {200:"Tarzan",900:"Hugenholtzbocht",1500:"Scheivlak",
        2000:"Mastersbocht",2600:"Luyendyk",3200:"Renaultbocht",4100:"Financierbocht"},
    "British": {3200:"Copse",3700:"Maggotts",3900:"Becketts",4200:"Chapel",4900:"Stowe",5200:"Vale",5500:"Club"},
    "Italian": {200:"Rettifilo",700:"Variante 1",1500:"Curva Grande",2000:"Variante 2",3000:"Lesmo 1",3400:"Lesmo 2",4200:"Ascari",5100:"Parabolica"},
    "Japanese": {200:"T1",600:"Dunlop",900:"Degner 1",1100:"Degner 2",1500:"Hairpin",2000:"Spoon S1",2400:"Spoon S2",3200:"130R",3600:"Casio"},
    "Belgian": {250:"La Source",750:"Eau Rouge",900:"Raidillon",2000:"Les Combes",2500:"Malmedy",3500:"Rivage",4200:"Pouhon",6900:"Bus Stop"},
    "Spanish": {400:"T1",800:"T3",1500:"T5",2200:"T7",2800:"T9",3500:"T10",4000:"T12",4600:"T14"},
}


# ── App setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="F1 AI Driver Coach",
    description="Real-time telemetry analysis and AI-powered coaching",
    version="1.0.0",
)

# Allow React dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Shared state ──────────────────────────────────────────────────────────────

_fixture: Optional[dict] = None
_inference_engine = None
_lap_records: Optional[list] = None


def get_fixture() -> dict:
    global _fixture
    if _fixture is None:
        _fixture = load_fixture()
    return _fixture


def get_engine():
    global _inference_engine
    if _inference_engine is None:
        try:
            from backend.models.export import load_or_create_engine
            _inference_engine = load_or_create_engine()
            print("✅ Loaded ONNX inference engine")
        except Exception as e:
            print(f"⚠️  Could not load inference engine: {e}")
            print("   Running in heuristic mode (no model required)")
            _inference_engine = None
    return _inference_engine

def get_lap_records() -> list:
    """Load and cache training_laps.pkl."""
    global _lap_records
    if _lap_records is not None:
        return _lap_records
    pkl = ROOT / "data" / "training_laps.pkl"
    if not pkl.exists():
        _lap_records = []
        return _lap_records
    with open(pkl, "rb") as f:
        _lap_records = pickle.load(f)
    print(f"📚 Loaded {len(_lap_records)} laps from training_laps.pkl")
    return _lap_records


def build_fixture_for_lap(lap_id: int) -> dict:
    """Build a replay fixture from a real training lap record."""
    records = get_lap_records()
    if not records or lap_id >= len(records):
        return get_fixture()

    drv_rec = records[lap_id]
    drv_df = drv_rec["df"].copy()
    gp, year, driver = drv_rec["gp"], drv_rec["year"], drv_rec["driver"]

    same = [r for r in records if r["gp"] == gp and r["year"] == year and r["driver"] != driver]
    if same:
        ver = [r for r in same if r["driver"] == "VER"]
        ref_rec = ver[0] if ver else min(same, key=lambda r: r["lap_time_s"])
    else:
        ref_rec = drv_rec

    ref_df = ref_rec["df"].copy()
    T = min(len(drv_df), len(ref_df))
    drv_df = drv_df.iloc[:T].reset_index(drop=True)
    ref_df = ref_df.iloc[:T].reset_index(drop=True)

    def add_xy(df):
        # Only synthesise coordinates if real GPS data is absent or all-zero
        if "x" in df.columns and "y" in df.columns:
            if df["x"].abs().max() > 1.0 and df["y"].abs().max() > 1.0:
                return df  # real coords already present — keep them
        dist = df["distance"].values
        total = max(float(dist[-1]), 1.0)
        theta = dist / total * 2 * np.pi
        r = 800 + 200 * np.sin(3 * theta)
        df = df.copy()
        df["x"] = r * np.cos(theta)
        df["y"] = r * np.sin(theta)
        return df

    def recompute_lat_g(df):
        """
        Recompute lateral_g from speed and steer curvature proxy.
        The pkl lateral_g from curvature of 5m-resampled X/Y is near-zero
        because interpolation smooths corners. Speed-based approximation is
        more realistic for coaching display purposes.
        """
        speed_ms = df["speed"].values / 3.6
        # Use steer as curvature proxy: steer is normalised signed curvature
        # Scale factor tuned so peak steer (~0.3) at 200km/h gives ~3G
        steer = df["steer"].values if "steer" in df.columns else np.zeros(len(df))
        lat_g = speed_ms**2 * np.abs(steer) * 0.02 / 9.81  # steer = curvature/0.02, scale=0.02
        lat_g = np.clip(lat_g, 0, 5.0)
        return lat_g

    def to_dict(df):
        cols = ["distance", "speed", "throttle", "brake", "gear", "steer", "x", "y"]
        d = {c: df[c].tolist() if c in df.columns else [0.0] * len(df) for c in cols}
        d["lateral_g"] = recompute_lat_g(df).tolist()
        return d

    drv_df = add_xy(drv_df)
    ref_df = add_xy(ref_df)
    track_name = gp.replace(" Grand Prix", "").replace(" GP", "")
    print(f"🏎️  Real lap: {driver} vs {ref_rec['driver']} @ {gp} {year}")

    return {
        "driver": driver,
        "reference_driver": ref_rec["driver"],
        "gp": gp, "year": year,
        "track_name": track_name,
        "lap_time_s": round(float(drv_rec["lap_time_s"]), 3),
        "reference_lap_time_s": round(float(ref_rec["lap_time_s"]), 3),
        "driver_data": to_dict(drv_df),
        "reference_data": to_dict(ref_df),
    }


# ── HTTP endpoints ────────────────────────────────────────────────────────────

@app.get("/laps")
async def list_laps():
    """List all available real laps from training_laps.pkl, grouped by circuit."""
    records = get_lap_records()
    laps = []
    for i, r in enumerate(records):
        laps.append({
            "id": i,
            "driver": r["driver"],
            "gp": r["gp"],
            "year": r["year"],
            "session": r.get("session", "Q"),
            "lap_time_s": round(float(r["lap_time_s"]), 3),
            "n_samples": len(r["df"]),
        })
    circuits: dict[str, list] = {}
    for lap in laps:
        key = f"{lap['gp']} {lap['year']}"
        circuits.setdefault(key, []).append(lap)
    return {"laps": laps, "total": len(laps), "circuits": circuits}


@app.get("/health")
async def health():
    engine = get_engine()
    return {
        "status": "ok",
        "model_loaded": engine is not None,
        "inference_mode": "onnx" if engine else "heuristic",
    }


@app.get("/fixture")
async def fixture_info():
    f = get_fixture()
    return {
        "driver": f["driver"],
        "reference_driver": f["reference_driver"],
        "gp": f["gp"],
        "year": f["year"],
        "track_name": f.get("track_name", "Silverstone"),
        "lap_time_s": f["lap_time_s"],
        "reference_lap_time_s": f["reference_lap_time_s"],
        "total_samples": len(f["driver_data"]["distance"]),
    }


@app.get("/reference")
async def reference_data():
    """Return reference lap track coordinates for map rendering."""
    f = get_fixture()
    ref = f["reference_data"]
    return {
        "x": ref["x"],
        "y": ref["y"],
        "distance": ref["distance"],
        "speed": ref["speed"],
    }


@app.get("/driver-track")
async def driver_track_data():
    """Return driver lap track coordinates."""
    f = get_fixture()
    d = f["driver_data"]
    return {
        "x": d["x"],
        "y": d["y"],
        "distance": d["distance"],
        "speed": d["speed"],
    }


# ── WebSocket: Demo replay ────────────────────────────────────────────────────

@app.websocket("/ws/replay")
async def websocket_replay(
    websocket: WebSocket,
    speed: float = Query(default=1.0, ge=0.1, le=20.0),
    lap_id: int = Query(default=-1),
):
    """
    Stream demo replay frames over WebSocket.

    Client sends:
        { "action": "start" }  — begin replay
        { "action": "stop" }   — stop replay
        { "action": "pause" }  — pause replay

    Server sends:
        { "type": "frame", "data": { ...telemetry frame... } }
        { "type": "complete", "data": { ...lap summary... } }
        { "type": "error", "data": { "message": "..." } }
    """
    await websocket.accept()
    print(f"🔌 WebSocket connected (speed={speed}x, lap_id={lap_id})")

    fixture = build_fixture_for_lap(lap_id) if lap_id >= 0 else get_fixture()
    # Update the global fixture so /reference and /driver-track serve the correct lap
    global _fixture
    _fixture = fixture
    engine = get_engine()

    try:
        # Wait for start signal
        msg = await websocket.receive_text()
        data = json.loads(msg)

        if data.get("action") != "start":
            await websocket.send_json({"type": "error", "data": {"message": "Send {action: 'start'} to begin"}})
            return

        # Send fixture metadata first so the frontend can update the header immediately
        await websocket.send_json({
            "type": "fixture",
            "data": {
                "driver": fixture["driver"],
                "reference_driver": fixture["reference_driver"],
                "gp": fixture["gp"],
                "year": fixture["year"],
                "track_name": fixture.get("track_name", ""),
                "lap_time_s": fixture["lap_time_s"],
                "reference_lap_time_s": fixture["reference_lap_time_s"],
            }
        })

        # Collect lap stats for summary
        frames_sent = 0
        final_delta = 0.0
        all_mistakes: dict[str, int] = {}

        # Stream replay
        async for frame in replay_stream(fixture, speed_multiplier=speed, inference_engine=engine):
            await websocket.send_json({"type": "frame", "data": frame})
            frames_sent += 1
            final_delta = frame["delta_time_s"]

            # Accumulate mistake counts
            for mistake, prob in frame["mistake_probs"].items():
                if prob > 0.5:
                    all_mistakes[mistake] = all_mistakes.get(mistake, 0) + 1

            # Check for stop/pause from client (non-blocking)
            try:
                client_msg = await asyncio.wait_for(websocket.receive_text(), timeout=0.001)
                client_data = json.loads(client_msg)
                if client_data.get("action") == "stop":
                    break
            except asyncio.TimeoutError:
                pass
            except WebSocketDisconnect:
                break
            except Exception:
                pass

        # Send lap summary
        top_mistakes = sorted(all_mistakes.items(), key=lambda x: x[1], reverse=True)[:3]
        await websocket.send_json({
            "type": "complete",
            "data": {
                "frames_sent": frames_sent,
                "final_delta_s": final_delta,
                "top_mistakes": [{"type": m, "count": c} for m, c in top_mistakes],
                "lap_time_s": fixture["lap_time_s"],
                "reference_lap_time_s": fixture["reference_lap_time_s"],
            }
        })

    except WebSocketDisconnect:
        print("🔌 Client disconnected")
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"❌ WebSocket error: {e}")
        try:
            await websocket.send_json({"type": "error", "data": {"message": str(e)}})
        except Exception:
            pass


# ── WebSocket: Live sim (future) ──────────────────────────────────────────────

@app.websocket("/ws/live")
async def websocket_live(websocket: WebSocket):
    """
    Placeholder for live simulator UDP data stream.
    F1 24 / Assetto Corsa sends telemetry via UDP on port 20777.
    Future: parse UDP packets and stream through inference pipeline.
    """
    await websocket.accept()
    await websocket.send_json({
        "type": "info",
        "data": {
            "message": "Live sim mode not yet enabled. Use /ws/replay for demo.",
            "supported_sims": ["F1 24", "Assetto Corsa", "iRacing"],
            "udp_port": 20777,
        }
    })
    await websocket.close()


# ── Compare laps ─────────────────────────────────────────────────────────────

@app.get("/compare/{lap_id_a}/{lap_id_b}")
async def compare_laps(lap_id_a: int, lap_id_b: int):
    """
    Return full telemetry for two laps so the frontend can overlay them.
    Both laps are resampled to the same distance axis (lap A's grid).
    Returns speed, throttle, brake, gear, lateral_g, delta arrays.
    """
    import pandas as pd
    records = get_lap_records()

    def get_rec(lap_id):
        if lap_id < 0 or lap_id >= len(records):
            return None
        return records[lap_id]

    rec_a = get_rec(lap_id_a)
    rec_b = get_rec(lap_id_b)

    if rec_a is None or rec_b is None:
        return JSONResponse({"error": "Invalid lap ID"}, status_code=404)

    df_a = rec_a["df"].copy()
    df_b = rec_b["df"].copy()

    # Resample lap B onto lap A's distance axis for clean comparison
    dist_a = df_a["distance"].values
    dist_b = df_b["distance"].values
    max_dist = min(dist_a[-1], dist_b[-1])
    mask_a = dist_a <= max_dist
    dist_a = dist_a[mask_a]
    df_a = df_a[mask_a]

    channels = ["speed", "throttle", "brake", "gear", "steer", "lateral_g"]
    b_interp = {}
    for ch in channels:
        if ch in df_b.columns:
            b_interp[ch] = np.interp(dist_a, dist_b, df_b[ch].values).tolist()
        else:
            b_interp[ch] = [0.0] * len(dist_a)

    # Recompute lateral_g for both using steer proxy
    def lat_g_from_steer(speed_arr, steer_arr):
        speed_ms = np.array(speed_arr) / 3.6
        steer = np.abs(np.array(steer_arr))
        from scipy.ndimage import uniform_filter1d
        steer = uniform_filter1d(steer, size=5)
        lat_g = speed_ms**2 * steer * 0.02 / 9.81
        return np.clip(lat_g, 0, 5.5).tolist()

    a_speed = df_a["speed"].values.tolist()
    a_steer = df_a["steer"].values.tolist() if "steer" in df_a.columns else [0.0]*len(dist_a)
    b_speed = b_interp["speed"]
    b_steer = b_interp["steer"]

    # Compute cumulative time delta: A faster than B = negative delta
    from backend.pipeline.alignment import compute_rolling_time_delta
    tmp_a = df_a.copy()
    tmp_b = pd.DataFrame({ch: b_interp[ch] for ch in channels if ch in b_interp})
    tmp_b["distance"] = dist_a
    T = min(len(tmp_a), len(tmp_b))
    delta = compute_rolling_time_delta(tmp_a.iloc[:T], tmp_b.iloc[:T])

    return {
        "lap_a": {
            "id": lap_id_a,
            "driver": rec_a["driver"],
            "gp": rec_a["gp"],
            "year": rec_a["year"],
            "lap_time_s": round(float(rec_a["lap_time_s"]), 3),
            "distance": dist_a.tolist(),
            "speed": a_speed,
            "throttle": df_a["throttle"].values.tolist(),
            "brake": df_a["brake"].values.tolist(),
            "gear": df_a["gear"].values.tolist(),
            "lateral_g": lat_g_from_steer(a_speed, a_steer),
        },
        "lap_b": {
            "id": lap_id_b,
            "driver": rec_b["driver"],
            "gp": rec_b["gp"],
            "year": rec_b["year"],
            "lap_time_s": round(float(rec_b["lap_time_s"]), 3),
            "distance": dist_a.tolist(),
            "speed": b_speed,
            "throttle": b_interp["throttle"],
            "brake": b_interp["brake"],
            "gear": b_interp["gear"],
            "lateral_g": lat_g_from_steer(b_speed, b_steer),
        },
        "delta": delta.tolist(),  # positive = A is ahead of B (A faster at this point)
        "max_distance_m": float(max_dist),
    }


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n🏎️  F1 AI Driver Coach — Backend\n")
    print(f"   API docs:  http://localhost:8000/docs")
    print(f"   Health:    http://localhost:8000/health")
    print(f"   Laps:      http://localhost:8000/laps")
    print(f"   WebSocket: ws://localhost:8000/ws/replay\n")

    uvicorn.run(
        "backend.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,          # reload=True kills active WebSocket connections on file changes
        log_level="info",
        ws_ping_interval=20,   # keep-alive ping every 20s
        ws_ping_timeout=30,    # allow 30s for pong before dropping
    )

@app.get("/sector-report")
async def sector_report():
    """Return pre-computed sector time deltas for the demo lap."""
    from backend.pipeline.labels import generate_labels, compute_sector_report
    from backend.pipeline.alignment import compute_rolling_time_delta

    f = get_fixture()
    driver_df = __import__('pandas').DataFrame(f["driver_data"])
    reference_df = __import__('pandas').DataFrame(f["reference_data"])
    T = min(len(driver_df), len(reference_df))
    labels = generate_labels(driver_df.iloc[:T], reference_df.iloc[:T])
    report = compute_sector_report(driver_df.iloc[:T], reference_df.iloc[:T], labels)
    return {"sectors": report}


# ── Corner report ─────────────────────────────────────────────────────────────

@app.get("/corner-report")
async def corner_report():
    """
    Return per-corner breakdown: apex speed, time delta, dominant mistake.
    Uses the reference lap apices as canonical corner positions.
    """
    import pandas as pd
    # from backend.pipeline.labels import (
    #     generate_labels, find_corner_apices, MISTAKE_NAMES, compute_rolling_time_delta
    # )
    from backend.pipeline.labels import (
        generate_labels, find_corner_apices, MISTAKE_NAMES
    )
    from backend.pipeline.alignment import compute_rolling_time_delta
    f = get_fixture()
    driver_df = pd.DataFrame(f["driver_data"])
    reference_df = pd.DataFrame(f["reference_data"])
    T = min(len(driver_df), len(reference_df))

    drv = driver_df.iloc[:T]
    ref = reference_df.iloc[:T]

    labels = generate_labels(drv, ref)
    delta = compute_rolling_time_delta(drv, ref)
    apices = find_corner_apices(ref)

    corners = []
    for corner_num, apex_idx in enumerate(apices, 1):
        win = slice(max(0, apex_idx - 6), min(T, apex_idx + 6))
        drv_apex_speed = float(drv["speed"].values[win].min())
        ref_apex_speed = float(ref["speed"].values[win].min())
        speed_delta = drv_apex_speed - ref_apex_speed

        # Time delta at the apex
        apex_delta = float(delta[apex_idx])

        # Dominant mistake in ±20 samples around apex
        zone = slice(max(0, apex_idx - 20), min(T, apex_idx + 20))
        zone_labels = labels[zone]
        mistake_counts = {MISTAKE_NAMES[i]: float(zone_labels[:, i].mean()) for i in range(4)}
        dominant = max(mistake_counts, key=mistake_counts.get)
        dominant_prob = mistake_counts[dominant]

        dist_m = round(float(drv["distance"].values[apex_idx]), 0)
        track = f.get("track_name", "")
        name_map = CORNER_NAMES.get(track, {})
        corner_name = None
        if name_map:
            nearest = min(name_map.keys(), key=lambda d: abs(d - dist_m))
            if abs(nearest - dist_m) < 500:
                corner_name = name_map[nearest]

        corners.append({
            "corner": corner_num,
            "distance_m": dist_m,
            "drv_apex_speed_kmh": round(drv_apex_speed, 1),
            "ref_apex_speed_kmh": round(ref_apex_speed, 1),
            "speed_delta_kmh": round(speed_delta, 1),
            "time_delta_s": round(apex_delta, 3),
            "dominant_mistake": dominant if dominant_prob > 0.2 else None,
            "mistake_prob": round(dominant_prob, 2),
            "name": corner_name,
        })

    return {"corners": corners, "total": len(corners)}


# ── LLM coaching summary ─────────────────────────────────────────────────────

@app.post("/llm-summary")
async def llm_summary(request: Request):
    """
    Generate a personalised post-lap coaching debrief via local Ollama.
    Payload: { sector_report, corner_report, lap_summary, driver, reference_driver }
    """
    import httpx
    import re
    payload = await request.json()

    sector_report = payload.get("sector_report", [])
    corner_report = payload.get("corner_report", [])
    lap_summary = payload.get("lap_summary", {})
    driver = payload.get("driver", "HAM")
    reference_driver = payload.get("reference_driver", "VER")
    track = payload.get("track", "Silverstone")

    lap_time = lap_summary.get("lap_time_s", 0)
    ref_time = lap_summary.get("reference_lap_time_s", 0)
    gap = round(lap_time - ref_time, 3)

    # Build top corners that lost the most time
    sorted_corners = sorted(corner_report, key=lambda c: c.get("time_delta_s", 0), reverse=True)
    worst_corners = sorted_corners[:3]

    # Build sector summary string
    sector_lines = []
    for s in sector_report:
        sector_lines.append(
            f"  S{s['sector']}: {s['time_delta_s']:+.3f}s | "
            f"avg speed delta {s['avg_speed_delta_kmh']:+.1f} km/h | "
            f"primary issue: {s.get('primary_issue') or 'none'}"
        )

    corner_lines = []
    for c in worst_corners:
        corner_lines.append(
            f"  Corner {c['corner']} ({c['distance_m']:.0f}m): "
            f"apex speed {c['drv_apex_speed_kmh']:.0f} vs {c['ref_apex_speed_kmh']:.0f} km/h ref "
            f"({c['speed_delta_kmh']:+.1f}), delta {c['time_delta_s']:+.3f}s"
            + (f", dominant issue: {c['dominant_mistake']}" if c['dominant_mistake'] else "")
        )

    prompt = f"""You are a professional F1 driver coach giving post-lap feedback. Be direct, specific, and technical. No filler phrases.

Lap data:
- Driver: {driver} vs reference {reference_driver} at {track}
- Lap time: {lap_time:.3f}s (reference: {ref_time:.3f}s, gap: {gap:+.3f}s)

Sector breakdown:
{chr(10).join(sector_lines)}

Worst corners:
{chr(10).join(corner_lines)}

Write exactly 3 SHORT sentences (25 words max each). Complete all 3 sentences fully.
1. Where the biggest time was lost and why.
2. The single most important technique fix.
3. One positive observation.
No bullet points. Technical F1 language only. Do not stop mid-sentence."""

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                "http://localhost:11434/api/chat",
                json={
                    "model": "gemma4:e4b",
                    "stream": False,
                    "options": {
                        "temperature": 1.0,
                        "top_p": 0.95,
                        "top_k": 64,
                        "num_predict": 800,
                    },
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a professional F1 driver coach. Give direct, technical, specific feedback. No filler."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                },
            )
            data = response.json()
            msg = data.get("message") or {}
            raw = msg.get("content") or data.get("response") or ""
            # Strip Gemma 4 thinking tags if present
            if "<channel|>" in raw:
                cleaned = raw.split("<channel|>")[-1].strip()
            elif "<think>" in raw:
                cleaned = raw.split("</think>")[-1].strip()
            else:
                cleaned = raw.strip()
            print(f"✅ LLM debrief ({len(cleaned)} chars, reason={data.get('done_reason')})")
            return {"summary": cleaned or "Lap complete — check sector breakdown.", "ok": True}
    except Exception as e:
        import traceback
        print(f"LLM error: {e}")
        traceback.print_exc()
        return {
            "summary": f"Lap complete. Gap to reference: {gap:+.3f}s. Check sector breakdown for details.",
            "ok": False,
            "error": str(e),
        }