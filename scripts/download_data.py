"""
download_data.py — Fetch FastF1 training data and demo fixture.

Run this once before training. Downloads and caches F1 telemetry locally.
Takes ~10–20 minutes on first run depending on connection speed (all sessions
are cached locally after that, so subsequent runs are fast).

Usage:
    python scripts/download_data.py
"""

import sys
import json
import pickle
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from backend.pipeline.telemetry import fetch_training_data, load_reference_lap
from backend.pipeline.replay import create_demo_fixture, FIXTURE_PATH

DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

# ── Training events ───────────────────────────────────────────────────────────
#
# Chosen for circuit variety — the model generalises better when it has seen
# very different corner profiles:
#
#   Silverstone   — high-speed sweepers (Copse, Maggotts/Becketts)
#   Monza         — low-downforce, heavy braking zones, chicanes
#   Monaco        — ultra-slow, tight streets, maximum braking label density
#   Spa           — Eau Rouge high-speed entry, long La Source hairpin
#   Suzuka        — technical S-curves, 130R, slow hairpin
#   Barcelona     — balanced mix, good reference circuit
#
# Three years per circuit: 2021 (pre-regs), 2022 (new regs), 2023 (evolved cars).
# Four new circuits added: Abu Dhabi, Bahrain, Singapore, Zandvoort.

# ── Circuit selection rationale ───────────────────────────────────────────────
#
# Original 6 circuits retained (Silverstone, Monza, Monaco, Spa, Suzuka, Barcelona)
# New additions chosen to maximise corner-profile diversity:
#
#   Abu Dhabi     — long straights, Sector 3 technical complex, night race
#   Bahrain       — high-deg tarmac, heavy braking Turn 1, balanced layout
#   Singapore     — street circuit, tight walls, maximum late-brake density
#   Zandvoort     — banked Arie Luyendyk corner, unique camber profiles
#   Silverstone Q (2021) — adds pre-2022 car regulation era
#   Monaco Q (2021)      — ultra-slow era contrast
#
# Year coverage: 2021 (pre-regs), 2022 (new regs), 2023 (evolved cars)
# This gives the model three distinct car characteristic eras to generalise across.

EVENTS = [
    # ── 2023 ─────────────────────────────────────────────────────────────────
    (2023, "British Grand Prix",      "Q"),  # Silverstone — high-speed sweepers
    (2023, "Italian Grand Prix",      "Q"),  # Monza — low df, heavy braking
    (2023, "Monaco Grand Prix",       "Q"),  # Monaco — slow/tight, braking-heavy
    (2023, "Belgian Grand Prix",      "Q"),  # Spa — Eau Rouge, La Source
    (2023, "Japanese Grand Prix",     "Q"),  # Suzuka — S-curves, 130R
    (2023, "Spanish Grand Prix",      "Q"),  # Barcelona — balanced reference
    (2023, "Abu Dhabi Grand Prix",    "Q"),  # New: long straights, Sector 3 complex
    (2023, "Bahrain Grand Prix",      "Q"),  # New: heavy braking, high deg
    (2023, "Singapore Grand Prix",    "Q"),  # New: street circuit, tight walls
    (2023, "Dutch Grand Prix",        "Q"),  # New: Zandvoort — banked corners
    # ── 2022 ─────────────────────────────────────────────────────────────────
    (2022, "British Grand Prix",      "Q"),  # Silverstone — 2022 regs
    (2022, "Italian Grand Prix",      "Q"),  # Monza — 2022 regs
    (2022, "Monaco Grand Prix",       "Q"),  # Monaco — 2022 regs
    (2022, "Belgian Grand Prix",      "Q"),  # Spa — 2022 regs
    (2022, "Japanese Grand Prix",     "Q"),  # Suzuka — 2022 regs
    (2022, "Spanish Grand Prix",      "Q"),  # Barcelona — 2022 regs
    (2022, "Abu Dhabi Grand Prix",    "Q"),  # New: Abu Dhabi 2022 regs
    (2022, "Bahrain Grand Prix",      "Q"),  # New: Bahrain 2022 regs
    (2022, "Singapore Grand Prix",    "Q"),  # New: Singapore 2022 regs
    (2022, "Dutch Grand Prix",        "Q"),  # New: Zandvoort 2022 regs
    # ── 2021 (pre-ground-effect regulations) ─────────────────────────────────
    (2021, "British Grand Prix",      "Q"),  # pre-2022 car era
    (2021, "Italian Grand Prix",      "Q"),  # pre-2022 car era
    (2021, "Monaco Grand Prix",       "Q"),  # pre-2022 car era
    (2021, "Belgian Grand Prix",      "Q"),  # pre-2022 car era
    (2021, "Abu Dhabi Grand Prix",    "Q"),  # pre-2022 car era
    (2021, "Bahrain Grand Prix",      "Q"),  # pre-2022 car era
    # ── 2024 (further evolved ground-effect cars) ─────────────────────────────
    (2024, "British Grand Prix",      "Q"),  # Silverstone 2024
    (2024, "Monaco Grand Prix",       "Q"),  # Monaco 2024
    (2024, "Italian Grand Prix",      "Q"),  # Monza 2024
    (2024, "Japanese Grand Prix",     "Q"),  # Suzuka 2024
    (2024, "Abu Dhabi Grand Prix",    "Q"),  # Abu Dhabi 2024
    (2024, "Singapore Grand Prix",    "Q"),  # Singapore 2024
]

# Drivers chosen for style diversity across the dataset
# Note: NOR and RUS did not race in 2021 — handled gracefully by fetch_training_data
DRIVERS = [
    "VER",  # Verstappen — dominant across all 3 eras, excellent reference
    "HAM",  # Hamilton — consistent across all 3 eras, diverse styles
    "LEC",  # Leclerc — aggressive entry, good oversteer examples
    "NOR",  # Norris — smooth, consistent (2022/2023 only)
    "SAI",  # Sainz — technically precise, sector 2 specialist
    "BOT",  # Bottas — experienced, present in all 3 eras
    "ALO",  # Alonso — unusual braking profiles, all 3 eras
    "RUS",  # Russell — clean, systematic (2022/2023 only)
]

TOTAL_EXPECTED = len(EVENTS) * len(DRIVERS)  # 256 laps target (~230 actual, some drivers missing in 2021/2024)


def main():
    print("=" * 60)
    print("  F1 AI Driver Coach — Data Download")
    print(f"  {len(EVENTS)} sessions × {len(DRIVERS)} drivers = {TOTAL_EXPECTED} laps target")
    print("=" * 60)

    # ── 1. Demo fixture ───────────────────────────────────────────────────────
    print("\n[1/3] Creating demo fixture (synthetic Silverstone lap)...")
    fixture = create_demo_fixture()
    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FIXTURE_PATH, "w") as f:
        json.dump(fixture, f, indent=2)
    print(f"  ✅ Demo fixture saved to {FIXTURE_PATH}")

    # ── 2. Reference lap ─────────────────────────────────────────────────────
    print("\n[2/3] Downloading reference lap (VER 2023 Silverstone Q)...")
    print("  This may take 1–2 minutes on first run (FastF1 caching)...")
    try:
        ref = load_reference_lap("British Grand Prix", 2023, "VER")
        with open(DATA_DIR / "reference_lap.pkl", "wb") as f:
            pickle.dump(ref, f)
        print(f"  ✅ Reference: {ref['lap_time_s']:.3f}s, {len(ref['df'])} samples")
    except Exception as e:
        print(f"  ⚠️  Could not download reference lap: {e}")
        print("  → Demo mode will use synthetic fixture instead (still fully functional)")

    # ── 3. Training laps ──────────────────────────────────────────────────────
    print(f"\n[3/3] Downloading {TOTAL_EXPECTED} training laps across {len(EVENTS)} sessions...")
    print("  First run: ~10–20 min (FastF1 caches each session after download)")
    print("  Subsequent runs: ~1–2 min (served from local cache)\n")

    try:
        records = fetch_training_data(
            events=EVENTS,
            drivers=DRIVERS,
            save_dir=DATA_DIR,
        )

        success_rate = len(records) / TOTAL_EXPECTED * 100
        print(f"\n  ✅ Downloaded {len(records)}/{TOTAL_EXPECTED} laps ({success_rate:.0f}%)")

        if len(records) < TOTAL_EXPECTED * 0.7:
            print("  ⚠️  Less than 70% success — check your internet connection")
            print("     Re-running the script will pick up where FastF1 left off (cached sessions are free)")

        # Print per-circuit summary
        from collections import defaultdict
        by_circuit = defaultdict(list)
        for r in records:
            by_circuit[f"{r['gp']} {r['year']}"].append(r["driver"])
        print("\n  Per-circuit breakdown:")
        for circuit, drivers in sorted(by_circuit.items()):
            print(f"    {circuit:<40} {', '.join(sorted(drivers))}")

    except Exception as e:
        print(f"  ⚠️  Could not download training data: {e}")
        print("  → You can still run the demo. Training requires FastF1 data.")

    print("\n" + "=" * 60)
    print("  Done! Next steps:")
    print("  1. python scripts/train.py          (trains TCN model)")
    print("  2. python backend/api/main.py       (starts API server)")
    print("  3. cd frontend && npm run dev")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()