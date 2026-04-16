import React, { useState, useEffect } from 'react'
import { useWebSocket } from './hooks/useWebSocket.js'
import { LiveChart } from './components/LiveChart.jsx'
import { TrackMap } from './components/TrackMap.jsx'
import { CoachPanel } from './components/CoachPanel.jsx'
import { LapSelector } from './components/LapSelector.jsx'
import { CompareLaps } from './components/CompareLaps.jsx'

const API_BASE = 'http://localhost:8000'

// ── Telemetry readout tile ────────────────────────────────────────────────────
function TelemetryTile({ label, value, unit, color = '#e8eaed', mono = true }) {
  return (
    <div style={{
      background: 'rgba(255,255,255,0.025)',
      border: '1px solid rgba(255,255,255,0.07)',
      borderRadius: 6,
      padding: '10px 14px',
      minWidth: 80,
      flex: 1,
    }}>
      <div style={{
        fontFamily: 'JetBrains Mono', fontSize: 9, color: '#444',
        letterSpacing: '0.12em', textTransform: 'uppercase', marginBottom: 4,
      }}>
        {label}
      </div>
      <div style={{
        fontFamily: mono ? 'JetBrains Mono' : 'Exo 2',
        fontSize: 22, fontWeight: 700, color,
        lineHeight: 1, letterSpacing: '-0.02em',
      }}>
        {value ?? '—'}
        {unit && <span style={{ fontSize: 11, color: '#444', marginLeft: 3, fontWeight: 400 }}>{unit}</span>}
      </div>
    </div>
  )
}

// ── Status pill ───────────────────────────────────────────────────────────────
function StatusPill({ status }) {
  const map = {
    idle:       { label: 'IDLE',       color: '#444', bg: 'rgba(100,100,100,0.1)' },
    connecting: { label: 'CONNECTING', color: '#ffa726', bg: 'rgba(255,167,38,0.1)' },
    running:    { label: '● LIVE',     color: '#00e676', bg: 'rgba(0,230,118,0.1)' },
    complete:   { label: 'COMPLETE',   color: '#4a9eff', bg: 'rgba(74,158,255,0.1)' },
    error:      { label: 'ERROR',      color: '#ef5350', bg: 'rgba(239,83,80,0.1)' },
  }
  const s = map[status] || map.idle
  return (
    <div style={{
      padding: '4px 12px', borderRadius: 20,
      background: s.bg, color: s.color,
      fontFamily: 'JetBrains Mono', fontSize: 10, letterSpacing: '0.1em',
      border: `1px solid ${s.color}44`,
    }}>
      {s.label}
    </div>
  )
}

// ── Progress bar ─────────────────────────────────────────────────────────────
function LapProgress({ progress = 0, lapTimeS, refLapTimeS }) {
  return (
    <div style={{ flex: 1, padding: '0 20px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 5 }}>
        <span style={{ fontFamily: 'JetBrains Mono', fontSize: 9, color: '#444' }}>
          LAP PROGRESS
        </span>
        <span style={{ fontFamily: 'JetBrains Mono', fontSize: 9, color: '#666' }}>
          {(progress * 100).toFixed(1)}%
        </span>
      </div>
      <div style={{
        height: 4, background: 'rgba(255,255,255,0.05)', borderRadius: 2, overflow: 'hidden',
      }}>
        <div style={{
          height: '100%',
          width: `${progress * 100}%`,
          background: 'linear-gradient(to right, #e10600, #ff6b35)',
          borderRadius: 2,
          transition: 'width 0.2s linear',
          boxShadow: '0 0 8px rgba(225,6,0,0.4)',
        }} />
      </div>
      {lapTimeS && refLapTimeS && (
        <div style={{
          display: 'flex', justifyContent: 'space-between', marginTop: 4,
        }}>
          <span style={{ fontFamily: 'JetBrains Mono', fontSize: 9, color: '#555' }}>
            DRV {lapTimeS}s
          </span>
          <span style={{ fontFamily: 'JetBrains Mono', fontSize: 9, color: '#4a9eff' }}>
            REF {refLapTimeS}s
          </span>
        </div>
      )}
    </div>
  )
}

// ── Gear display ──────────────────────────────────────────────────────────────
function GearDisplay({ gear, speed }) {
  return (
    <div style={{
      display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
      background: 'rgba(225,6,0,0.06)', border: '1px solid rgba(225,6,0,0.2)',
      borderRadius: 8, padding: '8px 20px', minWidth: 80,
    }}>
      <div style={{
        fontFamily: 'JetBrains Mono', fontSize: 42, fontWeight: 900,
        color: '#e10600', lineHeight: 1, letterSpacing: '-0.04em',
      }}>
        {gear || '—'}
      </div>
      <div style={{ fontFamily: 'JetBrains Mono', fontSize: 9, color: '#444', letterSpacing: '0.1em' }}>
        GEAR
      </div>
    </div>
  )
}


// ── Sector delta badges ───────────────────────────────────────────────────────
function SectorBadge({ label, delta }) {
  if (delta === null) return (
    <div style={{
      padding: '3px 10px', borderRadius: 4,
      background: 'rgba(255,255,255,0.03)',
      border: '1px solid rgba(255,255,255,0.07)',
      fontFamily: 'JetBrains Mono', fontSize: 9, color: '#333',
    }}>
      {label} —
    </div>
  )
  const pos = delta > 0
  return (
    <div style={{
      padding: '3px 10px', borderRadius: 4,
      background: pos ? 'rgba(255,68,68,0.08)' : 'rgba(0,230,118,0.08)',
      border: `1px solid ${pos ? 'rgba(255,68,68,0.3)' : 'rgba(0,230,118,0.3)'}`,
      fontFamily: 'JetBrains Mono', fontSize: 9,
      color: pos ? '#ff4444' : '#00e676',
    }}>
      {label} {delta > 0 ? '+' : ''}{delta.toFixed(3)}s
    </div>
  )
}


// ── Personal best banner ──────────────────────────────────────────────────────
function PbBanner({ pbDelta, newPb }) {
  if (pbDelta === null) return null
  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: 8,
      padding: '3px 12px', borderRadius: 4,
      background: newPb ? 'rgba(255,215,0,0.12)' : 'rgba(255,255,255,0.03)',
      border: `1px solid ${newPb ? 'rgba(255,215,0,0.5)' : 'rgba(255,255,255,0.08)'}`,
      transition: 'all 0.5s ease',
    }}>
      <span style={{
        fontFamily: 'JetBrains Mono', fontSize: 9,
        color: newPb ? '#ffd700' : '#444',
        letterSpacing: '0.08em',
      }}>
        {newPb ? '🏆 NEW PB' : 'PB'}
      </span>
      <span style={{
        fontFamily: 'JetBrains Mono', fontSize: 10, fontWeight: 700,
        color: newPb ? '#ffd700' : '#555',
      }}>
        {pbDelta > 0 ? '+' : ''}{pbDelta.toFixed(3)}s
      </span>
    </div>
  )
}

// ── Main App ──────────────────────────────────────────────────────────────────
export default function App() {
  const { status, frames, current, summary, cues, sectorDeltas, pbDelta, newPb, fixtureInfo, lapKey, start, stop } = useWebSocket()
  const [replaySpeed, setReplaySpeed] = useState(1.5)
  const [backendOnline, setBackendOnline] = useState(null)
  const [selectedLapId, setSelectedLapId] = useState(-1)
  const [compareLapId, setCompareLapId] = useState(-1)
  const [viewMode, setViewMode] = useState('live') // 'live' | 'compare'

  // Check backend health
  useEffect(() => {
    fetch(`${API_BASE}/health`)
      .then(r => r.json())
      .then(d => setBackendOnline(d.status === 'ok'))
      .catch(() => setBackendOnline(false))
  }, [])

  const handleStart = () => start(replaySpeed, selectedLapId)
  const handleStop = () => stop()

  const speed    = current?.speed ?? 0
  const throttle = current?.throttle ?? 0
  const brake    = current?.brake ?? 0
  const gear     = current?.gear ?? 0
  const latG     = current?.lateral_g ?? 0
  const delta    = current?.delta_time_s ?? 0
  const progress = current?.lap_progress ?? 0

  const isRunning = status === 'running'
  const canStart  = status === 'idle' || status === 'complete' || status === 'error'

  return (
    <div style={{
      width: '100vw', height: '100vh', overflow: 'hidden',
      background: '#080b10',
      display: 'flex', flexDirection: 'column',
    }}>

      {/* ── Header ── */}
      <div style={{
        padding: '12px 20px',
        borderBottom: '1px solid rgba(255,255,255,0.06)',
        display: 'flex', alignItems: 'center', gap: 16,
        background: 'rgba(0,0,0,0.4)',
        flexShrink: 0,
      }}>
        {/* Logo */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <div style={{
            width: 28, height: 28, background: '#e10600', borderRadius: 4,
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontWeight: 900, fontSize: 14, fontFamily: 'Exo 2',
          }}>
            F1
          </div>
          <div>
            <div style={{ fontSize: 13, fontWeight: 700, letterSpacing: '0.05em', lineHeight: 1 }}>
              AI DRIVER COACH
            </div>
            <div style={{ fontFamily: 'JetBrains Mono', fontSize: 9, color: '#444', letterSpacing: '0.1em' }}>
              {fixtureInfo
                ? `${fixtureInfo.driver} vs ${fixtureInfo.reference_driver} — ${fixtureInfo.track_name} ${fixtureInfo.year}`
                : 'TELEMETRY ANALYSIS SYSTEM'}
            </div>
          </div>
        </div>

        {/* Status */}
        <StatusPill status={status} />

        {/* Sector deltas */}
        {(isRunning || status === 'complete') && (
          <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
            <SectorBadge label="S1" delta={sectorDeltas[0]} />
            <SectorBadge label="S2" delta={sectorDeltas[1]} />
            <SectorBadge label="S3" delta={sectorDeltas[2]} />
          </div>
        )}

        {/* Personal best */}
        <PbBanner pbDelta={pbDelta} newPb={newPb} />

        {/* Backend health */}
        <div style={{
          fontFamily: 'JetBrains Mono', fontSize: 9,
          color: backendOnline === null ? '#444' : backendOnline ? '#00e676' : '#ef5350',
        }}>
          {backendOnline === null ? '...' : backendOnline ? '◆ BACKEND' : '◆ OFFLINE'}
        </div>

        {/* Spacer */}
        <div style={{ flex: 1 }} />

        {/* Lap progress */}
        {isRunning && (
          <LapProgress
            progress={progress}
            lapTimeS={fixtureInfo?.lap_time_s}
            refLapTimeS={fixtureInfo?.reference_lap_time_s}
          />
        )}

        {/* Lap selector — only in live mode (compare has its own selectors inline) */}
        {canStart && viewMode === 'live' && (
          <LapSelector onSelect={setSelectedLapId} disabled={!canStart} />
        )}

        {/* Speed selector */}
        {canStart && (
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <span style={{ fontFamily: 'JetBrains Mono', fontSize: 9, color: '#444' }}>SPEED</span>
            {[1, 1.5, 2, 3].map(s => (
              <button key={s} onClick={() => setReplaySpeed(s)} style={{
                padding: '3px 8px', borderRadius: 4, cursor: 'pointer', fontSize: 10,
                fontFamily: 'JetBrains Mono',
                background: replaySpeed === s ? 'rgba(225,6,0,0.2)' : 'transparent',
                border: `1px solid ${replaySpeed === s ? '#e10600' : 'rgba(255,255,255,0.1)'}`,
                color: replaySpeed === s ? '#e10600' : '#555',
              }}>
                {s}×
              </button>
            ))}
          </div>
        )}

        {/* Start / Stop */}
        <button
          onClick={canStart ? handleStart : handleStop}
          disabled={status === 'connecting'}
          style={{
            padding: '8px 20px', borderRadius: 6, cursor: 'pointer',
            fontFamily: 'JetBrains Mono', fontSize: 11, fontWeight: 700,
            letterSpacing: '0.08em',
            background: canStart ? '#e10600' : 'rgba(100,100,100,0.2)',
            border: 'none', color: '#fff',
            transition: 'all 0.2s ease',
            opacity: status === 'connecting' ? 0.5 : 1,
          }}
        >
          {canStart ? (selectedLapId >= 0 ? '▶ START LAP' : '▶ START DEMO') : '■ STOP'}
        </button>
      </div>

      {/* ── Telemetry strip ── */}
      <div style={{
        padding: '10px 20px',
        borderBottom: '1px solid rgba(255,255,255,0.04)',
        display: 'flex', gap: 8, alignItems: 'stretch',
        background: 'rgba(0,0,0,0.2)',
        flexShrink: 0,
      }}>
        <GearDisplay gear={gear} speed={speed} />

        <TelemetryTile label="Speed" value={Math.round(speed)} unit="km/h"
          color={speed > 280 ? '#e10600' : speed > 200 ? '#ffa726' : '#e8eaed'} />
        <TelemetryTile label="Throttle" value={Math.round(throttle)} unit="%"
          color={throttle > 80 ? '#00e676' : '#e8eaed'} />
        <TelemetryTile label="Brake"
          value={brake > 0 ? 'ON' : 'OFF'} unit=""
          color={brake > 0 ? '#ef5350' : '#333'} mono />
        <TelemetryTile label="Lateral G" value={latG.toFixed(2)} unit="G"
          color={latG > 3 ? '#ffa726' : '#e8eaed'} />
        <TelemetryTile label="Δ Time"
          value={`${delta >= 0 ? '+' : ''}${delta.toFixed(3)}`} unit="s"
          color={delta > 0 ? '#ef5350' : '#00e676'} />
        <TelemetryTile label="Ref Speed" value={Math.round(current?.ref_speed ?? 0)} unit="km/h"
          color="#4a9eff" />
        <TelemetryTile label="Frames" value={frames.length} unit="" color="#444" />
      </div>

      {/* ── Main content ── */}
      <div style={{
        flex: 1, minHeight: 0,
        display: 'grid',
        gridTemplateColumns: '260px 1fr 300px',
        gap: 0,
      }}>

        {/* Left: Track map + info */}
        <div style={{
          borderRight: '1px solid rgba(255,255,255,0.05)',
          padding: '12px',
          display: 'flex', flexDirection: 'column', gap: 8,
          overflow: 'hidden',
        }}>
          <div style={{
            fontFamily: 'JetBrains Mono', fontSize: 9, color: '#444',
            letterSpacing: '0.15em', textTransform: 'uppercase',
          }}>
            Track Map
          </div>

          <div style={{ display: 'flex', justifyContent: 'center' }}>
            <TrackMap current={current} frames={frames} lapKey={lapKey} fixtureInfo={fixtureInfo} />
          </div>

          {/* Corner info */}
          {current && (
            <div style={{
              background: 'rgba(255,255,255,0.02)',
              border: '1px solid rgba(255,255,255,0.05)',
              borderRadius: 6, padding: '10px 12px',
            }}>
              <div style={{ fontFamily: 'JetBrains Mono', fontSize: 9, color: '#444', marginBottom: 6, letterSpacing: '0.1em' }}>
                POSITION
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6 }}>
                {[
                  ['Distance', `${Math.round(current.distance_m)}m`],
                  ['Sim Time', `${current.sim_time_s?.toFixed(1)}s`],
                  ['Progress', `${(progress * 100).toFixed(1)}%`],
                  ['Ref Throttle', `${Math.round(current.ref_throttle ?? 0)}%`],
                ].map(([k, v]) => (
                  <div key={k}>
                    <div style={{ fontFamily: 'JetBrains Mono', fontSize: 8, color: '#333' }}>{k}</div>
                    <div style={{ fontFamily: 'JetBrains Mono', fontSize: 12, color: '#888' }}>{v}</div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* About panel when idle */}
          {!current && (
            <div style={{
              background: 'rgba(225,6,0,0.04)',
              border: '1px solid rgba(225,6,0,0.15)',
              borderRadius: 6, padding: '14px',
            }}>
              <div style={{ fontFamily: 'JetBrains Mono', fontSize: 9, color: '#e10600', marginBottom: 8, letterSpacing: '0.1em' }}>
                HOW IT WORKS
              </div>
              <div style={{ fontSize: 11, color: '#666', lineHeight: 1.7 }}>
                The AI ingests real F1 telemetry (speed, throttle, brake, gear, lateral G) and compares your lap against a reference using a Temporal Convolutional Network trained on Verstappen & Hamilton data via FastF1.
              </div>
              <div style={{ fontSize: 11, color: '#444', marginTop: 8, lineHeight: 1.6 }}>
                Mistakes are detected automatically from telemetry delta — no human annotation required.
              </div>
            </div>
          )}
        </div>

        {/* Centre: Live charts */}
        <div style={{
          padding: '12px 16px',
          overflowY: 'auto',
          overflowX: 'hidden',
          display: 'flex', flexDirection: 'column',
        }}>
          {/* Mode toggle */}
          <div style={{ display: 'flex', gap: 0, marginBottom: 12, flexShrink: 0 }}>
            {['live', 'compare'].map(mode => (
              <button key={mode} onClick={() => setViewMode(mode)} style={{
                padding: '5px 16px', cursor: 'pointer',
                fontFamily: 'JetBrains Mono', fontSize: 9, letterSpacing: '0.08em',
                textTransform: 'uppercase',
                background: viewMode === mode ? 'rgba(225,6,0,0.15)' : 'transparent',
                border: `1px solid ${viewMode === mode ? '#e10600' : 'rgba(255,255,255,0.08)'}`,
                borderRadius: mode === 'live' ? '4px 0 0 4px' : '0 4px 4px 0',
                color: viewMode === mode ? '#e10600' : '#444',
                marginRight: mode === 'live' ? -1 : 0,
              }}>
                {mode === 'live' ? '▶ Live Coach' : '⚡ Compare Laps'}
              </button>
            ))}
            {viewMode === 'compare' && (
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginLeft: 16 }}>
                <span style={{ fontFamily: 'JetBrains Mono', fontSize: 9, color: '#555' }}>A</span>
                <LapSelector onSelect={setSelectedLapId} disabled={false} />
                <span style={{ fontFamily: 'JetBrains Mono', fontSize: 9, color: '#555' }}>B</span>
                <LapSelector onSelect={setCompareLapId} disabled={false} />
              </div>
            )}
            {viewMode === 'live' && (
              <span style={{ fontFamily: 'JetBrains Mono', fontSize: 9, color: '#333', marginLeft: 16, alignSelf: 'center' }}>
                Telemetry Traces — Last {frames.length} samples
              </span>
            )}
          </div>

          {viewMode === 'compare' ? (
            <CompareLaps lapIdA={selectedLapId} lapIdB={compareLapId} />
          ) : frames.length > 5 ? (
            <LiveChart frames={frames} current={current} />
          ) : (
            <div style={{
              flex: 1, display: 'flex', flexDirection: 'column',
              alignItems: 'center', justifyContent: 'center',
              color: '#222', gap: 16,
            }}>
              <div style={{ fontSize: 48 }}>🏎️</div>
              <div style={{ fontFamily: 'JetBrains Mono', fontSize: 13, letterSpacing: '0.05em' }}>
                {backendOnline === false
                  ? 'Backend offline — start the FastAPI server'
                  : (selectedLapId >= 0 ? 'Press START LAP to replay selected lap' : 'Press START DEMO to begin replay')}
              </div>
              <div style={{ fontFamily: 'JetBrains Mono', fontSize: 10, color: '#1a1a1a', textAlign: 'center', lineHeight: 1.8 }}>
                {backendOnline === false ? (
                  <>python backend/api/main.py</>
                ) : (
                  <>Verstappen 2023 Silverstone Q3 · HAM vs VER<br />
                  TCN · FastF1 · ONNX · 8ms inference</>
                )}
              </div>
            </div>
          )}
        </div>

        {/* Right: Coach panel */}
        <div style={{
          borderLeft: '1px solid rgba(255,255,255,0.05)',
          padding: '16px',
          display: 'flex', flexDirection: 'column',
          overflow: 'hidden',
        }}>
          <div style={{
            fontFamily: 'JetBrains Mono', fontSize: 9, color: '#444',
            letterSpacing: '0.15em', textTransform: 'uppercase',
            marginBottom: 12,
          }}>
            AI Coach
          </div>

          <CoachPanel current={current} cues={cues} summary={summary} />
        </div>
      </div>

      {/* ── Footer ── */}
      <div style={{
        padding: '6px 20px',
        borderTop: '1px solid rgba(255,255,255,0.04)',
        display: 'flex', alignItems: 'center', gap: 20,
        background: 'rgba(0,0,0,0.4)',
        flexShrink: 0,
      }}>
        {[
          ['Model', 'TCN · 64ch · 4 blocks'],
          ['Inference', '~8ms CPU'],
          ['Data', 'FastF1 · Real F1 Telemetry'],
          ['Alignment', 'Distance-axis DTW'],
          ['Labels', 'Auto-generated from Δ telemetry'],
        ].map(([k, v]) => (
          <div key={k} style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
            <span style={{ fontFamily: 'JetBrains Mono', fontSize: 8, color: '#333', textTransform: 'uppercase', letterSpacing: '0.1em' }}>{k}</span>
            <span style={{ fontFamily: 'JetBrains Mono', fontSize: 8, color: '#555' }}>{v}</span>
          </div>
        ))}
        <div style={{ flex: 1 }} />
        <span style={{ fontFamily: 'JetBrains Mono', fontSize: 8, color: '#2a2a2a' }}>
          F1 AI DRIVER COACH · PORTFOLIO PROJECT
        </span>
      </div>
    </div>
  )
}
