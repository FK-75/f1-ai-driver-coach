import React, { useEffect, useRef, useState } from 'react'
import { CornerTable } from './CornerTable.jsx'

const API_BASE = 'http://localhost:8000'

const MISTAKE_META = {
  late_brake:      { icon: '⚠️', label: 'Late Brake',      color: '#ff6b35', bar: '#ff6b35' },
  early_throttle:  { icon: '🟠', label: 'Early Throttle',  color: '#ffa726', bar: '#ffa726' },
  late_throttle:   { icon: '🟢', label: 'Late Throttle',   color: '#66bb6a', bar: '#66bb6a' },
  oversteer:       { icon: '🔴', label: 'Oversteer',       color: '#ef5350', bar: '#ef5350' },
  understeer:      { icon: '🟡', label: 'Understeer',      color: '#ffee58', bar: '#ffee58' },
  missed_apex:     { icon: '📍', label: 'Missed Apex',     color: '#ab47bc', bar: '#ab47bc' },
}

function MistakeMeter({ name, prob }) {
  const meta = MISTAKE_META[name] || { icon: '◆', label: name, color: '#888', bar: '#888' }
  const pct = Math.round(prob * 100)
  return (
    <div style={{ marginBottom: 6 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 2 }}>
        <span style={{ fontSize: 10, color: pct > 50 ? meta.color : '#555', fontWeight: pct > 50 ? 600 : 400 }}>
          {meta.icon} {meta.label}
        </span>
        <span style={{ fontFamily: 'JetBrains Mono', fontSize: 9, color: pct > 50 ? meta.color : '#444' }}>
          {pct}%
        </span>
      </div>
      <div style={{ height: 2, background: 'rgba(255,255,255,0.06)', borderRadius: 2, overflow: 'hidden' }}>
        <div style={{
          height: '100%', width: `${pct}%`,
          background: pct > 50 ? meta.bar : 'rgba(255,255,255,0.15)',
          borderRadius: 2,
          transition: 'width 0.3s ease',
          boxShadow: pct > 70 ? `0 0 4px ${meta.bar}` : 'none',
        }} />
      </div>
    </div>
  )
}

function CueToast({ cue, index }) {
  const age = index
  const opacity = Math.max(0.2, 1 - age * 0.25)
  const meta = MISTAKE_META[cue.type] || { color: '#888' }
  return (
    <div style={{
      padding: '7px 10px',
      background: `rgba(${cue.severity === 'high' ? '225,6,0' : '60,60,60'}, 0.10)`,
      border: `1px solid ${meta.color}${age === 0 ? '99' : '22'}`,
      borderRadius: 5,
      marginBottom: 5,
      opacity,
      transition: 'all 0.3s ease',
    }}>
      <div style={{ fontSize: 11, color: age === 0 ? meta.color : '#777', lineHeight: 1.3 }}>
        {cue.message}
      </div>
      <div style={{ fontFamily: 'JetBrains Mono', fontSize: 8, color: '#444', marginTop: 3 }}>
        {cue.distance_m ? `${Math.round(cue.distance_m)}m` : ''}
      </div>
    </div>
  )
}

export function CoachPanel({ current, cues, summary }) {
  const [llmSummary, setLlmSummary] = useState(null)
  const [llmLoading, setLlmLoading] = useState(false)
  const [showCorners, setShowCorners] = useState(false)
  const llmFetchedRef = useRef(false)

  useEffect(() => {
    if (!summary || llmFetchedRef.current) return
    llmFetchedRef.current = true
    setLlmLoading(true)
    const run = async () => {
      try {
        const [cornerResp, sectorResp] = await Promise.all([
          fetch(`${API_BASE}/corner-report`).then(r => r.json()).catch(() => ({ corners: [] })),
          fetch(`${API_BASE}/sector-report`).then(r => r.json()).catch(() => ({ sectors: [] })),
        ])
        const corners = cornerResp.corners || []
        const sectors = sectorResp.sectors || []
        const llmResp = await fetch(`${API_BASE}/llm-summary`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            sector_report: sectors, corner_report: corners,
            lap_summary: summary, driver: 'HAM', reference_driver: 'VER', track: 'Silverstone',
          }),
        })
        const data = await llmResp.json()
        setLlmSummary(data.summary || 'Lap complete — check sector breakdown.')
      } catch {
        setLlmSummary('Lap complete — check sector breakdown.')
      } finally {
        setLlmLoading(false)
      }
    }
    run()
  }, [summary])

  useEffect(() => {
    if (!summary) {
      llmFetchedRef.current = false
      setLlmSummary(null)
      setLlmLoading(false)
      setShowCorners(false)
    }
  }, [summary])

  const mistakeProbs = current?.mistake_probs || {}
  const delta = current?.delta_time_s ?? 0
  const deltaPos = delta > 0

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', gap: 8, overflow: 'hidden' }}>

      {/* Delta badge — compact */}
      <div style={{
        background: deltaPos ? 'rgba(255,68,68,0.08)' : 'rgba(0,230,118,0.08)',
        border: `1px solid ${deltaPos ? 'rgba(255,68,68,0.25)' : 'rgba(0,230,118,0.25)'}`,
        borderRadius: 6, padding: '8px 12px', textAlign: 'center', flexShrink: 0,
      }}>
        <div style={{
          fontFamily: 'JetBrains Mono', fontSize: 24, fontWeight: 700,
          color: deltaPos ? '#ff4444' : '#00e676', letterSpacing: '-0.02em',
        }}>
          {delta > 0 ? '+' : ''}{delta.toFixed(3)}s
        </div>
        <div style={{ fontSize: 9, color: '#444', marginTop: 1, textTransform: 'uppercase', letterSpacing: '0.1em' }}>
          Lap Time Delta
        </div>
      </div>

      {/* Mistake meters — compact, no scroll */}
      <div style={{
        background: 'rgba(255,255,255,0.02)',
        border: '1px solid rgba(255,255,255,0.06)',
        borderRadius: 6, padding: '8px 10px', flexShrink: 0,
      }}>
        <div style={{
          fontFamily: 'JetBrains Mono', fontSize: 9, color: '#444',
          letterSpacing: '0.12em', textTransform: 'uppercase', marginBottom: 8,
        }}>
          Mistake Detection
        </div>
        {Object.entries(mistakeProbs).length > 0
          ? Object.entries(mistakeProbs).map(([name, prob]) => (
              <MistakeMeter key={name} name={name} prob={prob} />
            ))
          : <div style={{ color: '#2a2a2a', fontSize: 10, textAlign: 'center', padding: '4px 0' }}>Waiting...</div>
        }
      </div>

      {/* Coaching cues — flex: 1 so it takes remaining space */}
      <div style={{
        background: 'rgba(255,255,255,0.02)',
        border: '1px solid rgba(255,255,255,0.06)',
        borderRadius: 6, padding: '8px 10px',
        flex: 1, minHeight: 0, overflowY: 'auto',
      }}>
        <div style={{
          fontFamily: 'JetBrains Mono', fontSize: 9, color: '#444',
          letterSpacing: '0.12em', textTransform: 'uppercase', marginBottom: 8,
          display: 'flex', justifyContent: 'space-between',
        }}>
          <span>Coaching Cues</span>
          {cues.length > 0 && <span style={{ color: '#333' }}>{cues.length}</span>}
        </div>
        {cues.length === 0
          ? <div style={{ color: '#2a2a2a', fontSize: 10, textAlign: 'center', padding: '12px 0', fontStyle: 'italic' }}>
              No cues yet
            </div>
          : cues.map((cue, i) => (
              <CueToast key={`${cue.type}-${cue.distance_m}-${i}`} cue={cue} index={i} />
            ))
        }
      </div>

      {/* Post-lap section — only shown when complete, scrollable */}
      {summary && (
        <div style={{
          borderTop: '1px solid rgba(255,255,255,0.06)',
          paddingTop: 8, flexShrink: 0, overflowY: 'auto', maxHeight: '45%',
        }}>
          {/* Lap times */}
          <div style={{
            background: 'rgba(225,6,0,0.05)',
            border: '1px solid rgba(225,6,0,0.25)',
            borderRadius: 6, padding: '8px 10px', marginBottom: 6,
          }}>
            <div style={{
              fontFamily: 'JetBrains Mono', fontSize: 9, color: '#e10600',
              letterSpacing: '0.12em', textTransform: 'uppercase', marginBottom: 6,
            }}>
              Lap Complete
            </div>
            <div style={{ fontSize: 11, color: '#aaa', lineHeight: 1.5 }}>
              <div>Ref: <span style={{ fontFamily: 'JetBrains Mono', color: '#4a9eff' }}>
                {summary.reference_lap_time_s?.toFixed(3)}s
              </span></div>
              <div>You: <span style={{ fontFamily: 'JetBrains Mono', color: '#e10600' }}>
                {summary.lap_time_s?.toFixed(3)}s
              </span></div>
            </div>
          </div>

          {/* AI Debrief */}
          <div style={{
            background: 'rgba(74,158,255,0.04)',
            border: '1px solid rgba(74,158,255,0.18)',
            borderRadius: 6, padding: '8px 10px', marginBottom: 6,
          }}>
            <div style={{
              fontFamily: 'JetBrains Mono', fontSize: 9, color: '#4a9eff',
              letterSpacing: '0.12em', textTransform: 'uppercase', marginBottom: 6,
            }}>
              AI Debrief {llmLoading && <span style={{ color: '#333', fontWeight: 400 }}>...</span>}
            </div>
            {llmLoading
              ? <div style={{ color: '#333', fontSize: 10, fontStyle: 'italic' }}>Analysing...</div>
              : llmSummary
                ? <div style={{
                    fontSize: 10, color: '#aaa', lineHeight: 1.6,
                    overflowY: 'auto', maxHeight: 130,
                  }}>{llmSummary}</div>
                : null
            }
          </div>

          {/* Corner breakdown toggle */}
          <button
            onClick={() => setShowCorners(v => !v)}
            style={{
              width: '100%', padding: '6px 0', borderRadius: 5, cursor: 'pointer',
              fontFamily: 'JetBrains Mono', fontSize: 9, letterSpacing: '0.08em',
              background: 'transparent', border: '1px solid rgba(255,255,255,0.08)',
              color: '#444', textTransform: 'uppercase',
            }}
          >
            {showCorners ? '▲ Hide Corners' : '▼ Corner Breakdown'}
          </button>
          {showCorners && <CornerTable visible={showCorners} />}
        </div>
      )}
    </div>
  )
}
