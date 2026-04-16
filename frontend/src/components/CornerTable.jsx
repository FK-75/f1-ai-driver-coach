import React, { useEffect, useState } from 'react'

const API_BASE = 'http://localhost:8000'

const MISTAKE_META = {
  late_brake:     { icon: '⚠️', label: 'Late Brake',     color: '#ff6b35' },
  early_throttle: { icon: '🟠', label: 'Early Throttle', color: '#ffa726' },
  late_throttle:  { icon: '🟢', label: 'Late Throttle',  color: '#66bb6a' },
  oversteer:      { icon: '🔴', label: 'Oversteer',      color: '#ef5350' },
  understeer:     { icon: '🟡', label: 'Understeer',     color: '#ffee58' },
  missed_apex:    { icon: '📍', label: 'Missed Apex',    color: '#ab47bc' },
}

function DeltaBar({ value }) {
  const capped = Math.max(-0.5, Math.min(0.5, value ?? 0))
  const pct = Math.abs(capped) / 0.5 * 100
  return (
    <div style={{
      display: 'inline-block', verticalAlign: 'middle',
      width: 22, height: 5, background: 'rgba(255,255,255,0.06)',
      borderRadius: 2, overflow: 'hidden', marginRight: 4, flexShrink: 0,
    }}>
      <div style={{
        height: '100%', width: `${pct}%`,
        background: capped > 0 ? '#ef5350' : '#00e676',
        borderRadius: 2,
      }} />
    </div>
  )
}

export function CornerTable({ visible }) {
  const [corners, setCorners] = useState([])
  const [loading, setLoading] = useState(false)
  const [sortKey, setSortKey] = useState('time_delta_s')
  const [sortDir, setSortDir] = useState('desc')

  useEffect(() => {
    if (!visible) return
    setLoading(true)
    fetch(`${API_BASE}/corner-report`)
      .then(r => r.json())
      .then(d => { setCorners(d.corners || []); setLoading(false) })
      .catch(() => setLoading(false))
  }, [visible])

  if (!visible) return null

  const sorted = [...corners].sort((a, b) => {
    const av = a[sortKey] ?? 0
    const bv = b[sortKey] ?? 0
    return sortDir === 'desc' ? bv - av : av - bv
  })

  function toggleSort(key) {
    if (sortKey === key) setSortDir(d => d === 'desc' ? 'asc' : 'desc')
    else { setSortKey(key); setSortDir('desc') }
  }

  const H = ({ k, label, w }) => (
    <th onClick={() => toggleSort(k)} style={{
      width: w, fontFamily: 'JetBrains Mono', fontSize: 8,
      color: sortKey === k ? '#e10600' : '#444',
      letterSpacing: '0.08em', textTransform: 'uppercase',
      cursor: 'pointer', userSelect: 'none',
      padding: '0 4px 8px 0', textAlign: 'left', whiteSpace: 'nowrap',
    }}>
      {label}{sortKey === k ? (sortDir === 'desc' ? ' ↓' : ' ↑') : ''}
    </th>
  )

  const cell = {
    padding: '4px 4px 4px 0',
    borderBottom: '1px solid rgba(255,255,255,0.04)',
    verticalAlign: 'middle', overflow: 'hidden',
  }

  return (
    <div style={{
      background: 'rgba(255,255,255,0.02)',
      border: '1px solid rgba(255,255,255,0.06)',
      borderRadius: 8, padding: '10px 12px', marginTop: 6,
    }}>
      <div style={{
        fontFamily: 'JetBrains Mono', fontSize: 9, color: '#555',
        letterSpacing: '0.12em', textTransform: 'uppercase', marginBottom: 10,
        display: 'flex', justifyContent: 'space-between',
      }}>
        <span>Corner Breakdown</span>
        <span style={{ color: '#333' }}>{corners.length} corners</span>
      </div>

      {loading ? (
        <div style={{ color: '#333', fontSize: 10, textAlign: 'center', padding: '12px 0' }}>Loading...</div>
      ) : corners.length === 0 ? (
        <div style={{ color: '#2a2a2a', fontSize: 10, textAlign: 'center', padding: '12px 0' }}>
          Complete a lap to see corner breakdown
        </div>
      ) : (
        <table style={{ width: '100%', borderCollapse: 'collapse', tableLayout: 'fixed' }}>
          <colgroup>
            <col style={{ width: '26px' }} />
            <col style={{ width: '62px' }} />
            <col style={{ width: '36px' }} />
            <col style={{ width: '32px' }} />
            <col style={{ width: '80px' }} />
            <col style={{ width: '22px' }} />
          </colgroup>
          <thead>
            <tr>
              <H k="corner" label="#" />
              <H k="name" label="Name" />
              <H k="drv_apex_speed_kmh" label="Apex" />
              <H k="speed_delta_kmh" label="Spd" />
              <H k="time_delta_s" label="Time Δ" />
              <H k="dominant_mistake" label="!" />
            </tr>
          </thead>
          <tbody>
            {sorted.map(c => {
              const meta = c.dominant_mistake ? MISTAKE_META[c.dominant_mistake] : null
              const isLoss = c.time_delta_s > 0.1
              const dPos = c.time_delta_s > 0
              const sPos = c.speed_delta_kmh >= 0
              return (
                <tr key={c.corner} style={{ background: isLoss ? 'rgba(225,6,0,0.04)' : 'transparent' }}>
                  <td style={{ ...cell, fontFamily: 'JetBrains Mono', fontSize: 9, color: '#555' }}>
                    C{c.corner}
                  </td>
                  <td style={{ ...cell, fontSize: 9, color: '#666' }}>
                    <span style={{ display: 'block', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {c.name || '—'}
                    </span>
                  </td>
                  <td style={{ ...cell, fontFamily: 'JetBrains Mono', fontSize: 9, color: '#777' }}>
                    {c.drv_apex_speed_kmh}
                  </td>
                  <td style={{ ...cell, fontFamily: 'JetBrains Mono', fontSize: 9, color: sPos ? '#00e676' : '#ff4444' }}>
                    {sPos ? '+' : ''}{c.speed_delta_kmh}
                  </td>
                  <td style={{ ...cell }}>
                    <div style={{ display: 'flex', alignItems: 'center' }}>
                      <DeltaBar value={c.time_delta_s} />
                      <span style={{
                        fontFamily: 'JetBrains Mono', fontSize: 9, fontWeight: 600,
                        color: dPos ? '#ff4444' : '#00e676',
                      }}>
                        {dPos ? '+' : ''}{c.time_delta_s.toFixed(3)}s
                      </span>
                    </div>
                  </td>
                  <td style={{ ...cell, textAlign: 'center' }}>
                    {meta
                      ? <span style={{ fontSize: 10 }} title={meta.label}>{meta.icon}</span>
                      : <span style={{ color: '#2a2a2a', fontSize: 9 }}>—</span>
                    }
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      )}
    </div>
  )
}
