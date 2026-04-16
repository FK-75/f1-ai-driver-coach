import React, { useEffect, useRef, useState, useMemo, useCallback } from 'react'

const API_BASE = 'http://localhost:8000'

function normaliseTrack(xArr, yArr, size = 280, padding = 30) {
  const minX = Math.min(...xArr), maxX = Math.max(...xArr)
  const minY = Math.min(...yArr), maxY = Math.max(...yArr)
  const rangeX = maxX - minX || 1
  const rangeY = maxY - minY || 1
  const scale = Math.min((size - padding * 2) / rangeX, (size - padding * 2) / rangeY)

  const normX = xArr.map(x => padding + (x - minX) * scale + (size - padding * 2 - rangeX * scale) / 2)
  const normY = yArr.map(y => size - padding - (y - minY) * scale + -(size - padding * 2 - rangeY * scale) / 2)

  return { normX, normY, scale, minX, minY }
}

function speedToColor(speed, minSpeed = 80, maxSpeed = 330) {
  const t = Math.max(0, Math.min(1, (speed - minSpeed) / (maxSpeed - minSpeed)))
  // Blue (slow) → Green → Yellow → Red (fast)
  if (t < 0.33) {
    const u = t / 0.33
    return `rgb(${Math.round(30 + u * 50)}, ${Math.round(100 + u * 100)}, ${Math.round(220 - u * 100)})`
  } else if (t < 0.66) {
    const u = (t - 0.33) / 0.33
    return `rgb(${Math.round(80 + u * 175)}, ${Math.round(200 + u * 30)}, ${Math.round(120 - u * 100)})`
  } else {
    const u = (t - 0.66) / 0.34
    return `rgb(${Math.round(255)}, ${Math.round(230 - u * 200)}, ${Math.round(20 - u * 20)})`
  }
}

export function TrackMap({ current, frames = [], lapKey = 0, fixtureInfo = null }) {
  const [mapMode, setMapMode] = useState('speed') // 'speed' | 'mistakes'

  const [trackData, setTrackData] = useState(null)
  const [loading, setLoading] = useState(true)
  const SIZE = 232

  useEffect(() => {
    setLoading(true)
    Promise.all([
      fetch(`${API_BASE}/reference`).then(r => r.json()).catch(() => null),
      fetch(`${API_BASE}/driver-track`).then(r => r.json()).catch(() => null),
    ]).then(([ref, drv]) => {
      setTrackData({ ref, drv })
      setLoading(false)
    })
  }, [lapKey])

  const { refPath, coloredSegments, norm } = useMemo(() => {
    if (!trackData?.ref) return { refPath: '', coloredSegments: [], norm: null }

    const { x, y, speed } = trackData.ref
    const n = normaliseTrack(x, y, SIZE)
    const { normX, normY } = n

    const refPath = normX.map((px, i) => `${i === 0 ? 'M' : 'L'}${px.toFixed(1)},${normY[i].toFixed(1)}`).join(' ')

    const segments = []
    for (let i = 0; i < normX.length - 1; i++) {
      segments.push({
        x1: normX[i], y1: normY[i],
        x2: normX[i + 1], y2: normY[i + 1],
        color: speedToColor(speed[i]),
      })
    }

    return { refPath, coloredSegments: segments, norm: n }
  }, [trackData])

  // Current car position
  const carPos = useMemo(() => {
    if (!current || !trackData?.ref || !norm) return null
    const { x, y } = trackData.ref
    const progress = current.lap_progress || 0
    const idx = Math.min(Math.floor(progress * x.length), x.length - 1)

    const { minX, minY, scale } = norm
    const padding = 30
    const size = SIZE
    const rangeX = Math.max(...x) - Math.min(...x)
    const rangeY = Math.max(...y) - Math.min(...y)

    const cx = padding + (x[idx] - minX) * scale + (size - padding * 2 - rangeX * scale) / 2
    const cy = size - padding - (y[idx] - minY) * scale + -(size - padding * 2 - rangeY * scale) / 2

    return { cx, cy }
  }, [current, trackData, norm])

  // Build mistake heatmap: for each track position, count how many frames had an active mistake
  const mistakeSegments = useMemo(() => {
    if (!trackData?.ref || !norm || frames.length === 0) return []
    const { x, y } = trackData.ref
    const n = x.length
    // Count mistake hits per position bucket
    const hits = new Float32Array(n).fill(0)
    const counts = new Float32Array(n).fill(0)
    for (const frame of frames) {
      const prog = frame.lap_progress ?? 0
      const idx = Math.min(Math.floor(prog * n), n - 1)
      const probs = frame.mistake_probs ?? {}
      const maxProb = Math.max(0, ...Object.values(probs))
      hits[idx] += maxProb
      counts[idx] += 1
    }
    const { normX, normY, minX, minY, scale } = norm
    const padding = 30
    const segments = []
    for (let i = 0; i < normX.length - 1; i++) {
      const density = counts[i] > 0 ? hits[i] / counts[i] : 0
      // Color: green (clean) → yellow → red (mistake-heavy)
      let color
      if (density < 0.3) {
        color = `rgba(0,230,118,${0.3 + density})`
      } else if (density < 0.6) {
        const u = (density - 0.3) / 0.3
        color = `rgb(${Math.round(255 * u)},${Math.round(230 - 130 * u)},${Math.round(50 - 50 * u)})`
      } else {
        color = `rgb(225,6,0)`
      }
      segments.push({ x1: normX[i], y1: normY[i], x2: normX[i+1], y2: normY[i+1], color })
    }
    return segments
  }, [frames, trackData, norm])

  if (loading) {
    return (
      <div style={{
        width: SIZE, height: SIZE, display: 'flex', alignItems: 'center', justifyContent: 'center',
        color: '#444', fontSize: 12, fontFamily: 'JetBrains Mono',
      }}>
        Loading track...
      </div>
    )
  }

  const { lap_progress = 0, speed = 0, gear = 0, delta_time_s = 0 } = current || {}

  return (
    <div style={{ position: 'relative', width: SIZE, height: SIZE }}>
      {/* Mode toggle */}
      <div style={{
        position: 'absolute', top: 4, right: 6, zIndex: 10,
        display: 'flex', gap: 4,
      }}>
        {['speed', 'mistakes'].map(mode => (
          <button key={mode} onClick={() => setMapMode(mode)} style={{
            padding: '2px 7px', borderRadius: 3, cursor: 'pointer',
            fontFamily: 'JetBrains Mono', fontSize: 8, letterSpacing: '0.05em',
            background: mapMode === mode ? 'rgba(225,6,0,0.2)' : 'transparent',
            border: `1px solid ${mapMode === mode ? '#e10600' : 'rgba(255,255,255,0.1)'}`,
            color: mapMode === mode ? '#e10600' : '#555',
          }}>
            {mode === 'speed' ? 'SPEED' : 'HEAT'}
          </button>
        ))}
      </div>
      <svg width={SIZE} height={SIZE} style={{ position: 'absolute', top: 0, left: 0 }}>
        {/* Background circuit shape */}
        <path d={refPath} fill="none" stroke="rgba(255,255,255,0.05)" strokeWidth={12}
          strokeLinecap="round" strokeLinejoin="round" />

        {/* Speed/mistake-colored segments */}
        {(mapMode === 'speed' ? coloredSegments : (mistakeSegments.length > 0 ? mistakeSegments : coloredSegments)).map((seg, i) => (
          <line key={i} x1={seg.x1} y1={seg.y1} x2={seg.x2} y2={seg.y2}
            stroke={seg.color} strokeWidth={3.5} strokeLinecap="round" />
        ))}

        {/* Car position */}
        {carPos && (
          <g>
            <circle cx={carPos.cx} cy={carPos.cy} r={10}
              fill="rgba(225,6,0,0.2)" stroke="#e10600" strokeWidth={2}>
              <animate attributeName="r" values="8;14;8" dur="1.2s" repeatCount="indefinite" />
              <animate attributeName="opacity" values="1;0.4;1" dur="1.2s" repeatCount="indefinite" />
            </circle>
            <circle cx={carPos.cx} cy={carPos.cy} r={5} fill="#e10600" />
          </g>
        )}

        {/* Sector markers */}
        {[0.33, 0.66].map((frac, i) => {
          if (!trackData?.ref?.x) return null
          const { x, y } = trackData.ref
          const idx = Math.floor(frac * x.length)
          if (!norm) return null
          const { minX, minY, scale } = norm
          const padding = 30
          const rangeX = Math.max(...x) - Math.min(...x)
          const rangeY = Math.max(...y) - Math.min(...y)
          const cx = padding + (x[idx] - minX) * scale + (SIZE - padding * 2 - rangeX * scale) / 2
          const cy = SIZE - padding - (y[idx] - minY) * scale + -(SIZE - padding * 2 - rangeY * scale) / 2

          return (
            <g key={i}>
              <circle cx={cx} cy={cy} r={4} fill="#ffd700" />
              <text x={cx + 6} y={cy + 4} fontSize={8} fill="#ffd700"
                fontFamily="JetBrains Mono">S{i + 2}</text>
            </g>
          )
        })}
      </svg>

      {/* Legend bar */}
      <div style={{
        position: 'absolute', bottom: 8, left: 8, right: 8,
        height: 4, borderRadius: 2,
        background: mapMode === 'speed'
          ? 'linear-gradient(to right, #1e64dc, #00e676, #ffd700, #e10600)'
          : 'linear-gradient(to right, #00e676, #ffd700, #e10600)',
        opacity: 0.7,
      }} />
      <div style={{
        position: 'absolute', bottom: 14, left: 8, right: 8,
        display: 'flex', justifyContent: 'space-between',
        fontFamily: 'JetBrains Mono', fontSize: 8, color: '#555',
      }}>
        <span>{mapMode === 'speed' ? '80' : 'clean'}</span>
        <span style={{ color: '#888' }}>{mapMode === 'speed' ? 'km/h' : 'mistakes'}</span>
        <span>{mapMode === 'speed' ? '330' : 'heavy'}</span>
      </div>

      {/* Track name */}
      <div style={{
        position: 'absolute', top: 8, left: 10,
        fontFamily: 'JetBrains Mono', fontSize: 9, color: '#444',
        letterSpacing: '0.1em', textTransform: 'uppercase',
      }}>
        {fixtureInfo?.track_name ?? 'Track'}
      </div>

      {/* Progress — shown bottom-left above legend so it doesn't clash with mode buttons */}
      {current && (
        <div style={{
          position: 'absolute', bottom: 26, left: 8,
          fontFamily: 'JetBrains Mono', fontSize: 8, color: '#555',
        }}>
          {(lap_progress * 100).toFixed(0)}%
        </div>
      )}
    </div>
  )
}
