import React, { useState, useEffect, useMemo } from 'react'
import {
  ResponsiveContainer, AreaChart, LineChart, Area, Line,
  XAxis, YAxis, CartesianGrid, Tooltip, ReferenceLine, Legend,
} from 'recharts'

const API_BASE = 'http://localhost:8000'

const DRIVER_COLORS = {
  VER: '#3671c6', HAM: '#27f4d2', LEC: '#e8002d', NOR: '#ff8000',
  SAI: '#e8002d', BOT: '#900000', ALO: '#358c75', RUS: '#27f4d2',
}

const driverColor = (driver) => DRIVER_COLORS[driver] || '#e10600'

const TOOLTIP_STYLE = {
  background: '#0d1117',
  border: '1px solid rgba(255,255,255,0.1)',
  borderRadius: 4,
  fontFamily: "'JetBrains Mono', monospace",
  fontSize: 10,
  padding: '6px 10px',
}

function ChartSection({ title, children }) {
  return (
    <div style={{
      background: 'rgba(255,255,255,0.02)',
      border: '1px solid rgba(255,255,255,0.06)',
      borderRadius: 6, padding: '8px 12px', marginBottom: 8,
    }}>
      <div style={{
        fontFamily: 'JetBrains Mono', fontSize: 9, color: '#555',
        letterSpacing: '0.12em', textTransform: 'uppercase', marginBottom: 6,
      }}>
        {title}
      </div>
      {children}
    </div>
  )
}

function StatCard({ label, valueA, valueB, driverA, driverB, unit = '', better = 'lower' }) {
  const diff = parseFloat(valueA) - parseFloat(valueB)
  const aWins = better === 'lower' ? diff < 0 : diff > 0
  return (
    <div style={{
      background: 'rgba(255,255,255,0.02)',
      border: '1px solid rgba(255,255,255,0.06)',
      borderRadius: 6, padding: '8px 10px', flex: 1,
    }}>
      <div style={{ fontFamily: 'JetBrains Mono', fontSize: 8, color: '#444', textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: 6 }}>
        {label}
      </div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
        <div>
          <span style={{ fontFamily: 'JetBrains Mono', fontSize: 13, fontWeight: 700, color: aWins ? driverColor(driverA) : '#666' }}>
            {valueA}{unit}
          </span>
          <div style={{ fontFamily: 'JetBrains Mono', fontSize: 8, color: '#444', marginTop: 2 }}>{driverA}</div>
        </div>
        <div style={{ fontFamily: 'JetBrains Mono', fontSize: 9, color: '#333' }}>vs</div>
        <div style={{ textAlign: 'right' }}>
          <span style={{ fontFamily: 'JetBrains Mono', fontSize: 13, fontWeight: 700, color: !aWins ? driverColor(driverB) : '#666' }}>
            {valueB}{unit}
          </span>
          <div style={{ fontFamily: 'JetBrains Mono', fontSize: 8, color: '#444', marginTop: 2 }}>{driverB}</div>
        </div>
      </div>
    </div>
  )
}

export function CompareLaps({ lapIdA, lapIdB }) {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    if (lapIdA < 0 || lapIdB < 0 || lapIdA === lapIdB) return
    setLoading(true)
    setError(null)
    setData(null)
    fetch(`${API_BASE}/compare/${lapIdA}/${lapIdB}`)
      .then(r => r.json())
      .then(d => {
        if (d.error) { setError(d.error); setLoading(false); return }
        setData(d)
        setLoading(false)
      })
      .catch(e => { setError(e.message); setLoading(false) })
  }, [lapIdA, lapIdB])

  // Downsample for chart performance
  const chartData = useMemo(() => {
    if (!data) return []
    const { lap_a, lap_b, delta } = data
    const n = lap_a.distance.length
    const step = Math.max(1, Math.ceil(n / 200))
    const pts = []
    for (let i = 0; i < n; i += step) {
      pts.push({
        dist: Math.round(lap_a.distance[i]),
        speedA: parseFloat(lap_a.speed[i]?.toFixed(1)),
        speedB: parseFloat(lap_b.speed[i]?.toFixed(1)),
        throttleA: parseFloat(lap_a.throttle[i]?.toFixed(1)),
        throttleB: parseFloat(lap_b.throttle[i]?.toFixed(1)),
        brakeA: (lap_a.brake[i] ?? 0) * 100,
        brakeB: (lap_b.brake[i] ?? 0) * 100,
        gearA: lap_a.gear[i] ?? 0,
        gearB: lap_b.gear[i] ?? 0,
        latGA: parseFloat(lap_a.lateral_g[i]?.toFixed(2)),
        latGB: parseFloat(lap_b.lateral_g[i]?.toFixed(2)),
        delta: parseFloat(delta[i]?.toFixed(3) ?? 0),
      })
    }
    return pts
  }, [data])

  // Stats
  const stats = useMemo(() => {
    if (!data) return null
    const a = data.lap_a, b = data.lap_b
    const avgSpeed = arr => (arr.reduce((s, v) => s + v, 0) / arr.length).toFixed(1)
    const maxVal = arr => Math.max(...arr).toFixed(1)
    const minSpeed = arr => Math.min(...arr).toFixed(1)
    return {
      lapTimeA: a.lap_time_s.toFixed(3),
      lapTimeB: b.lap_time_s.toFixed(3),
      avgSpeedA: avgSpeed(a.speed),
      avgSpeedB: avgSpeed(b.speed),
      maxSpeedA: maxVal(a.speed),
      maxSpeedB: maxVal(b.speed),
      minSpeedA: minSpeed(a.speed),
      minSpeedB: minSpeed(b.speed),
      maxLatGA: maxVal(a.lateral_g),
      maxLatGB: maxVal(b.lateral_g),
    }
  }, [data])

  // ── Empty state ──────────────────────────────────────────────────────────────
  if (lapIdA < 0 || lapIdB < 0) {
    return (
      <div style={{
        flex: 1, display: 'flex', flexDirection: 'column',
        alignItems: 'center', justifyContent: 'center',
        color: '#222', gap: 12,
      }}>
        <div style={{ fontSize: 36 }}>⚡</div>
        <div style={{ fontFamily: 'JetBrains Mono', fontSize: 12, color: '#333', letterSpacing: '0.05em' }}>
          Select two laps to compare
        </div>
        <div style={{ fontFamily: 'JetBrains Mono', fontSize: 9, color: '#222', textAlign: 'center', lineHeight: 1.8 }}>
          Use the LAP A and LAP B selectors above<br />
          Pick any two drivers, circuits, or years
        </div>
      </div>
    )
  }

  if (lapIdA === lapIdB) {
    return (
      <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <div style={{ fontFamily: 'JetBrains Mono', fontSize: 11, color: '#444' }}>
          Select two different laps to compare
        </div>
      </div>
    )
  }

  if (loading) {
    return (
      <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 12 }}>
        <div style={{ fontFamily: 'JetBrains Mono', fontSize: 11, color: '#444' }}>Loading comparison...</div>
      </div>
    )
  }

  if (error) {
    return (
      <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <div style={{ fontFamily: 'JetBrains Mono', fontSize: 11, color: '#ef5350' }}>Error: {error}</div>
      </div>
    )
  }

  if (!data) return null

  const { lap_a, lap_b } = data
  const colorA = driverColor(lap_a.driver)
  const colorB = driverColor(lap_b.driver)
  const finalDelta = data.delta[data.delta.length - 1] ?? 0
  const aFaster = finalDelta < 0

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', overflow: 'hidden' }}>

      {/* Header bar */}
      <div style={{
        display: 'flex', gap: 10, alignItems: 'center', marginBottom: 10, flexShrink: 0,
      }}>
        {/* Lap A badge */}
        <div style={{
          padding: '6px 14px', borderRadius: 5,
          background: `${colorA}18`, border: `1px solid ${colorA}55`,
        }}>
          <div style={{ fontFamily: 'JetBrains Mono', fontSize: 11, fontWeight: 700, color: colorA }}>
            {lap_a.driver}
          </div>
          <div style={{ fontFamily: 'JetBrains Mono', fontSize: 9, color: '#555', marginTop: 1 }}>
            {lap_a.gp.replace(' Grand Prix', '')} {lap_a.year} · {lap_a.lap_time_s.toFixed(3)}s
          </div>
        </div>

        <div style={{ fontFamily: 'JetBrains Mono', fontSize: 10, color: '#333' }}>vs</div>

        {/* Lap B badge */}
        <div style={{
          padding: '6px 14px', borderRadius: 5,
          background: `${colorB}18`, border: `1px solid ${colorB}55`,
        }}>
          <div style={{ fontFamily: 'JetBrains Mono', fontSize: 11, fontWeight: 700, color: colorB }}>
            {lap_b.driver}
          </div>
          <div style={{ fontFamily: 'JetBrains Mono', fontSize: 9, color: '#555', marginTop: 1 }}>
            {lap_b.gp.replace(' Grand Prix', '')} {lap_b.year} · {lap_b.lap_time_s.toFixed(3)}s
          </div>
        </div>

        {/* Gap */}
        <div style={{
          marginLeft: 'auto', padding: '6px 14px', borderRadius: 5,
          background: aFaster ? `${colorA}12` : `${colorB}12`,
          border: `1px solid ${aFaster ? colorA : colorB}44`,
        }}>
          <div style={{ fontFamily: 'JetBrains Mono', fontSize: 9, color: '#444', marginBottom: 2 }}>GAP</div>
          <div style={{ fontFamily: 'JetBrains Mono', fontSize: 14, fontWeight: 700, color: aFaster ? colorA : colorB }}>
            {aFaster ? lap_a.driver : lap_b.driver} +{Math.abs(finalDelta).toFixed(3)}s
          </div>
        </div>
      </div>

      {/* Stat cards */}
      {stats && (
        <div style={{ display: 'flex', gap: 6, marginBottom: 8, flexShrink: 0 }}>
          <StatCard label="Lap Time" valueA={stats.lapTimeA} valueB={stats.lapTimeB}
            driverA={lap_a.driver} driverB={lap_b.driver} unit="s" better="lower" />
          <StatCard label="Avg Speed" valueA={stats.avgSpeedA} valueB={stats.avgSpeedB}
            driverA={lap_a.driver} driverB={lap_b.driver} unit=" km/h" better="higher" />
          <StatCard label="Top Speed" valueA={stats.maxSpeedA} valueB={stats.maxSpeedB}
            driverA={lap_a.driver} driverB={lap_b.driver} unit=" km/h" better="higher" />
          <StatCard label="Max Lat G" valueA={stats.maxLatGA} valueB={stats.maxLatGB}
            driverA={lap_a.driver} driverB={lap_b.driver} unit="G" better="higher" />
        </div>
      )}

      {/* Charts */}
      <div style={{ flex: 1, overflowY: 'auto', overflowX: 'hidden' }}>

        {/* Speed overlay */}
        <ChartSection title={`Speed (km/h) — ${lap_a.driver} vs ${lap_b.driver}`}>
          <ResponsiveContainer width="100%" height={130}>
            <LineChart data={chartData} margin={{ top: 2, right: 6, bottom: 0, left: -24 }}>
              <CartesianGrid strokeDasharray="2 4" stroke="rgba(255,255,255,0.04)" />
              <XAxis dataKey="dist" tick={{ fontSize: 8, fill: '#444', fontFamily: 'JetBrains Mono' }} tickLine={false} />
              <YAxis domain={[60, 340]} tick={{ fontSize: 8, fill: '#444', fontFamily: 'JetBrains Mono' }} tickLine={false} />
              <Tooltip contentStyle={TOOLTIP_STYLE} labelFormatter={v => `${v}m`}
                formatter={(v, name) => [`${v} km/h`, name]} />
              <Legend wrapperStyle={{ fontFamily: 'JetBrains Mono', fontSize: 9, paddingTop: 4 }} />
              <Line type="monotone" dataKey="speedA" name={lap_a.driver} stroke={colorA}
                strokeWidth={2} dot={false} />
              <Line type="monotone" dataKey="speedB" name={lap_b.driver} stroke={colorB}
                strokeWidth={1.5} strokeDasharray="4 2" dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </ChartSection>

        {/* Time delta */}
        <ChartSection title="Cumulative Time Delta (positive = A faster)">
          <ResponsiveContainer width="100%" height={80}>
            <AreaChart data={chartData} margin={{ top: 2, right: 6, bottom: 0, left: -24 }}>
              <defs>
                <linearGradient id="deltaGradPos" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor={colorA} stopOpacity={0.25} />
                  <stop offset="95%" stopColor={colorA} stopOpacity={0} />
                </linearGradient>
                <linearGradient id="deltaGradNeg" x1="0" y1="1" x2="0" y2="0">
                  <stop offset="5%" stopColor={colorB} stopOpacity={0.25} />
                  <stop offset="95%" stopColor={colorB} stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="2 4" stroke="rgba(255,255,255,0.04)" />
              <XAxis dataKey="dist" tick={{ fontSize: 8, fill: '#444', fontFamily: 'JetBrains Mono' }} tickLine={false} />
              <YAxis tick={{ fontSize: 8, fill: '#444', fontFamily: 'JetBrains Mono' }} tickLine={false} />
              <Tooltip contentStyle={TOOLTIP_STYLE} labelFormatter={v => `${v}m`}
                formatter={(v) => [`${v > 0 ? '+' : ''}${v?.toFixed(3)}s`, 'Δ']} />
              <ReferenceLine y={0} stroke="rgba(255,255,255,0.2)" strokeDasharray="4 2" />
              <Area type="monotone" dataKey="delta" stroke={aFaster ? colorA : colorB}
                strokeWidth={2} fill={aFaster ? 'url(#deltaGradPos)' : 'url(#deltaGradNeg)'} dot={false} />
            </AreaChart>
          </ResponsiveContainer>
        </ChartSection>

        {/* Throttle overlay */}
        <ChartSection title="Throttle (%)">
          <ResponsiveContainer width="100%" height={80}>
            <LineChart data={chartData} margin={{ top: 2, right: 6, bottom: 0, left: -24 }}>
              <CartesianGrid strokeDasharray="2 4" stroke="rgba(255,255,255,0.04)" />
              <XAxis dataKey="dist" tick={{ fontSize: 8, fill: '#444', fontFamily: 'JetBrains Mono' }} tickLine={false} />
              <YAxis domain={[0, 100]} tick={{ fontSize: 8, fill: '#444', fontFamily: 'JetBrains Mono' }} tickLine={false} />
              <Tooltip contentStyle={TOOLTIP_STYLE} labelFormatter={v => `${v}m`}
                formatter={(v, name) => [`${v}%`, name]} />
              <Line type="monotone" dataKey="throttleA" name={lap_a.driver} stroke={colorA}
                strokeWidth={1.5} dot={false} />
              <Line type="monotone" dataKey="throttleB" name={lap_b.driver} stroke={colorB}
                strokeWidth={1.5} strokeDasharray="4 2" dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </ChartSection>

        {/* Brake overlay */}
        <ChartSection title="Brake">
          <ResponsiveContainer width="100%" height={55}>
            <LineChart data={chartData} margin={{ top: 2, right: 6, bottom: 0, left: -24 }}>
              <CartesianGrid strokeDasharray="2 4" stroke="rgba(255,255,255,0.04)" />
              <XAxis dataKey="dist" tick={{ fontSize: 8, fill: '#444', fontFamily: 'JetBrains Mono' }} tickLine={false} />
              <YAxis domain={[0, 100]} tick={false} tickLine={false} />
              <Tooltip contentStyle={TOOLTIP_STYLE} labelFormatter={v => `${v}m`} />
              <Line type="monotone" dataKey="brakeA" name={lap_a.driver} stroke={colorA}
                strokeWidth={1.5} dot={false} />
              <Line type="monotone" dataKey="brakeB" name={lap_b.driver} stroke={colorB}
                strokeWidth={1.5} strokeDasharray="4 2" dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </ChartSection>

        {/* Gear overlay */}
        <ChartSection title="Gear">
          <ResponsiveContainer width="100%" height={65}>
            <LineChart data={chartData} margin={{ top: 2, right: 6, bottom: 0, left: -24 }}>
              <CartesianGrid strokeDasharray="2 4" stroke="rgba(255,255,255,0.04)" />
              <XAxis dataKey="dist" tick={{ fontSize: 8, fill: '#444', fontFamily: 'JetBrains Mono' }} tickLine={false} />
              <YAxis domain={[1, 8]} ticks={[1,2,3,4,5,6,7,8]} tick={{ fontSize: 8, fill: '#444', fontFamily: 'JetBrains Mono' }} tickLine={false} />
              <Tooltip contentStyle={TOOLTIP_STYLE} labelFormatter={v => `${v}m`} />
              <Line type="stepAfter" dataKey="gearA" name={lap_a.driver} stroke={colorA}
                strokeWidth={2} dot={false} />
              <Line type="stepAfter" dataKey="gearB" name={lap_b.driver} stroke={colorB}
                strokeWidth={1.5} strokeDasharray="4 2" dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </ChartSection>

        {/* Lateral G overlay */}
        <ChartSection title="Lateral G">
          <ResponsiveContainer width="100%" height={65}>
            <LineChart data={chartData} margin={{ top: 2, right: 6, bottom: 0, left: -24 }}>
              <CartesianGrid strokeDasharray="2 4" stroke="rgba(255,255,255,0.04)" />
              <XAxis dataKey="dist" tick={{ fontSize: 8, fill: '#444', fontFamily: 'JetBrains Mono' }} tickLine={false} />
              <YAxis domain={[0, 5]} ticks={[0,2,4]} tick={{ fontSize: 8, fill: '#444', fontFamily: 'JetBrains Mono' }} tickLine={false} />
              <Tooltip contentStyle={TOOLTIP_STYLE} labelFormatter={v => `${v}m`}
                formatter={(v) => [`${v?.toFixed(2)}G`]} />
              <ReferenceLine y={3} stroke="rgba(255,167,38,0.2)" strokeDasharray="3 3" />
              <Line type="monotone" dataKey="latGA" name={lap_a.driver} stroke={colorA}
                strokeWidth={1.5} dot={false} />
              <Line type="monotone" dataKey="latGB" name={lap_b.driver} stroke={colorB}
                strokeWidth={1.5} strokeDasharray="4 2" dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </ChartSection>

      </div>
    </div>
  )
}
