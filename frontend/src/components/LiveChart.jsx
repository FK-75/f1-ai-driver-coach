import React, { useMemo } from 'react'
import {
  ResponsiveContainer, LineChart, Line, AreaChart, Area,
  XAxis, YAxis, CartesianGrid, Tooltip, ReferenceLine,
} from 'recharts'

const TOOLTIP_STYLE = {
  background: '#0d1117',
  border: '1px solid #e10600',
  borderRadius: 4,
  fontFamily: "'JetBrains Mono', monospace",
  fontSize: 11,
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
        fontFamily: "'JetBrains Mono', monospace",
        fontSize: 9, color: '#666',
        letterSpacing: '0.12em', textTransform: 'uppercase', marginBottom: 6,
      }}>
        {title}
      </div>
      {children}
    </div>
  )
}

export function LiveChart({ frames, current }) {
  const chartData = useMemo(() => {
    if (frames.length <= 150) return frames
    const step = Math.ceil(frames.length / 150)
    return frames.filter((_, i) => i % step === 0)
  }, [frames])

  const deltaColor = current?.delta_time_s > 0 ? '#ff4444' : '#00e676'

  return (
    <div style={{ display: 'flex', flexDirection: 'column' }}>

      {/* Speed */}
      <ChartSection title="Speed — Driver vs Reference (km/h)">
        <ResponsiveContainer width="100%" height={130}>
          <AreaChart data={chartData} margin={{ top: 2, right: 6, bottom: 0, left: -24 }}>
            <defs>
              <linearGradient id="speedGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#e10600" stopOpacity={0.2} />
                <stop offset="95%" stopColor="#e10600" stopOpacity={0} />
              </linearGradient>
              <linearGradient id="refSpeedGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#4a9eff" stopOpacity={0.15} />
                <stop offset="95%" stopColor="#4a9eff" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="2 4" stroke="rgba(255,255,255,0.04)" />
            <XAxis dataKey="dist" tick={{ fontSize: 8, fill: '#444', fontFamily: 'JetBrains Mono' }} tickLine={false} />
            <YAxis domain={[60, 340]} tick={{ fontSize: 8, fill: '#444', fontFamily: 'JetBrains Mono' }} tickLine={false} />
            <Tooltip contentStyle={TOOLTIP_STYLE} labelFormatter={v => `${v}m`} />
            <Area type="monotone" dataKey="refSpeed" name="REF" stroke="#4a9eff" strokeWidth={1.5}
              strokeDasharray="4 2" fill="url(#refSpeedGrad)" dot={false} />
            <Area type="monotone" dataKey="speed" name="DRV" stroke="#e10600" strokeWidth={2}
              fill="url(#speedGrad)" dot={false} />
            {/* Sector boundary markers at 33% and 66% of lap distance */}
            {chartData.length > 2 && (() => {
              const maxDist = chartData[chartData.length - 1]?.dist ?? 0
              return [0.33, 0.66].map((frac, i) => (
                <ReferenceLine key={i} x={Math.round(maxDist * frac)}
                  stroke="rgba(255,215,0,0.4)" strokeDasharray="3 3" strokeWidth={1}
                  label={{ value: `S${i+2}`, position: 'insideTopRight', fontSize: 8, fill: '#ffd70088', fontFamily: 'JetBrains Mono' }}
                />
              ))
            })()}
          </AreaChart>
        </ResponsiveContainer>
      </ChartSection>

      {/* Pedals */}
      <ChartSection title="Throttle (green) / Brake (red)">
        <ResponsiveContainer width="100%" height={90}>
          <AreaChart data={chartData} margin={{ top: 2, right: 6, bottom: 0, left: -24 }}>
            <defs>
              <linearGradient id="throttleGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#00e676" stopOpacity={0.3} />
                <stop offset="95%" stopColor="#00e676" stopOpacity={0} />
              </linearGradient>
              <linearGradient id="brakeGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#ff1744" stopOpacity={0.4} />
                <stop offset="95%" stopColor="#ff1744" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="2 4" stroke="rgba(255,255,255,0.04)" />
            <XAxis dataKey="dist" tick={{ fontSize: 8, fill: '#444', fontFamily: 'JetBrains Mono' }} tickLine={false} />
            <YAxis domain={[0, 100]} tick={{ fontSize: 8, fill: '#444', fontFamily: 'JetBrains Mono' }} tickLine={false} />
            <Tooltip contentStyle={TOOLTIP_STYLE} labelFormatter={v => `${v}m`} />
            <Area type="monotone" dataKey="throttle" name="Throttle %" stroke="#00e676" strokeWidth={1.5}
              fill="url(#throttleGrad)" dot={false} />
            <Area type="monotone" dataKey="brake" name="Brake %" stroke="#ff1744" strokeWidth={1.5}
              fill="url(#brakeGrad)" dot={false} />
          </AreaChart>
        </ResponsiveContainer>
      </ChartSection>

      {/* Delta */}
      <ChartSection title="Lap Time Delta (s vs reference)">
        <ResponsiveContainer width="100%" height={80}>
          <LineChart data={chartData} margin={{ top: 2, right: 6, bottom: 0, left: -24 }}>
            <CartesianGrid strokeDasharray="2 4" stroke="rgba(255,255,255,0.04)" />
            <XAxis dataKey="dist" tick={{ fontSize: 8, fill: '#444', fontFamily: 'JetBrains Mono' }} tickLine={false} />
            <YAxis tick={{ fontSize: 8, fill: '#444', fontFamily: 'JetBrains Mono' }} tickLine={false} />
            <Tooltip contentStyle={TOOLTIP_STYLE} labelFormatter={v => `${v}m`}
              formatter={(v) => [`${v > 0 ? '+' : ''}${v?.toFixed(3)}s`, 'Δ']} />
            <ReferenceLine y={0} stroke="rgba(255,255,255,0.2)" strokeDasharray="4 2" />
            <Line type="monotone" dataKey="delta" stroke={deltaColor} strokeWidth={2} dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </ChartSection>

      {/* Gear */}
      <ChartSection title="Gear — Driver vs Reference">
        <ResponsiveContainer width="100%" height={70}>
          <LineChart data={chartData} margin={{ top: 2, right: 6, bottom: 0, left: -24 }}>
            <CartesianGrid strokeDasharray="2 4" stroke="rgba(255,255,255,0.04)" />
            <XAxis dataKey="dist" tick={{ fontSize: 8, fill: '#444', fontFamily: 'JetBrains Mono' }} tickLine={false} />
            <YAxis domain={[1, 8]} ticks={[1,2,3,4,5,6,7,8]} tick={{ fontSize: 8, fill: '#444', fontFamily: 'JetBrains Mono' }} tickLine={false} />
            <Tooltip contentStyle={TOOLTIP_STYLE} labelFormatter={v => `${v}m`}
              formatter={(v, name) => [v, name === 'gear' ? 'Driver' : 'Ref']} />
            <Line type="stepAfter" dataKey="refGear" name="refGear" stroke="#4a9eff"
              strokeWidth={1.5} strokeDasharray="4 2" dot={false} />
            <Line type="stepAfter" dataKey="gear" name="gear" stroke="#e10600"
              strokeWidth={2} dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </ChartSection>

      {/* Lateral G */}
      <ChartSection title="Lateral G — Cornering Load">
        <ResponsiveContainer width="100%" height={70}>
          <AreaChart data={chartData} margin={{ top: 2, right: 6, bottom: 0, left: -24 }}>
            <defs>
              <linearGradient id="latGGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#ffa726" stopOpacity={0.35} />
                <stop offset="95%" stopColor="#ffa726" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="2 4" stroke="rgba(255,255,255,0.04)" />
            <XAxis dataKey="dist" tick={{ fontSize: 8, fill: '#444', fontFamily: 'JetBrains Mono' }} tickLine={false} />
            <YAxis domain={[0, 5]} ticks={[0,2,4]} tick={{ fontSize: 8, fill: '#444', fontFamily: 'JetBrains Mono' }} tickLine={false} />
            <Tooltip contentStyle={TOOLTIP_STYLE} labelFormatter={v => `${v}m`}
              formatter={(v) => [`${v?.toFixed(2)}G`, 'Lat G']} />
            <ReferenceLine y={3} stroke="rgba(255,167,38,0.25)" strokeDasharray="3 3" />
            <Area type="monotone" dataKey="latG" stroke="#ffa726" strokeWidth={1.5}
              fill="url(#latGGrad)" dot={false} />
          </AreaChart>
        </ResponsiveContainer>
      </ChartSection>

    </div>
  )
}
