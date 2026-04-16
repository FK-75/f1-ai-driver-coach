import React, { useEffect, useState } from 'react'

const API_BASE = 'http://localhost:8000'

const DRIVER_COLORS = {
  VER: '#3671c6', HAM: '#27f4d2', LEC: '#e8002d', NOR: '#ff8000',
  SAI: '#e8002d', BOT: '#900000', ALO: '#358c75', RUS: '#27f4d2',
}

export function LapSelector({ onSelect, disabled }) {
  const [circuits, setCircuits] = useState({})
  const [laps, setLaps] = useState([])
  const [selectedCircuit, setSelectedCircuit] = useState('')
  const [selectedLapId, setSelectedLapId] = useState(-1)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetch(`${API_BASE}/laps`)
      .then(r => r.json())
      .then(d => {
        setLaps(d.laps || [])
        setCircuits(d.circuits || {})
        setLoading(false)
      })
      .catch(() => setLoading(false))
  }, [])

  const circuitKeys = Object.keys(circuits).sort()
  const circuitLaps = selectedCircuit ? (circuits[selectedCircuit] || []) : []

  function handleCircuitChange(e) {
    const val = e.target.value
    setSelectedCircuit(val)
    if (val) {
      // Auto-select the fastest lap in this circuit as a circuit-demo
      const cLaps = circuits[val] || []
      const fastest = cLaps.reduce((a, b) => b.lap_time_s < a.lap_time_s ? b : a, cLaps[0])
      if (fastest) {
        setSelectedLapId(fastest.id)
        onSelect(fastest.id)
      }
    } else {
      setSelectedLapId(-1)
      onSelect(-1)
    }
  }

  function handleLapChange(e) {
    const id = parseInt(e.target.value)
    setSelectedLapId(id)
    onSelect(id)
  }

  function handleReset() {
    setSelectedCircuit('')
    setSelectedLapId(-1)
    onSelect(-1)
  }

  const sel = {
    background: '#0d1117',
    border: '1px solid rgba(255,255,255,0.12)',
    borderRadius: 4,
    color: '#888',
    fontFamily: 'JetBrains Mono',
    fontSize: 10,
    padding: '4px 8px',
    cursor: disabled ? 'not-allowed' : 'pointer',
    opacity: disabled ? 0.5 : 1,
    outline: 'none',
  }

  if (loading || laps.length === 0) return null

  const selectedLap = laps.find(l => l.id === selectedLapId)
  const driverColor = selectedLap ? (DRIVER_COLORS[selectedLap.driver] || '#888') : '#888'

  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
      <span style={{ fontFamily: 'JetBrains Mono', fontSize: 9, color: '#444' }}>LAP</span>

      <select value={selectedCircuit} onChange={handleCircuitChange} disabled={disabled} style={sel}>
        <option value="">Demo (Silverstone)</option>
        {circuitKeys.map(k => <option key={k} value={k}>{k}</option>)}
      </select>

      {selectedCircuit && (
        <select value={selectedLapId} onChange={handleLapChange} disabled={disabled} style={sel}>
          {circuitLaps.map(lap => (
            <option key={lap.id} value={lap.id}>
              {lap.driver}  {lap.lap_time_s.toFixed(3)}s
            </option>
          ))}
        </select>
      )}

      {selectedLap && (
        <div style={{
          padding: '2px 8px', borderRadius: 3,
          background: `${driverColor}22`,
          border: `1px solid ${driverColor}66`,
          fontFamily: 'JetBrains Mono', fontSize: 9,
          color: driverColor,
        }}>
          {selectedLap.driver}
        </div>
      )}

      {selectedLapId >= 0 && (
        <button onClick={handleReset} disabled={disabled} style={{
          background: 'transparent',
          border: '1px solid rgba(255,255,255,0.08)',
          borderRadius: 3, color: '#444',
          fontFamily: 'JetBrains Mono', fontSize: 9,
          padding: '2px 6px', cursor: 'pointer',
        }}>✕</button>
      )}
    </div>
  )
}
