import { useEffect, useRef, useState } from 'react'

const WS_BASE = 'ws://localhost:8000'
const MAX_CUES = 8

// Batch chart points every N frames — avoids 1000+ individual React re-renders
const CHART_BATCH = 5

function toChartPoint(frame) {
  return {
    dist: Math.round(frame.distance_m ?? 0),
    speed: frame.speed ?? 0,
    refSpeed: frame.ref_speed ?? 0,
    throttle: frame.throttle ?? 0,
    brake: (frame.brake ?? 0) * 100,
    delta: frame.delta_time_s ?? 0,
    gear: frame.gear ?? 0,
    refGear: frame.ref_gear ?? 0,
    latG: frame.lateral_g ?? 0,
  }
}

export function useWebSocket() {
  const socketRef = useRef(null)
  const stopRequestedRef = useRef(false)
  const pendingPointsRef = useRef([])
  const pendingCuesRef = useRef([])

  const [status, setStatus] = useState('idle')
  const [frames, setFrames] = useState([])
  const [current, setCurrent] = useState(null)
  const [summary, setSummary] = useState(null)
  const [cues, setCues] = useState([])
  const [fixtureInfo, setFixtureInfo] = useState(null)
  const [lapKey, setLapKey] = useState(0)
  const [sectorDeltas, setSectorDeltas] = useState([null, null, null])
  const [pbDelta, setPbDelta] = useState(() => {
    const stored = localStorage.getItem('f1coach_pb_delta')
    return stored !== null ? parseFloat(stored) : null
  })
  const [newPb, setNewPb] = useState(false)

  useEffect(() => {
    return () => {
      stopRequestedRef.current = true
      socketRef.current?.close()
      socketRef.current = null
    }
  }, [])

  function resetSession() {
    setFrames([])
    setCurrent(null)
    setSummary(null)
    setCues([])
    setSectorDeltas([null, null, null])
    setNewPb(false)
    pendingPointsRef.current = []
    pendingCuesRef.current = []
  }

  function flushBatch() {
    const pts = pendingPointsRef.current.splice(0)
    const newCues = pendingCuesRef.current.splice(0)
    if (pts.length > 0) setFrames((prev) => [...prev, ...pts])
    if (newCues.length > 0) setCues((prev) => [...newCues, ...prev].slice(0, MAX_CUES))
  }

  function start(speed = 1, lapId = -1) {
    // Close any existing socket cleanly first, without triggering stop logic
    const prev = socketRef.current
    if (prev) {
      prev.onclose = null  // detach handler so it doesn't interfere
      prev.close()
      socketRef.current = null
    }

    stopRequestedRef.current = false
    resetSession()
    setStatus('connecting')

    const socket = new WebSocket(`${WS_BASE}/ws/replay?speed=${encodeURIComponent(speed)}&lap_id=${lapId}`)
    socketRef.current = socket

    socket.onopen = () => {
      if (socketRef.current !== socket) return
      socket.send(JSON.stringify({ action: 'start' }))
      setStatus('running')
    }

    socket.onmessage = (event) => {
      if (socketRef.current !== socket) return

      const message = JSON.parse(event.data)

      if (message.type === 'fixture') {
        setFixtureInfo(message.data)
        setLapKey(k => k + 1)
        return
      }

      if (message.type === 'frame') {
        const frame = message.data
        setCurrent(frame)
        pendingPointsRef.current.push(toChartPoint(frame))
        if (frame.cue) {
          pendingCuesRef.current.push({
            ...frame.cue,
            distance_m: frame.cue.distance_m ?? frame.distance_m ?? 0,
          })
        }
        // Capture delta at sector boundaries (S1 at 33%, S2 at 66%)
        const prog = frame.lap_progress ?? 0
        const delta = frame.delta_time_s ?? 0
        if (prog >= 0.33 && prog < 0.34) {
          setSectorDeltas(prev => [delta, prev[1], prev[2]])
        } else if (prog >= 0.66 && prog < 0.67) {
          setSectorDeltas(prev => [prev[0], delta, prev[2]])
        } else if (prog >= 0.99) {
          setSectorDeltas(prev => [prev[0], prev[1], delta])
        }
        if (pendingPointsRef.current.length >= CHART_BATCH) flushBatch()
        return
      }

      if (message.type === 'complete') {
        flushBatch()
        setSummary(message.data)
        setStatus('complete')
        // Check personal best using final_delta_s (negative = faster than ref = good)
        const finalDelta = message.data?.final_delta_s ?? null
        if (finalDelta !== null) {
          setPbDelta(prev => {
            if (prev === null || finalDelta < prev) {
              localStorage.setItem('f1coach_pb_delta', String(finalDelta))
              setNewPb(true)
              return finalDelta
            }
            return prev
          })
        }
        setTimeout(() => socket.close(), 50)
        return
      }

      if (message.type === 'error') {
        setStatus('error')
      }
    }

    socket.onerror = () => {
      if (socketRef.current !== socket) return
      // Don't overwrite a clean complete with a spurious error
      setStatus((prev) => (prev === 'complete' ? 'complete' : 'error'))
    }

    socket.onclose = () => {
      // This fires for both user-initiated stop and natural completion
      if (socketRef.current !== socket) return
      socketRef.current = null

      if (stopRequestedRef.current) {
        // User pressed Stop — go back to idle so Start works again
        stopRequestedRef.current = false
        setStatus('idle')
        return
      }

      // Natural close: keep whatever status was already set (complete/error)
      setStatus((prev) => {
        if (prev === 'complete' || prev === 'error') return prev
        return prev === 'connecting' ? 'error' : 'idle'
      })
    }
  }

  function stop() {
    stopRequestedRef.current = true
    // Don't null socketRef here — let onclose do it so the guard works correctly
    if (socketRef.current?.readyState === WebSocket.OPEN) {
      socketRef.current.send(JSON.stringify({ action: 'stop' }))
    }
    socketRef.current?.close()
    // socketRef.current intentionally NOT nulled here
  }

  return { status, frames, current, summary, cues, sectorDeltas, pbDelta, newPb, fixtureInfo, lapKey, start, stop }
}