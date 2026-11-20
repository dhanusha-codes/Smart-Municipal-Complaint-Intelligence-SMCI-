import { useEffect, useState, useCallback } from 'react'
import { MapContainer, TileLayer, Circle, Popup, Marker, useMap } from 'react-leaflet'
import L from 'leaflet'
import api from '../../utils/api'
import { MAP_CENTER } from '../../utils/constants'
import styles from './AdminHotspots.module.css'

// ─── Leaflet icon fix ────────────────────────────────────────────────────────
delete L.Icon.Default.prototype._getIconUrl
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png',
  iconUrl:       'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png',
  shadowUrl:     'https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png',
})

// ─── Constants ───────────────────────────────────────────────────────────────
const RISK_COLORS = {
  Critical: '#ef4444',
  High:     '#f97316',
  Moderate: '#eab308',
  Low:      '#22c55e',
}

const RISK_BG = {
  Critical: 'rgba(239,68,68,0.15)',
  High:     'rgba(249,115,22,0.15)',
  Moderate: 'rgba(234,179,8,0.15)',
  Low:      'rgba(34,197,94,0.12)',
}

function createCentroidIcon(risk) {
  const color = RISK_COLORS[risk] || '#94a3b8'
  return L.divIcon({
    className: '',
    html: `<div style="
      width:18px;height:18px;border-radius:50%;
      background:${color};
      border:3px solid rgba(255,255,255,0.9);
      box-shadow:0 2px 8px rgba(0,0,0,0.5);
    "></div>`,
    iconSize:    [18, 18],
    iconAnchor:  [9, 9],
    popupAnchor: [0, -12],
  })
}

// ─── Map auto-fit & resize ───────────────────────────────────────────────────
function FitBounds({ hotspots }) {
  const map = useMap()
  useEffect(() => {
    map.invalidateSize()
    const t = setTimeout(() => map.invalidateSize(), 200)

    if (hotspots && hotspots.length > 0) {
      const pts = hotspots.map(h => [h.centroid_lat, h.centroid_lon])
      if (pts.length === 1) {
        map.setView(pts[0], 13)
      } else {
        map.fitBounds(pts, { padding: [40, 40] })
      }
    }
    return () => clearTimeout(t)
  }, [hotspots, map])
  return null
}

// ─── Component ───────────────────────────────────────────────────────────────
export default function AdminHotspots() {
  const [hotspots,     setHotspots]     = useState([])
  const [summary,      setSummary]      = useState(null)
  const [loading,      setLoading]      = useState(false)
  const [error,        setError]        = useState(null)
  const [selected,     setSelected]     = useState(null)
  const [filterRisk,   setFilterRisk]   = useState('all')
  const [filterCat,    setFilterCat]    = useState('all')

  // DBSCAN params
  const [epsKm,        setEpsKm]        = useState(0.5)
  const [minSamples,   setMinSamples]   = useState(2)

  const fetchHotspots = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const { data } = await api.get('/admin/hotspots', {
        params: { eps_km: epsKm, min_samples: minSamples },
      })
      setHotspots(data.hotspots || [])
      setSummary({
        total:    data.total_incidents_clustered,
        clusters: data.cluster_count,
        noise:    data.noise_count,
        risk:     data.risk_summary || {},
        eps_km:   data.eps_km,
        min_samples: data.min_samples,
      })
    } catch (err) {
      setError(err.response?.data?.error || 'Failed to load hotspot data.')
    } finally {
      setLoading(false)
    }
  }, [epsKm, minSamples])

  useEffect(() => { fetchHotspots() }, [fetchHotspots])

  // Derived display list
  const clusters = hotspots.filter(h => !h.is_noise)
  const noise    = hotspots.filter(h => h.is_noise)

  const displayList = clusters.filter(h => {
    if (filterRisk !== 'all' && h.risk_level !== filterRisk) return false
    if (filterCat  !== 'all' && h.dominant_category !== filterCat) return false
    return true
  })

  const allCategories = [...new Set(clusters.map(h => h.dominant_category))].sort()

  return (
    <div className={styles.container}>
      {/* Header */}
      <div className={styles.header}>
        <div>
          <h2 className={styles.title}>Spatial Hotspot Analysis</h2>
          <p className={styles.subtitle}>
            DBSCAN clustering of geo-located incidents — identifies high-density complaint zones
          </p>
        </div>
        <button className={styles.refreshBtn} onClick={fetchHotspots} disabled={loading}>
          {loading ? '⏳ Running...' : '🔄 Re-run DBSCAN'}
        </button>
      </div>

      {/* DBSCAN params */}
      <div className={styles.params}>
        <span className={styles.paramLabel}>Algorithm parameters:</span>
        <label className={styles.paramItem}>
          Radius (ε)
          <input
            type="number" min={0.1} max={20} step={0.1}
            value={epsKm}
            onChange={e => setEpsKm(parseFloat(e.target.value) || 0.5)}
            className={styles.paramInput}
          />
          <span className={styles.paramUnit}>km</span>
        </label>
        <label className={styles.paramItem}>
          Min. incidents
          <input
            type="number" min={1} max={20} step={1}
            value={minSamples}
            onChange={e => setMinSamples(parseInt(e.target.value) || 2)}
            className={styles.paramInput}
          />
        </label>
        <button className={styles.applyBtn} onClick={fetchHotspots} disabled={loading}>
          Apply
        </button>
      </div>

      {error && <div className={styles.error}>⚠ {error}</div>}

      {/* Summary cards */}
      {summary && (
        <div className={styles.summaryRow}>
          <div className={styles.summaryCard}>
            <span className={styles.summaryVal}>{summary.total}</span>
            <span className={styles.summaryLabel}>Incidents Analysed</span>
          </div>
          <div className={styles.summaryCard}>
            <span className={styles.summaryVal}>{summary.clusters}</span>
            <span className={styles.summaryLabel}>Hotspot Clusters</span>
          </div>
          <div className={styles.summaryCard}>
            <span className={styles.summaryVal}>{summary.noise}</span>
            <span className={styles.summaryLabel}>Isolated Incidents</span>
          </div>
          {Object.entries(summary.risk || {}).map(([level, count]) => (
            <div
              key={level}
              className={styles.summaryCard}
              style={{ borderColor: RISK_COLORS[level] }}
            >
              <span className={styles.summaryVal} style={{ color: RISK_COLORS[level] }}>
                {count}
              </span>
              <span className={styles.summaryLabel}>{level} Risk</span>
            </div>
          ))}
        </div>
      )}

      {/* Map */}
      <div className={styles.mapWrapper}>
        <MapContainer
          center={MAP_CENTER || [19.076, 72.877]}
          zoom={11}
          className={styles.map}
        >
          <TileLayer
            url="https://tile.openstreetmap.org/{z}/{x}/{y}.png"
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          />
          <FitBounds hotspots={clusters} />

          {/* Cluster circles */}
          {clusters.map((h, i) => {
            const color = RISK_COLORS[h.risk_level] || '#94a3b8'
            const radiusM = h.eps_km * 1000
            return (
              <Circle
                key={i}
                center={[h.centroid_lat, h.centroid_lon]}
                radius={radiusM}
                pathOptions={{
                  color,
                  fillColor: color,
                  fillOpacity: selected === i ? 0.35 : 0.18,
                  weight: selected === i ? 3 : 1.5,
                }}
                eventHandlers={{ click: () => setSelected(i) }}
              >
                <Popup>
                  <div className={styles.popup}>
                    <div
                      className={styles.popupRisk}
                      style={{ background: RISK_BG[h.risk_level], color: RISK_COLORS[h.risk_level] }}
                    >
                      {h.risk_level} Risk
                    </div>
                    <strong>{h.member_count} incidents</strong>
                    <div>Category: {h.dominant_category}</div>
                    <div>Avg Priority: {h.avg_priority_score}</div>
                    <div>Max Priority: {h.max_priority_score}</div>
                    <div className={styles.popupCodes}>
                      {h.incident_codes.slice(0, 5).join(', ')}
                      {h.incident_codes.length > 5 && ` +${h.incident_codes.length - 5} more`}
                    </div>
                  </div>
                </Popup>
              </Circle>
            )
          })}

          {/* Cluster centroid markers */}
          {clusters.map((h, i) => (
            <Marker
              key={`m-${i}`}
              position={[h.centroid_lat, h.centroid_lon]}
              icon={createCentroidIcon(h.risk_level)}
              eventHandlers={{ click: () => setSelected(i) }}
            />
          ))}

          {/* Noise points */}
          {noise.map((h, i) => (
            <Marker
              key={`n-${i}`}
              position={[h.centroid_lat, h.centroid_lon]}
            >
              <Popup>
                <div className={styles.popup}>
                  <div className={styles.popupRisk} style={{ background: '#1e293b', color: '#94a3b8' }}>
                    Isolated
                  </div>
                  <div>Category: {h.dominant_category}</div>
                  <div>{h.incident_codes[0]}</div>
                </div>
              </Popup>
            </Marker>
          ))}
        </MapContainer>
      </div>

      {/* Cluster list */}
      <div className={styles.listSection}>
        <div className={styles.listHeader}>
          <h3 className={styles.listTitle}>
            Cluster Details ({displayList.length})
          </h3>
          <div className={styles.filters}>
            <select
              value={filterRisk}
              onChange={e => setFilterRisk(e.target.value)}
              className={styles.filterSelect}
            >
              <option value="all">All Risk Levels</option>
              {['Critical', 'High', 'Moderate', 'Low'].map(r => (
                <option key={r} value={r}>{r}</option>
              ))}
            </select>
            <select
              value={filterCat}
              onChange={e => setFilterCat(e.target.value)}
              className={styles.filterSelect}
            >
              <option value="all">All Categories</option>
              {allCategories.map(c => <option key={c} value={c}>{c}</option>)}
            </select>
          </div>
        </div>

        {loading && <div className={styles.loadingMsg}>Running DBSCAN…</div>}

        {!loading && displayList.length === 0 && (
          <div className={styles.empty}>
            No clusters found. Try increasing ε (radius) or decreasing min. incidents.
          </div>
        )}

        <div className={styles.clusterGrid}>
          {displayList.map((h, i) => (
            <div
              key={i}
              className={`${styles.clusterCard} ${selected === clusters.indexOf(h) ? styles.selectedCard : ''}`}
              style={{ borderLeftColor: RISK_COLORS[h.risk_level] }}
              onClick={() => setSelected(clusters.indexOf(h))}
            >
              <div className={styles.cardTop}>
                <span
                  className={styles.riskBadge}
                  style={{ background: RISK_BG[h.risk_level], color: RISK_COLORS[h.risk_level] }}
                >
                  {h.risk_level}
                </span>
                <span className={styles.memberCount}>
                  {h.member_count} incident{h.member_count !== 1 ? 's' : ''}
                </span>
              </div>

              <div className={styles.cardCat}>{h.dominant_category}</div>

              <div className={styles.cardMeta}>
                <span>📍 {h.centroid_lat.toFixed(4)}, {h.centroid_lon.toFixed(4)}</span>
              </div>

              <div className={styles.cardScores}>
                <div className={styles.scoreItem}>
                  <span className={styles.scoreLabel}>Avg Priority</span>
                  <span className={styles.scoreVal}>{h.avg_priority_score}</span>
                </div>
                <div className={styles.scoreItem}>
                  <span className={styles.scoreLabel}>Max Priority</span>
                  <span className={styles.scoreVal}>{h.max_priority_score}</span>
                </div>
              </div>

              <div className={styles.catBreakdown}>
                {Object.entries(h.category_breakdown).map(([cat, cnt]) => (
                  <span key={cat} className={styles.catPill}>
                    {cat} ({cnt})
                  </span>
                ))}
              </div>

              <div className={styles.codeList}>
                {h.incident_codes.slice(0, 4).map(code => (
                  <span key={code} className={styles.codePill}>{code}</span>
                ))}
                {h.incident_codes.length > 4 && (
                  <span className={styles.moreCode}>+{h.incident_codes.length - 4}</span>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

