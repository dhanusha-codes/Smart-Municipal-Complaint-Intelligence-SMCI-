import { useEffect, useState, useRef } from 'react'
import { MapContainer, TileLayer, Marker, Popup, useMap } from 'react-leaflet'
import L from 'leaflet'
import api from '../../utils/api'
import { PRIORITY_COLORS, STATUS_COLORS, CATEGORY_ICONS, MAP_CENTER } from '../../utils/constants'
import styles from './AdminMap.module.css'

// Fix Leaflet default icon paths
delete L.Icon.Default.prototype._getIconUrl
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png',
  iconUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png',
  shadowUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png',
})

const TILE_LAYERS = {
  osm: {
    name: '🌐 OpenStreetMap',
    url: 'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
  },
  dark: {
    name: '🌙 Dark Canvas',
    url: 'https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}',
    attribution: '&copy; Esri, HERE, Garmin, © OpenStreetMap contributors',
  },
  streets: {
    name: '🗺️ Esri Streets',
    url: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}',
    attribution: '&copy; Esri, DeLorme, NAVTEQ, TomTom',
  },
}

function createColorIcon(color) {
  return L.divIcon({
    className: '',
    html: `<div style="
      width:28px;height:28px;border-radius:50% 50% 50% 0;
      background:${color};transform:rotate(-45deg);
      border:3px solid rgba(255,255,255,0.85);
      box-shadow:0 3px 10px rgba(0,0,0,0.5);
    "></div>`,
    iconSize: [28, 28],
    iconAnchor: [14, 28],
    popupAnchor: [0, -30],
  })
}

// Controller to auto-invalidate size and fit bounds to incidents
function MapController({ incidents }) {
  const map = useMap()

  useEffect(() => {
    // Invalidate size immediately and after layout stabilization
    map.invalidateSize()
    const t1 = setTimeout(() => map.invalidateSize(), 100)
    const t2 = setTimeout(() => map.invalidateSize(), 300)

    const handleResize = () => map.invalidateSize()
    window.addEventListener('resize', handleResize)

    // Fit bounds if valid coordinates exist
    if (incidents && incidents.length > 0) {
      const validCoords = incidents
        .filter((i) => i.latitude && i.longitude && !isNaN(i.latitude) && !isNaN(i.longitude))
        .map((i) => [Number(i.latitude), Number(i.longitude)])

      if (validCoords.length === 1) {
        map.setView(validCoords[0], 13)
      } else if (validCoords.length > 1) {
        const bounds = L.latLngBounds(validCoords)
        map.fitBounds(bounds, { padding: [50, 50], maxZoom: 15 })
      }
    }

    return () => {
      clearTimeout(t1)
      clearTimeout(t2)
      window.removeEventListener('resize', handleResize)
    }
  }, [map, incidents])

  return null
}

export default function AdminMap() {
  const [incidents, setIncidents] = useState([])
  const [loading, setLoading] = useState(true)
  const [filterPriority, setFilterPriority] = useState('')
  const [filterCategory, setFilterCategory] = useState('')
  const [selected, setSelected] = useState(null)
  const [tileKey, setTileKey] = useState('osm')

  useEffect(() => { fetchMapData() }, [])

  const fetchMapData = async () => {
    try {
      const { data } = await api.get('/incidents/map')
      setIncidents(data.incidents || [])
    } catch (e) {
      console.error(e)
    } finally {
      setLoading(false)
    }
  }

  const filtered = incidents.filter((inc) => {
    if (filterPriority && inc.priority_category !== filterPriority) return false
    if (filterCategory && inc.category !== filterCategory) return false
    return true
  })

  const categories = [...new Set(incidents.map((i) => i.category))]
  const activeTile = TILE_LAYERS[tileKey] || TILE_LAYERS.osm

  return (
    <div className={styles.page}>
      <div className={styles.pageHeader}>
        <div>
          <h1 className={styles.title}>Geo-Visual Map</h1>
          <p className={styles.subtitle}>{filtered.length} incidents plotted</p>
        </div>
        <div className={styles.controls}>
          {/* Tile Layer Switcher */}
          <div className={styles.layerGroup}>
            {Object.entries(TILE_LAYERS).map(([key, layer]) => (
              <button
                key={key}
                type="button"
                className={`${styles.layerBtn} ${tileKey === key ? styles.layerActive : ''}`}
                onClick={() => setTileKey(key)}
              >
                {layer.name}
              </button>
            ))}
          </div>

          <select
            className={styles.filterSelect}
            value={filterPriority}
            onChange={(e) => setFilterPriority(e.target.value)}
          >
            <option value="">All Priorities</option>
            <option>Critical</option>
            <option>Urgent</option>
            <option>Normal</option>
          </select>
          <select
            className={styles.filterSelect}
            value={filterCategory}
            onChange={(e) => setFilterCategory(e.target.value)}
          >
            <option value="">All Categories</option>
            {categories.map((c) => <option key={c}>{c}</option>)}
          </select>
        </div>
      </div>

      {/* Legend */}
      <div className={styles.legend}>
        {Object.entries(PRIORITY_COLORS).map(([k, v]) => (
          <div key={k} className={styles.legendItem}>
            <div className={styles.legendDot} style={{ background: v }} />
            <span>{k}</span>
          </div>
        ))}
        <div className={styles.legendItem}>
          <div className={styles.legendDot} style={{ background: '#94a3b8' }} />
          <span>Normal</span>
        </div>
      </div>

      <div className={styles.mapWrapper}>
        {loading ? (
          <div className={styles.mapLoading}>Loading map data...</div>
        ) : (
          <MapContainer
            center={MAP_CENTER}
            zoom={12}
            style={{ height: '560px', width: '100%', borderRadius: '12px' }}
            className={styles.map}
          >
            <TileLayer
              key={tileKey}
              attribution={activeTile.attribution}
              url={activeTile.url}
              maxZoom={19}
            />
            <MapController incidents={filtered} />
            {filtered.map((inc) => {
              const color = PRIORITY_COLORS[inc.priority_category] || '#94a3b8'
              return (
                <Marker
                  key={inc.id}
                  position={[inc.latitude, inc.longitude]}
                  icon={createColorIcon(color)}
                  eventHandlers={{ click: () => setSelected(inc) }}
                >
                  <Popup maxWidth={280}>
                    <div className={styles.popup}>
                      <div className={styles.popupCode}>{inc.incident_code}</div>
                      <div className={styles.popupTitle}>{inc.title}</div>
                      <div className={styles.popupMeta}>{inc.location_address || `${inc.latitude.toFixed(4)}, ${inc.longitude.toFixed(4)}`}</div>
                      <div className={styles.popupRow}>
                        <span style={{ color }}>● {inc.priority_category}</span>
                        <span>Score: {inc.priority_score?.toFixed(1)}</span>
                      </div>
                      <div className={styles.popupRow}>
                        <span style={{ color: STATUS_COLORS[inc.status] }}>{inc.status}</span>
                        <span>📢 {inc.complaint_count} reports</span>
                      </div>
                      {inc.image_url && (
                        <img src={inc.image_url} alt="incident" className={styles.popupImg} />
                      )}
                    </div>
                  </Popup>
                </Marker>
              )
            })}
          </MapContainer>
        )}
      </div>

      {/* Stats below map */}
      <div className={styles.mapStats}>
        {Object.entries(PRIORITY_COLORS).map(([priority, color]) => {
          const count = incidents.filter((i) => i.priority_category === priority).length
          return (
            <div key={priority} className={styles.mapStat}>
              <div className={styles.mapStatDot} style={{ background: color }} />
              <div className={styles.mapStatLabel}>{priority}</div>
              <div className={styles.mapStatCount} style={{ color }}>{count}</div>
            </div>
          )
        })}
      </div>
    </div>
  )
}

