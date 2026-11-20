import { useState, useEffect } from 'react'
import toast from 'react-hot-toast'
import api from '../../utils/api'
import { PRIORITY_COLORS, STATUS_COLORS, CATEGORIES, STATUSES, CATEGORY_ICONS } from '../../utils/constants'
import styles from './AdminIncidents.module.css'

export default function AdminIncidents() {
  const [incidents, setIncidents] = useState([])
  const [total, setTotal] = useState(0)
  const [page, setPage] = useState(1)
  const [pages, setPages] = useState(1)
  const [loading, setLoading] = useState(true)
  const [filters, setFilters] = useState({ status: '', category: '', priority_category: '' })
  const [selected, setSelected] = useState(null)
  const [updating, setUpdating] = useState(false)

  useEffect(() => { fetchIncidents() }, [page, filters])

  const fetchIncidents = async () => {
    setLoading(true)
    try {
      const params = { page, per_page: 10, ...filters }
      const { data } = await api.get('/admin/incidents', { params })
      setIncidents(data.incidents)
      setTotal(data.total)
      setPages(data.pages)
    } catch (e) {
      toast.error('Failed to load incidents')
    } finally {
      setLoading(false)
    }
  }

  const handleStatusUpdate = async (incidentId, newStatus) => {
    setUpdating(true)
    try {
      await api.patch(`/admin/incidents/${incidentId}`, { status: newStatus })
      toast.success('Status updated')
      fetchIncidents()
      if (selected?.id === incidentId) {
        setSelected((prev) => ({ ...prev, status: newStatus }))
      }
    } catch (e) {
      toast.error(e.message)
    } finally {
      setUpdating(false)
    }
  }

  return (
    <div className={styles.page}>
      <div className={styles.pageHeader}>
        <div>
          <h1 className={styles.title}>Incidents</h1>
          <p className={styles.subtitle}>{total} unique incidents • sorted by priority score</p>
        </div>
      </div>

      {/* Filters */}
      <div className={styles.filtersBar}>
        <select
          className={styles.filterSelect}
          value={filters.status}
          onChange={(e) => { setFilters((f) => ({ ...f, status: e.target.value })); setPage(1) }}
        >
          <option value="">All Statuses</option>
          {STATUSES.map((s) => <option key={s}>{s}</option>)}
        </select>
        <select
          className={styles.filterSelect}
          value={filters.category}
          onChange={(e) => { setFilters((f) => ({ ...f, category: e.target.value })); setPage(1) }}
        >
          <option value="">All Categories</option>
          {CATEGORIES.map((c) => <option key={c}>{c}</option>)}
        </select>
        <select
          className={styles.filterSelect}
          value={filters.priority_category}
          onChange={(e) => { setFilters((f) => ({ ...f, priority_category: e.target.value })); setPage(1) }}
        >
          <option value="">All Priorities</option>
          <option>Critical</option>
          <option>Urgent</option>
          <option>Normal</option>
        </select>
      </div>

      <div className={styles.tableContainer}>
        <table className={styles.table}>
          <thead>
            <tr>
              <th>Code</th>
              <th>Category</th>
              <th>Title / Location</th>
              <th>Priority</th>
              <th>Score</th>
              <th>Reports</th>
              <th>Status</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {loading ? (
              <tr><td colSpan={8} className={styles.loadingCell}>Loading...</td></tr>
            ) : incidents.length === 0 ? (
              <tr><td colSpan={8} className={styles.loadingCell}>No incidents found</td></tr>
            ) : incidents.map((inc) => (
              <tr key={inc.id} className={styles.row} onClick={() => setSelected(inc)}>
                <td><span className={styles.code}>{inc.incident_code}</span></td>
                <td>
                  <span className={styles.category}>
                    {CATEGORY_ICONS[inc.category]} {inc.category}
                  </span>
                </td>
                <td>
                  <div className={styles.titleCell}>
                    <div className={styles.incTitle}>{inc.title}</div>
                    <div className={styles.incLocation}>{inc.location_address || `${inc.latitude?.toFixed(3)}, ${inc.longitude?.toFixed(3)}`}</div>
                  </div>
                </td>
                <td>
                  <span
                    className={styles.priorityBadge}
                    style={{ background: PRIORITY_COLORS[inc.priority_category] + '22', color: PRIORITY_COLORS[inc.priority_category] }}
                  >
                    {inc.priority_category}
                  </span>
                </td>
                <td>
                  <div className={styles.scoreCell}>
                    <span className={styles.scoreNum} style={{ color: PRIORITY_COLORS[inc.priority_category] }}>
                      {inc.priority_score?.toFixed(1)}
                    </span>
                    <div className={styles.miniBar}>
                      <div
                        className={styles.miniBarFill}
                        style={{ width: `${(inc.priority_score / 10) * 100}%`, background: PRIORITY_COLORS[inc.priority_category] }}
                      />
                    </div>
                  </div>
                </td>
                <td><span className={styles.countBadge}>{inc.complaint_count}</span></td>
                <td>
                  <select
                    className={styles.statusSelect}
                    value={inc.status}
                    onClick={(e) => e.stopPropagation()}
                    onChange={(e) => { e.stopPropagation(); handleStatusUpdate(inc.id, e.target.value) }}
                    style={{ color: STATUS_COLORS[inc.status] }}
                    disabled={updating}
                  >
                    {STATUSES.map((s) => <option key={s} value={s}>{s}</option>)}
                  </select>
                </td>
                <td>
                  <button className={styles.detailBtn} onClick={(e) => { e.stopPropagation(); setSelected(inc) }}>
                    Details →
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Pagination */}
      <div className={styles.pagination}>
        <button disabled={page === 1} onClick={() => setPage((p) => p - 1)} className={styles.pageBtn}>← Prev</button>
        <span className={styles.pageInfo}>Page {page} of {pages} · {total} total</span>
        <button disabled={page === pages} onClick={() => setPage((p) => p + 1)} className={styles.pageBtn}>Next →</button>
      </div>

      {/* Detail drawer */}
      {selected && (
        <IncidentDrawer
          incident={selected}
          onClose={() => setSelected(null)}
          onStatusUpdate={handleStatusUpdate}
          updating={updating}
        />
      )}
    </div>
  )
}

function IncidentDrawer({ incident, onClose, onStatusUpdate, updating }) {
  return (
    <div className={styles.drawerOverlay} onClick={onClose}>
      <div className={styles.drawer} onClick={(e) => e.stopPropagation()}>
        <div className={styles.drawerHeader}>
          <div>
            <div className={styles.drawerCode}>{incident.incident_code}</div>
            <div className={styles.drawerTitle}>{incident.title}</div>
          </div>
          <button className={styles.closeBtn} onClick={onClose}>✕</button>
        </div>

        <div className={styles.drawerBody}>
          <div className={styles.drawerSection}>
            <div className={styles.drawerSectionTitle}>Priority Breakdown</div>
            <div className={styles.priorityBreakdown}>
              <ScoreRow label="Severity (35%)" value={incident.severity} max={10} color="#ef4444" />
              <ScoreRow label="Public Impact (30%)" value={incident.public_impact} max={10} color="#f97316" />
              <ScoreRow label="Location (20%)" value={incident.location_importance} max={10} color="#8b5cf6" />
              <ScoreRow label="Time/History (15%)" value={incident.time_history_score} max={10} color="#3b82f6" />
              <div className={styles.totalRow}>
                <span>Total Priority Score</span>
                <span className={styles.totalScore} style={{ color: PRIORITY_COLORS[incident.priority_category] }}>
                  {incident.priority_score?.toFixed(2)} / 10 ({incident.priority_category})
                </span>
              </div>
            </div>
          </div>

          {incident.priority_explanation?.length > 0 && (
            <div className={styles.drawerSection}>
              <div className={styles.drawerSectionTitle}>Explanation</div>
              <ul className={styles.explList}>
                {incident.priority_explanation.map((e, i) => (
                  <li key={i} className={styles.explItem}>{e}</li>
                ))}
              </ul>
            </div>
          )}

          <div className={styles.drawerSection}>
            <div className={styles.drawerSectionTitle}>Details</div>
            <div className={styles.infoGrid}>
              <div className={styles.infoItem}><span>Category</span>{CATEGORY_ICONS[incident.category]} {incident.category}</div>
              {incident.reported_category && incident.reported_category !== incident.category && (
                <div className={styles.infoItem}>
                  <span>Citizen Reported</span>
                  <span style={{ color: '#fca5a5' }}>⚠️ {incident.reported_category}</span>
                </div>
              )}
              <div className={styles.infoItem}><span>Complaints</span>{incident.complaint_count}</div>
              <div className={styles.infoItem}><span>Location</span>{incident.location_address || `${incident.latitude?.toFixed(4)}, ${incident.longitude?.toFixed(4)}`}</div>
              {incident.ml_category && (
                <div className={styles.infoItem}><span>AI Category</span>{incident.ml_category} ({(incident.ml_confidence * 100).toFixed(0)}%)</div>
              )}
              {incident.assigned_to && (
                <div className={styles.infoItem}><span>Assigned To</span>{incident.assigned_to}</div>
              )}
            </div>
          </div>

          {incident.image_url && (
            <div className={styles.drawerSection}>
              <img src={incident.image_url} alt="Incident" className={styles.drawerImg} />
            </div>
          )}

          <div className={styles.drawerActions}>
            <select
              className={styles.statusSelectLarge}
              value={incident.status}
              onChange={(e) => onStatusUpdate(incident.id, e.target.value)}
              disabled={updating}
            >
              {STATUSES.map((s) => <option key={s}>{s}</option>)}
            </select>
          </div>
        </div>
      </div>
    </div>
  )
}

function ScoreRow({ label, value, max, color }) {
  return (
    <div className={styles.scoreRow}>
      <span className={styles.scoreLabel}>{label}</span>
      <span className={styles.scoreVal} style={{ color }}>{value?.toFixed(1)}</span>
      <div className={styles.scoreBar}>
        <div className={styles.scoreFill} style={{ width: `${(value / max) * 100}%`, background: color }} />
      </div>
    </div>
  )
}

