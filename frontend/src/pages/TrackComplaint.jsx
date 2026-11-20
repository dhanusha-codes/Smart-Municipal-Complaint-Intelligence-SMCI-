import { useState, useEffect } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import toast from 'react-hot-toast'
import api from '../utils/api'
import { STATUS_COLORS, PRIORITY_COLORS, CATEGORY_ICONS } from '../utils/constants'
import styles from './TrackComplaint.module.css'

export default function TrackComplaint() {
  const [params] = useSearchParams()
  const [complaintId, setComplaintId] = useState(params.get('id') || '')
  const [complaint, setComplaint] = useState(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    if (params.get('id')) {
      handleSearch(params.get('id'))
    }
  }, [])

  const handleSearch = async (id = complaintId) => {
    if (!id.trim()) return toast.error('Enter a complaint ID')
    setLoading(true)
    setComplaint(null)
    try {
      const { data } = await api.get(`/complaints/${id.trim()}`)
      setComplaint(data)
    } catch (err) {
      toast.error(err.message || 'Complaint not found')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className={styles.page}>
      <div className={styles.topbar}>
        <Link to="/" className={styles.backLink}>← Home</Link>
        <div className={styles.topbarTitle}>🏛️ SMCI — Track Complaint</div>
        <Link to="/report" className={styles.reportLink}>Report Issue →</Link>
      </div>

      <div className={styles.container}>
        <div className={styles.header}>
          <h1 className={styles.title}>Track Your Complaint</h1>
          <p className={styles.subtitle}>Enter your complaint ID to view the current status and AI analysis.</p>
        </div>

        <div className={styles.searchCard}>
          <div className={styles.searchRow}>
            <input
              type="text"
              placeholder="Enter Complaint ID (e.g. SMCI-123456)"
              value={complaintId}
              onChange={(e) => setComplaintId(e.target.value)}
              className={styles.searchInput}
              onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
            />
            <button
              className={styles.searchBtn}
              onClick={() => handleSearch()}
              disabled={loading}
            >
              {loading ? 'Searching...' : '🔍 Track'}
            </button>
          </div>
        </div>

        {complaint && <ComplaintDetail complaint={complaint} />}

        {!complaint && !loading && (
          <div className={styles.emptyState}>
            <div className={styles.emptyIcon}>📋</div>
            <div className={styles.emptyTitle}>No complaint loaded</div>
            <div className={styles.emptyDesc}>Enter your complaint ID above to track its status.</div>
          </div>
        )}
      </div>
    </div>
  )
}

function ComplaintDetail({ complaint }) {
  const statusColor = STATUS_COLORS[complaint.status] || '#94a3b8'
  const inc = complaint.incident

  return (
    <div className={styles.detailGrid}>
      {/* Main card */}
      <div className={styles.mainCard}>
        <div className={styles.cardHeader}>
          <div>
            <div className={styles.complaintId}>#{complaint.complaint_id}</div>
            <div className={styles.category}>
              {CATEGORY_ICONS[complaint.category]} {complaint.category}
              <span className={styles.categoryTag}>Authoritative</span>
            </div>
            {complaint.reported_category && (
              <div className={styles.reportedSub}>
                <span>Citizen Reported:</span> <strong>{complaint.reported_category}</strong>
              </div>
            )}
          </div>
          <div className={styles.statusBadge} style={{ background: statusColor + '22', color: statusColor, border: `1px solid ${statusColor}44` }}>
            {complaint.status}
          </div>
        </div>

        {complaint.category_mismatch && (
          <div className={styles.mismatchAlert}>
            ⚠️ <strong>Category Mismatch:</strong> Citizen reported as <em>{complaint.reported_category}</em>, but AI classified image as <strong>{complaint.predicted_category || complaint.ml_category}</strong>. System is handling this under <strong>{complaint.category}</strong>.
          </div>
        )}

        <div className={styles.description}>{complaint.description}</div>

        <div className={styles.metaGrid}>
          <div className={styles.metaItem}>
            <span className={styles.metaLabel}>📅 Submitted</span>
            <span className={styles.metaValue}>{new Date(complaint.submitted_at).toLocaleDateString('en-IN', { year: 'numeric', month: 'long', day: 'numeric' })}</span>
          </div>
          <div className={styles.metaItem}>
            <span className={styles.metaLabel}>📍 Location</span>
            <span className={styles.metaValue}>{complaint.location_address || `${complaint.latitude?.toFixed(4)}, ${complaint.longitude?.toFixed(4)}`}</span>
          </div>
          <div className={styles.metaItem}>
            <span className={styles.metaLabel}>👤 Reporter</span>
            <span className={styles.metaValue}>{complaint.reporter_name}</span>
          </div>
        </div>

        {complaint.image_url && (
          <div className={styles.imageSection}>
            <img src={complaint.image_url} alt="Reported issue" className={styles.issueImage} />
          </div>
        )}
      </div>

      {/* Sidebar */}
      <div className={styles.sidebar}>
        {/* ML Result */}
        {complaint.ml_category && (
          <div className={styles.sideCard}>
            <div className={styles.sideCardTitle}>🤖 AI Visual Analysis</div>
            <div className={styles.mlCategory}>
              {CATEGORY_ICONS[complaint.predicted_category || complaint.ml_category]} {complaint.predicted_category || complaint.ml_category}
            </div>
            <div className={styles.mlConf}>
              <span>Confidence</span>
              <span className={styles.confValue}>{(complaint.ml_confidence * 100).toFixed(1)}%</span>
            </div>
            <div className={styles.confBar}>
              <div className={styles.confFill} style={{ width: `${complaint.ml_confidence * 100}%` }} />
            </div>
            {complaint.ml_model_type && (
              <div className={styles.modelTypeTag}>
                Engine: {complaint.ml_model_type}
                {complaint.ml_dataset_label && ` (${complaint.ml_dataset_label})`}
              </div>
            )}
            {complaint.category_mismatch && (
              <div className={styles.sidebarMismatch}>
                ⚠️ Differs from reported: {complaint.reported_category}
              </div>
            )}
            {complaint.ml_status === 'heuristic' && (
              <div className={styles.heuristicNote}>
                ⚠️ Heuristic estimate — no trained model available
              </div>
            )}
          </div>
        )}

        {/* Incident Info */}
        {inc && (
          <div className={styles.sideCard}>
            <div className={styles.sideCardTitle}>📁 Linked Incident</div>
            <div className={styles.incCode}>{inc.incident_code}</div>
            <div className={styles.incMeta}>
              <span>Status:</span>
              <span style={{ color: STATUS_COLORS[inc.status] }}>{inc.status}</span>
            </div>
            <div className={styles.incMeta}>
              <span>Priority:</span>
              <span style={{ color: PRIORITY_COLORS[inc.priority_category] }}>{inc.priority_category}</span>
            </div>
            <div className={styles.incMeta}>
              <span>Reports:</span>
              <span>{inc.complaint_count} citizens</span>
            </div>
          </div>
        )}

        {/* Duplicate note */}
        {complaint.is_duplicate && (
          <div className={styles.dupNote}>
            🔗 This complaint was matched to an existing incident using geo-visual duplicate detection.
          </div>
        )}

        {/* Status Timeline */}
        <div className={styles.sideCard}>
          <div className={styles.sideCardTitle}>📊 Status Progress</div>
          <StatusTimeline current={complaint.status} />
        </div>
      </div>
    </div>
  )
}

const TIMELINE = ['Submitted', 'AI Analysed', 'Under Review', 'Assigned', 'In Progress', 'Resolved']

function StatusTimeline({ current }) {
  const idx = TIMELINE.indexOf(current)
  return (
    <div className={styles.timeline}>
      {TIMELINE.map((s, i) => (
        <div key={s} className={`${styles.timelineItem} ${i <= idx ? styles.timelineActive : ''}`}>
          <div className={styles.timelineDot} style={{ background: i <= idx ? STATUS_COLORS[s] : 'var(--navy-600)' }} />
          <div className={styles.timelineLabel}>{s}</div>
        </div>
      ))}
    </div>
  )
}

