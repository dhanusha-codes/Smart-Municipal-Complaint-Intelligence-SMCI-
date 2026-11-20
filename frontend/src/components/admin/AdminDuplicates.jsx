import { useState, useEffect } from 'react'
import toast from 'react-hot-toast'
import api from '../../utils/api'
import styles from './AdminDuplicates.module.css'

export default function AdminDuplicates() {
  const [matches, setMatches] = useState([])
  const [loading, setLoading] = useState(true)
  const [showReviewed, setShowReviewed] = useState(false)
  const [reviewing, setReviewing] = useState(null)

  useEffect(() => { fetchMatches() }, [showReviewed])

  const fetchMatches = async () => {
    setLoading(true)
    try {
      const { data } = await api.get('/admin/duplicate-matches', { params: { reviewed: showReviewed } })
      setMatches(data.matches)
    } catch {
      toast.error('Failed to load duplicate matches')
    } finally {
      setLoading(false)
    }
  }

  const handleReview = async (matchId, action) => {
    setReviewing(matchId)
    try {
      await api.patch(`/incidents/duplicates/${matchId}/review`, {
        action,
        reviewed_by: 'admin',
      })
      toast.success(`Match ${action}`)
      fetchMatches()
    } catch (e) {
      toast.error(e.message)
    } finally {
      setReviewing(null)
    }
  }

  const getMatchTypeColor = (type) => ({
    exact: '#ef4444',
    probable: '#f97316',
    possible: '#eab308',
    manual_review: '#94a3b8',
  }[type] || '#94a3b8')

  return (
    <div className={styles.page}>
      <div className={styles.pageHeader}>
        <div>
          <h1 className={styles.title}>Duplicate Detection Review</h1>
          <p className={styles.subtitle}>
            {matches.length} {showReviewed ? 'reviewed' : 'pending'} duplicate matches
          </p>
        </div>
        <div className={styles.toggle}>
          <button
            className={`${styles.toggleBtn} ${!showReviewed ? styles.toggleActive : ''}`}
            onClick={() => setShowReviewed(false)}
          >
            Pending Review
          </button>
          <button
            className={`${styles.toggleBtn} ${showReviewed ? styles.toggleActive : ''}`}
            onClick={() => setShowReviewed(true)}
          >
            Reviewed
          </button>
        </div>
      </div>

      {/* Explanation */}
      <div className={styles.explainBanner}>
        <div className={styles.explainIcon}>ℹ️</div>
        <div>
          <div className={styles.explainTitle}>How Duplicate Detection Works</div>
          <div className={styles.explainDesc}>
            The system combines <strong>geographic proximity</strong> (Haversine distance) and <strong>visual similarity</strong>
            (perceptual hash + color feature cosine similarity) to identify potential duplicate complaints.
            Exact matches (&gt;85% combined score) are auto-merged. Possible/manual matches require human review.
          </div>
        </div>
      </div>

      {loading ? (
        <div className={styles.loading}>Loading...</div>
      ) : matches.length === 0 ? (
        <div className={styles.empty}>
          <div className={styles.emptyIcon}>✅</div>
          <div>No {showReviewed ? 'reviewed' : 'pending'} duplicate matches</div>
        </div>
      ) : (
        <div className={styles.matchList}>
          {matches.map((m) => (
            <div key={m.match_id} className={styles.matchCard}>
              <div className={styles.matchHeader}>
                <div className={styles.matchType}
                  style={{ background: getMatchTypeColor(m.match_type) + '22', color: getMatchTypeColor(m.match_type) }}>
                  {m.match_type?.toUpperCase().replace('_', ' ')}
                </div>
                <div className={styles.combinedScore}>
                  Combined: <span style={{ color: getMatchTypeColor(m.match_type) }}>{(m.combined_score * 100).toFixed(0)}%</span>
                </div>
              </div>

              <div className={styles.matchBody}>
                <div className={styles.matchSide}>
                  <div className={styles.sideLabel}>New Complaint</div>
                  <div className={styles.sideCode}>{m.complaint_id}</div>
                  <div className={styles.sideCategory}>{m.complaint_category}</div>
                </div>

                <div className={styles.matchScores}>
                  <div className={styles.scoreItem}>
                    <span className={styles.scoreLabel}>📍 Distance</span>
                    <span className={styles.scoreValue}>{m.geo_distance_km?.toFixed(2)} km</span>
                  </div>
                  <div className={styles.scoreItem}>
                    <span className={styles.scoreLabel}>🖼️ Visual Sim.</span>
                    <span className={styles.scoreValue}>
                      {m.visual_similarity !== null ? `${(m.visual_similarity * 100).toFixed(0)}%` : 'N/A'}
                    </span>
                  </div>
                  <div className={styles.scoreArrow}>↔</div>
                </div>

                <div className={styles.matchSide}>
                  <div className={styles.sideLabel}>Existing Incident</div>
                  <div className={styles.sideCode}>{m.incident_code}</div>
                  <div className={styles.sideTitle} title={m.incident_title}>{m.incident_title}</div>
                  <div className={`${styles.incStatus} ${m.incident_status === 'Resolved' ? styles.resolved : ''}`}>
                    {m.incident_status}
                  </div>
                </div>
              </div>

              {!showReviewed && (
                <div className={styles.matchActions}>
                  <button
                    className={styles.confirmBtn}
                    onClick={() => handleReview(m.match_id, 'confirmed')}
                    disabled={reviewing === m.match_id}
                  >
                    ✅ Confirm Duplicate
                  </button>
                  <button
                    className={styles.rejectBtn}
                    onClick={() => handleReview(m.match_id, 'rejected')}
                    disabled={reviewing === m.match_id}
                  >
                    ✕ Not a Duplicate
                  </button>
                </div>
              )}

              {showReviewed && (
                <div className={`${styles.reviewedBadge} ${m.review_action === 'confirmed' ? styles.badgeConfirmed : styles.badgeRejected}`}>
                  {m.review_action === 'confirmed' ? '✅ Confirmed' : '✕ Rejected'} by {m.reviewed_by}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

