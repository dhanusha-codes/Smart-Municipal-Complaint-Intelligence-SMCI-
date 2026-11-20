import { useState, useEffect } from 'react'
import toast from 'react-hot-toast'
import api from '../../utils/api'
import { CATEGORY_ICONS, CATEGORIES, STATUSES } from '../../utils/constants'
import styles from './AdminComplaints.module.css'

export default function AdminComplaints() {
  const [complaints, setComplaints] = useState([])
  const [total, setTotal] = useState(0)
  const [page, setPage] = useState(1)
  const [pages, setPages] = useState(1)
  const [loading, setLoading] = useState(true)
  const [filters, setFilters] = useState({ status: '', category: '' })

  useEffect(() => { fetchComplaints() }, [page, filters])

  const fetchComplaints = async () => {
    setLoading(true)
    try {
      const { data } = await api.get('/admin/complaints', { params: { page, per_page: 10, ...filters } })
      setComplaints(data.complaints)
      setTotal(data.total)
      setPages(data.pages)
    } catch {
      toast.error('Failed to load complaints')
    } finally {
      setLoading(false)
    }
  }

  const handleDeleteComplaint = async (id, complaintId) => {
    if (!window.confirm(`Are you sure you want to permanently remove complaint #${complaintId}?`)) {
      return
    }
    try {
      await api.delete(`/complaints/${id}`)
      toast.success(`Complaint #${complaintId} removed`)
      setComplaints((prev) => prev.filter((c) => c.id !== id))
      setTotal((t) => Math.max(0, t - 1))
    } catch (err) {
      toast.error(err.message || 'Failed to remove complaint')
    }
  }

  return (
    <div className={styles.page}>
      <div className={styles.pageHeader}>
        <h1 className={styles.title}>Complaints</h1>
        <p className={styles.subtitle}>{total} total citizen complaints</p>
      </div>

      <div className={styles.filtersBar}>
        <select className={styles.filterSelect} value={filters.status}
          onChange={(e) => { setFilters((f) => ({ ...f, status: e.target.value })); setPage(1) }}>
          <option value="">All Statuses</option>
          {STATUSES.map((s) => <option key={s}>{s}</option>)}
        </select>
        <select className={styles.filterSelect} value={filters.category}
          onChange={(e) => { setFilters((f) => ({ ...f, category: e.target.value })); setPage(1) }}>
          <option value="">All Categories</option>
          {CATEGORIES.map((c) => <option key={c}>{c}</option>)}
        </select>
      </div>

      <div className={styles.tableContainer}>
        <table className={styles.table}>
          <thead>
            <tr>
              <th>Complaint ID</th>
              <th>Reporter</th>
              <th>Category</th>
              <th>Location</th>
              <th>AI Analysis</th>
              <th>Duplicate</th>
              <th>Submitted</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {loading ? (
              <tr><td colSpan={8} className={styles.loadingCell}>Loading...</td></tr>
            ) : complaints.length === 0 ? (
              <tr><td colSpan={8} className={styles.loadingCell}>No complaints found</td></tr>
            ) : complaints.map((c) => (
              <tr key={c.id} className={styles.row}>
                <td><span className={styles.cid}>{c.complaint_id}</span></td>
                <td>
                  <div className={styles.reporter}>{c.reporter_name}</div>
                  {c.reporter_email && <div className={styles.reporterEmail}>{c.reporter_email}</div>}
                </td>
                <td>
                  <div className={styles.categoryMain}>
                    <span className={styles.category}>{CATEGORY_ICONS[c.category]} {c.category}</span>
                  </div>
                  {c.category_mismatch ? (
                    <div className={styles.mismatchTag} title={`Reported: ${c.reported_category} | Predicted: ${c.predicted_category || c.ml_category}`}>
                      ⚠️ User: {c.reported_category}
                    </div>
                  ) : (
                    c.reported_category && (
                      <div className={styles.reportedTag}>
                        User: {c.reported_category}
                      </div>
                    )
                  )}
                </td>
                <td>
                  <div className={styles.location} title={c.location_address || ''}>
                    {c.location_address || `${c.latitude?.toFixed(4)}, ${c.longitude?.toFixed(4)}`}
                  </div>
                </td>
                <td>
                  {c.ml_category ? (
                    <div>
                      <div className={styles.mlCat}>
                        {c.predicted_category || c.ml_category}
                      </div>
                      <div className={styles.mlConf}>
                        {(c.ml_confidence * 100).toFixed(0)}% {c.ml_status === 'heuristic' ? '⚠️' : '🤖'}
                        {c.ml_dataset_label && (
                          <span className={styles.datasetLabel} title={`Dataset class: ${c.ml_dataset_label}`}>
                            {' '}({c.ml_dataset_label})
                          </span>
                        )}
                      </div>
                    </div>
                  ) : <span className={styles.naText}>No image</span>}
                </td>
                <td>
                  {c.is_duplicate ? (
                    <span className={styles.dupYes}>🔗 Merged</span>
                  ) : (
                    <span className={styles.dupNo}>New</span>
                  )}
                </td>
                <td><span className={styles.date}>{c.submitted_at ? new Date(c.submitted_at).toLocaleDateString('en-IN') : '—'}</span></td>
                <td>
                  <button
                    className={styles.deleteBtn}
                    onClick={() => handleDeleteComplaint(c.id, c.complaint_id)}
                    title="Permanently remove complaint"
                  >
                    🗑️
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className={styles.pagination}>
        <button disabled={page === 1} onClick={() => setPage((p) => p - 1)} className={styles.pageBtn}>← Prev</button>
        <span className={styles.pageInfo}>Page {page} of {pages} · {total} total</span>
        <button disabled={page === pages} onClick={() => setPage((p) => p + 1)} className={styles.pageBtn}>Next →</button>
      </div>
    </div>
  )
}

