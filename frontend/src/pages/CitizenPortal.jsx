import { useState, useRef, useCallback } from 'react'
import { Link } from 'react-router-dom'
import toast from 'react-hot-toast'
import api from '../utils/api'
import { CATEGORIES } from '../utils/constants'
import styles from './CitizenPortal.module.css'

const INITIAL_FORM = {
  reporter_name: '',
  reporter_email: '',
  reporter_phone: '',
  category: '',
  description: '',
  latitude: '',
  longitude: '',
  location_address: '',
}

export default function CitizenPortal() {
  const [form, setForm] = useState(INITIAL_FORM)
  const [image, setImage] = useState(null)
  const [imagePreview, setImagePreview] = useState(null)
  const [locationMode, setLocationMode] = useState('gps') // 'gps' | 'manual'
  const [gpsLoading, setGpsLoading] = useState(false)
  const [submitting, setSubmitting] = useState(false)
  const [result, setResult] = useState(null)
  const [dragOver, setDragOver] = useState(false)
  const fileInputRef = useRef()

  const handleChange = (e) => {
    setForm((prev) => ({ ...prev, [e.target.name]: e.target.value }))
  }

  const handleImageFile = (file) => {
    if (!file) return
    if (!file.type.startsWith('image/')) {
      toast.error('Please upload an image file')
      return
    }
    if (file.size > 16 * 1024 * 1024) {
      toast.error('Image must be under 16MB')
      return
    }
    setImage(file)
    const reader = new FileReader()
    reader.onload = (e) => setImagePreview(e.target.result)
    reader.readAsDataURL(file)
  }

  const handleDrop = (e) => {
    e.preventDefault()
    setDragOver(false)
    const file = e.dataTransfer.files[0]
    handleImageFile(file)
  }

  const getGPS = () => {
    if (!navigator.geolocation) {
      toast.error('Geolocation not supported by your browser')
      return
    }
    setGpsLoading(true)
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        setForm((prev) => ({
          ...prev,
          latitude: pos.coords.latitude.toFixed(6),
          longitude: pos.coords.longitude.toFixed(6),
        }))
        setGpsLoading(false)
        toast.success('GPS location captured!')
      },
      (err) => {
        setGpsLoading(false)
        toast.error(`GPS unavailable: ${err.message}. Use manual entry.`)
        setLocationMode('manual')
      },
      { enableHighAccuracy: true, timeout: 10000 }
    )
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!form.reporter_name) return toast.error('Name is required')
    if (!form.category) return toast.error('Category is required')
    if (!form.description) return toast.error('Description is required')
    if (!form.latitude || !form.longitude) return toast.error('Location is required')

    setSubmitting(true)
    const fd = new FormData()
    Object.entries(form).forEach(([k, v]) => fd.append(k, v))
    if (image) fd.append('image', image)

    try {
      const { data } = await api.post('/complaints', fd, {
        headers: { 'Content-Type': 'multipart/form-data' },
      })
      setResult(data)
      toast.success('Complaint submitted successfully!')
    } catch (err) {
      toast.error(err.message)
    } finally {
      setSubmitting(false)
    }
  }

  if (result) {
    return <SuccessScreen result={result} onNew={() => { setResult(null); setForm(INITIAL_FORM); setImage(null); setImagePreview(null) }} />
  }

  return (
    <div className={styles.page}>
      {/* Top bar */}
      <div className={styles.topbar}>
        <Link to="/" className={styles.backLink}>← Back to Home</Link>
        <div className={styles.topbarTitle}>🏛️ SMCI — Report Civic Issue</div>
        <Link to="/track" className={styles.trackLink}>Track Complaint →</Link>
      </div>

      <div className={styles.container}>
        <div className={styles.header}>
          <h1 className={styles.title}>Report a Civic Issue</h1>
          <p className={styles.subtitle}>
            Upload a photo, share your location, and our AI system will classify and prioritize your complaint.
          </p>
        </div>

        <form onSubmit={handleSubmit} className={styles.form}>
          <div className={styles.grid}>
            {/* Left column */}
            <div className={styles.col}>
              {/* Image Upload */}
              <section className={styles.section}>
                <label className={styles.sectionLabel}>📸 Issue Photograph</label>
                <div
                  className={`${styles.dropzone} ${dragOver ? styles.dropzoneActive : ''} ${imagePreview ? styles.dropzoneFilled : ''}`}
                  onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
                  onDragLeave={() => setDragOver(false)}
                  onDrop={handleDrop}
                  onClick={() => fileInputRef.current?.click()}
                >
                  {imagePreview ? (
                    <div className={styles.preview}>
                      <img src={imagePreview} alt="Preview" className={styles.previewImg} />
                      <div className={styles.previewOverlay}>
                        <span>Click to change image</span>
                      </div>
                    </div>
                  ) : (
                    <div className={styles.dropzoneContent}>
                      <div className={styles.uploadIcon}>📷</div>
                      <div className={styles.uploadText}>Drag & drop or click to upload</div>
                      <div className={styles.uploadSub}>PNG, JPG, WEBP · Max 16MB</div>
                      <div className={styles.uploadNote}>
                        📌 Photo helps AI identify the issue type and detect duplicates
                      </div>
                    </div>
                  )}
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept="image/*"
                    style={{ display: 'none' }}
                    onChange={(e) => handleImageFile(e.target.files[0])}
                  />
                </div>
                {image && (
                  <div className={styles.imageInfo}>
                    <span>✅ {image.name}</span>
                    <button type="button" className={styles.removeImg} onClick={() => { setImage(null); setImagePreview(null) }}>✕</button>
                  </div>
                )}
              </section>

              {/* Location */}
              <section className={styles.section}>
                <label className={styles.sectionLabel}>📍 Location</label>
                <div className={styles.locationTabs}>
                  <button
                    type="button"
                    className={`${styles.tab} ${locationMode === 'gps' ? styles.tabActive : ''}`}
                    onClick={() => setLocationMode('gps')}
                  >
                    🛰️ Use GPS
                  </button>
                  <button
                    type="button"
                    className={`${styles.tab} ${locationMode === 'manual' ? styles.tabActive : ''}`}
                    onClick={() => setLocationMode('manual')}
                  >
                    ✏️ Enter Manually
                  </button>
                </div>

                {locationMode === 'gps' ? (
                  <div>
                    <button
                      type="button"
                      className={styles.gpsBtn}
                      onClick={getGPS}
                      disabled={gpsLoading}
                    >
                      {gpsLoading ? '📡 Getting location...' : '📡 Capture GPS Coordinates'}
                    </button>
                    {form.latitude && (
                      <div className={styles.coordsDisplay}>
                        <span>✅ Lat: {form.latitude}</span>
                        <span>Lng: {form.longitude}</span>
                      </div>
                    )}
                  </div>
                ) : (
                  <div className={styles.coordsGrid}>
                    <div>
                      <label className={styles.fieldLabel}>Latitude *</label>
                      <input
                        type="number"
                        step="any"
                        name="latitude"
                        placeholder="e.g. 12.9716"
                        value={form.latitude}
                        onChange={handleChange}
                        className={styles.input}
                        required
                      />
                    </div>
                    <div>
                      <label className={styles.fieldLabel}>Longitude *</label>
                      <input
                        type="number"
                        step="any"
                        name="longitude"
                        placeholder="e.g. 77.5946"
                        value={form.longitude}
                        onChange={handleChange}
                        className={styles.input}
                        required
                      />
                    </div>
                  </div>
                )}
                <div className={styles.mt}>
                  <label className={styles.fieldLabel}>Address / Landmark (optional)</label>
                  <input
                    type="text"
                    name="location_address"
                    placeholder="e.g. Near Bus Stop, MG Road, Bangalore"
                    value={form.location_address}
                    onChange={handleChange}
                    className={styles.input}
                  />
                </div>
              </section>
            </div>

            {/* Right column */}
            <div className={styles.col}>
              {/* Reporter Info */}
              <section className={styles.section}>
                <label className={styles.sectionLabel}>👤 Your Information</label>
                <div className={styles.fieldGroup}>
                  <div>
                    <label className={styles.fieldLabel}>Full Name *</label>
                    <input
                      type="text"
                      name="reporter_name"
                      placeholder="Your full name"
                      value={form.reporter_name}
                      onChange={handleChange}
                      className={styles.input}
                      required
                    />
                  </div>
                  <div>
                    <label className={styles.fieldLabel}>Email (optional)</label>
                    <input
                      type="email"
                      name="reporter_email"
                      placeholder="For status updates"
                      value={form.reporter_email}
                      onChange={handleChange}
                      className={styles.input}
                    />
                  </div>
                  <div>
                    <label className={styles.fieldLabel}>Phone (optional)</label>
                    <input
                      type="tel"
                      name="reporter_phone"
                      placeholder="Contact number"
                      value={form.reporter_phone}
                      onChange={handleChange}
                      className={styles.input}
                    />
                  </div>
                </div>
              </section>

              {/* Issue Details */}
              <section className={styles.section}>
                <label className={styles.sectionLabel}>🔍 Issue Details</label>
                <div className={styles.fieldGroup}>
                  <div>
                    <label className={styles.fieldLabel}>Issue Category *</label>
                    <select
                      name="category"
                      value={form.category}
                      onChange={handleChange}
                      className={styles.select}
                      required
                    >
                      <option value="">Select issue type...</option>
                      {CATEGORIES.map((c) => (
                        <option key={c} value={c}>{c}</option>
                      ))}
                    </select>
                  </div>
                  <div>
                    <label className={styles.fieldLabel}>Description *</label>
                    <textarea
                      name="description"
                      placeholder="Describe the issue in detail. Include what, where, since when, and impact on residents..."
                      value={form.description}
                      onChange={handleChange}
                      className={styles.textarea}
                      rows={5}
                      required
                    />
                    <div className={styles.charCount}>{form.description.length} chars</div>
                  </div>
                </div>
              </section>

              {/* ML Info Banner */}
              <div className={styles.mlBanner}>
                <div className={styles.mlIcon}>🤖</div>
                <div>
                  <div className={styles.mlTitle}>AI Analysis on Submission</div>
                  <div className={styles.mlDesc}>
                    Your image will be analyzed to identify the issue type and check for existing similar reports nearby.
                  </div>
                </div>
              </div>

              {/* Submit */}
              <button
                type="submit"
                className={styles.submitBtn}
                disabled={submitting}
              >
                {submitting ? (
                  <><span className={styles.spinner} /> Analyzing & Submitting...</>
                ) : (
                  '🚀 Submit Complaint'
                )}
              </button>
            </div>
          </div>
        </form>
      </div>
    </div>
  )
}

function SuccessScreen({ result, onNew }) {
  const ml = result.ml_result || {}
  const reported  = result.reported_category
  const predicted = result.predicted_category
  const mismatch  = result.category_mismatch

  return (
    <div className={styles.page}>
      <div className={styles.topbar}>
        <Link to="/" className={styles.backLink}>← Home</Link>
        <div className={styles.topbarTitle}>🏛️ SMCI</div>
        <Link to="/track" className={styles.trackLink}>Track Complaint →</Link>
      </div>
      <div className={styles.successContainer}>
        <div className={styles.successCard}>
          <div className={styles.successIcon}>✅</div>
          <h2 className={styles.successTitle}>Complaint Submitted!</h2>
          <div className={styles.successId}>
            <span>Complaint ID</span>
            <span className={styles.idCode}>{result.complaint_id}</span>
          </div>
          {result.incident_code && (
            <div className={styles.successId}>
              <span>Incident Code</span>
              <span className={styles.idCode}>{result.incident_code}</span>
            </div>
          )}
          {result.is_duplicate && (
            <div className={styles.dupeBanner}>
              🔗 Linked to existing incident — complaint count updated
            </div>
          )}

          {/* Category Analysis Panel */}
          <div className={styles.categoryPanel}>
            <div className={styles.categoryPanelTitle}>📋 Category Analysis</div>
            <div className={styles.categoryRows}>
              <div className={styles.categoryRow}>
                <span className={styles.categoryRowLabel}>👤 You reported as</span>
                <span className={styles.categoryRowValue}>{reported || '—'}</span>
              </div>
              {predicted && (
                <div className={styles.categoryRow}>
                  <span className={styles.categoryRowLabel}>🤖 AI identified as</span>
                  <span className={`${styles.categoryRowValue} ${styles.aiPredicted}`}>
                    {predicted}
                    {ml.confidence && (
                      <span className={styles.confBadge}>{(ml.confidence * 100).toFixed(1)}%</span>
                    )}
                  </span>
                </div>
              )}
              {mismatch && (
                <div className={styles.mismatchBanner}>
                  ⚠️ <strong>Category Mismatch</strong> — The AI identified this as{' '}
                  <strong>{predicted}</strong>, but you selected <strong>{reported}</strong>.{' '}
                  The system will process it as <strong>{result.authoritative_category}</strong>.
                </div>
              )}
              {!mismatch && predicted && (
                <div className={styles.matchBanner}>
                  ✅ AI prediction matches your selection
                </div>
              )}
            </div>
            {ml.model_type && (
              <div className={styles.mlModelNote}>
                Model: {ml.model_type}
                {ml.dataset_label && ` · Dataset label: ${ml.dataset_label}`}
              </div>
            )}
          </div>

          <p className={styles.successMsg}>{result.message}</p>
          <div className={styles.successActions}>
            <Link to={`/track?id=${result.complaint_id}`} className={styles.trackBtn}>
              🔍 Track This Complaint
            </Link>
            <button onClick={onNew} className={styles.newBtn}>
              + Report Another Issue
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

