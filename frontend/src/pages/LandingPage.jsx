import { Link } from 'react-router-dom'
import styles from './LandingPage.module.css'

export default function LandingPage() {
  return (
    <div className={styles.landing}>
      {/* Header */}
      <header className={styles.header}>
        <div className={styles.headerInner}>
          <div className={styles.logo}>
            <span className={styles.logoIcon}>🏛️</span>
            <div>
              <div className={styles.logoName}>SMCI</div>
              <div className={styles.logoSub}>Smart Municipal Intelligence</div>
            </div>
          </div>
          <nav className={styles.nav}>
            <Link to="/report" className={styles.navLink}>Report Issue</Link>
            <Link to="/track" className={styles.navLink}>Track Complaint</Link>
            <Link to="/admin" className={styles.navBtn}>Admin Dashboard</Link>
          </nav>
        </div>
      </header>

      {/* Hero */}
      <section className={styles.hero}>
        <div className={styles.heroContent}>
          <div className={styles.heroBadge}>
            <span>🤖</span> AI-Powered Civic Intelligence
          </div>
          <h1 className={styles.heroTitle}>
            Smart Municipal<br />
            <span className={styles.heroGradient}>Complaint Intelligence</span>
          </h1>
          <p className={styles.heroDesc}>
            Report civic issues with AI-assisted classification, geo-visual duplicate detection,
            and explainable priority scoring for faster municipal response.
          </p>
          <div className={styles.heroActions}>
            <Link to="/report" className={styles.btnPrimary}>
              <span>📸</span> Report an Issue
            </Link>
            <Link to="/track" className={styles.btnSecondary}>
              <span>🔍</span> Track Complaint
            </Link>
          </div>
        </div>
        <div className={styles.heroVisual}>
          <div className={styles.heroCard}>
            <div className={styles.heroCardHeader}>
              <span className={styles.cardBadge + ' ' + styles.critical}>● Critical</span>
              <span className={styles.cardCode}>INC-003</span>
            </div>
            <div className={styles.heroCardTitle}>Severe Waterlogging at Underpass</div>
            <div className={styles.heroCardMeta}>Silk Board Junction, Bangalore</div>
            <div className={styles.heroCardScore}>
              <div>Priority Score</div>
              <div className={styles.scoreValue}>9.1 / 10</div>
            </div>
            <div className={styles.scoreBar}>
              <div className={styles.scoreBarFill} style={{ width: '91%' }} />
            </div>
            <div className={styles.heroCardStats}>
              <div className={styles.stat}><span>📢</span> 12 Reports</div>
              <div className={styles.stat}><span>📍</span> Junction</div>
              <div className={styles.stat}><span>🤖</span> AI: 94%</div>
            </div>
          </div>
        </div>
      </section>

      {/* Features */}
      <section className={styles.features}>
        <div className={styles.sectionLabel}>Core Capabilities</div>
        <h2 className={styles.sectionTitle}>End-to-End Civic Issue Intelligence</h2>
        <div className={styles.featureGrid}>
          {FEATURES.map((f) => (
            <div key={f.title} className={styles.featureCard}>
              <div className={styles.featureIcon}>{f.icon}</div>
              <h3 className={styles.featureTitle}>{f.title}</h3>
              <p className={styles.featureDesc}>{f.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Stats */}
      <section className={styles.statsSection}>
        <div className={styles.statsGrid}>
          {STATS.map((s) => (
            <div key={s.label} className={styles.statCard}>
              <div className={styles.statNum}>{s.value}</div>
              <div className={styles.statLabel}>{s.label}</div>
            </div>
          ))}
        </div>
      </section>

      {/* CTA */}
      <section className={styles.cta}>
        <h2 className={styles.ctaTitle}>Ready to Report a Civic Issue?</h2>
        <p className={styles.ctaDesc}>
          Upload a photo, share your location, and let AI do the rest.
        </p>
        <Link to="/report" className={styles.btnPrimary}>
          Get Started — It's Free
        </Link>
      </section>

      {/* Footer */}
      <footer className={styles.footer}>
        <div className={styles.footerInner}>
          <div>© 2024 SMCI — Smart Municipal Complaint Intelligence</div>
          <div>Academic Project — Major Project Demonstration</div>
        </div>
      </footer>
    </div>
  )
}

const FEATURES = [
  {
    icon: '📸',
    title: 'AI Issue Classification',
    desc: 'Upload a photo and our ML pipeline identifies potholes, garbage, waterlogging, and more.',
  },
  {
    icon: '🗺️',
    title: 'Geo-Visual Analysis',
    desc: 'Interactive Leaflet maps with incident clustering, proximity detection, and location intelligence.',
  },
  {
    icon: '🔗',
    title: 'Duplicate Detection',
    desc: 'Multi-factor duplicate detection combining GPS proximity and visual similarity to prevent redundant incidents.',
  },
  {
    icon: '📊',
    title: 'Explainable Priority',
    desc: 'Transparent weighted scoring (Severity 35%, Impact 30%, Location 20%, History 15%) with human-readable explanations.',
  },
  {
    icon: '📋',
    title: 'Admin Dashboard',
    desc: 'Comprehensive municipal dashboard with analytics, status tracking, and resolution management.',
  },
  {
    icon: '🔔',
    title: 'Status Tracking',
    desc: 'Citizens can track their complaints through the entire lifecycle — from submission to resolution.',
  },
]

const STATS = [
  { value: '6+', label: 'Issue Categories' },
  { value: '< 2s', label: 'AI Analysis Time' },
  { value: '500m', label: 'Duplicate Radius' },
  { value: '3-Factor', label: 'Similarity Scoring' },
]

