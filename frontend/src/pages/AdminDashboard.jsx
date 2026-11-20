import { useState, useEffect } from 'react'
import { Routes, Route, Link, NavLink, useNavigate } from 'react-router-dom'
import AdminOverview from '../components/admin/AdminOverview'
import AdminIncidents from '../components/admin/AdminIncidents'
import AdminComplaints from '../components/admin/AdminComplaints'
import AdminMap from '../components/admin/AdminMap'
import AdminDuplicates from '../components/admin/AdminDuplicates'
import AdminHotspots from '../components/admin/AdminHotspots'
import styles from './AdminDashboard.module.css'

const NAV_ITEMS = [
  { path: '/admin', label: 'Overview', icon: '📊', end: true },
  { path: '/admin/incidents', label: 'Incidents', icon: '🚨' },
  { path: '/admin/complaints', label: 'Complaints', icon: '📋' },
  { path: '/admin/map', label: 'Geo Map', icon: '🗺️' },
  { path: '/admin/duplicates', label: 'Duplicates', icon: '🔗' },
  { path: '/admin/hotspots', label: 'Hotspots', icon: '🔥' },
]

export default function AdminDashboard() {
  const [sidebarOpen, setSidebarOpen] = useState(true)

  return (
    <div className={styles.layout}>
      {/* Sidebar */}
      <aside className={`${styles.sidebar} ${sidebarOpen ? '' : styles.sidebarCollapsed}`}>
        <div className={styles.sidebarHeader}>
          <div className={styles.logo}>
            <span className={styles.logoIcon}>🏛️</span>
            {sidebarOpen && (
              <div>
                <div className={styles.logoName}>SMCI Admin</div>
                <div className={styles.logoSub}>Municipal Dashboard</div>
              </div>
            )}
          </div>
          <button className={styles.collapseBtn} onClick={() => setSidebarOpen((v) => !v)}>
            {sidebarOpen ? '◀' : '▶'}
          </button>
        </div>

        <nav className={styles.nav}>
          {NAV_ITEMS.map((item) => (
            <NavLink
              key={item.path}
              to={item.path}
              end={item.end}
              className={({ isActive }) => `${styles.navItem} ${isActive ? styles.navActive : ''}`}
            >
              <span className={styles.navIcon}>{item.icon}</span>
              {sidebarOpen && <span className={styles.navLabel}>{item.label}</span>}
            </NavLink>
          ))}
        </nav>

        {sidebarOpen && (
          <div className={styles.sidebarFooter}>
            <Link to="/" className={styles.footerLink}>← Citizen Portal</Link>
          </div>
        )}
      </aside>

      {/* Main content */}
      <main className={styles.main}>
        <div className={styles.topbar}>
          <div className={styles.topbarTitle}>
            Smart Municipal Complaint Intelligence
          </div>
          <div className={styles.topbarRight}>
            <span className={styles.liveIndicator}>● Live</span>
            <span className={styles.adminBadge}>Admin</span>
          </div>
        </div>

        <div className={styles.content}>
          <Routes>
            <Route index element={<AdminOverview />} />
            <Route path="incidents" element={<AdminIncidents />} />
            <Route path="complaints" element={<AdminComplaints />} />
            <Route path="map" element={<AdminMap />} />
            <Route path="duplicates" element={<AdminDuplicates />} />
            <Route path="hotspots" element={<AdminHotspots />} />
          </Routes>
        </div>
      </main>
    </div>
  )
}

