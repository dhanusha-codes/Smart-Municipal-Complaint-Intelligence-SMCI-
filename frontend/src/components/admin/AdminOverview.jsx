import { useEffect, useState } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, LineChart, Line, Legend
} from 'recharts'
import api from '../../utils/api'
import { PRIORITY_COLORS, STATUS_COLORS } from '../../utils/constants'
import styles from './AdminOverview.module.css'

export default function AdminOverview() {
  const [stats, setStats] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetchStats()
    const interval = setInterval(fetchStats, 30000)
    return () => clearInterval(interval)
  }, [])

  const fetchStats = async () => {
    try {
      const { data } = await api.get('/admin/stats')
      setStats(data)
    } catch (e) {
      console.error(e)
    } finally {
      setLoading(false)
    }
  }

  if (loading) return <div className={styles.loading}>Loading dashboard...</div>
  if (!stats) return <div className={styles.loading}>Failed to load stats</div>

  const priorityData = [
    { name: 'Critical', value: stats.critical_incidents, color: '#ef4444' },
    { name: 'Urgent', value: stats.urgent_incidents, color: '#f97316' },
    { name: 'Normal', value: stats.normal_incidents, color: '#22c55e' },
  ]

  return (
    <div className={styles.overview}>
      <div className={styles.pageHeader}>
        <div>
          <h1 className={styles.pageTitle}>Dashboard Overview</h1>
          <p className={styles.pageSubtitle}>Real-time municipal complaint intelligence</p>
        </div>
      </div>

      {/* KPI Cards */}
      <div className={styles.kpiGrid}>
        <KPICard icon="📋" label="Total Complaints" value={stats.total_complaints} color="#3b82f6" />
        <KPICard icon="🚨" label="Total Incidents" value={stats.total_incidents} color="#8b5cf6" />
        <KPICard icon="🔴" label="Critical" value={stats.critical_incidents} color="#ef4444" />
        <KPICard icon="🟠" label="Urgent" value={stats.urgent_incidents} color="#f97316" />
        <KPICard icon="✅" label="Resolved" value={stats.resolved_incidents} color="#22c55e" />
        <KPICard icon="🔗" label="Pending Reviews" value={stats.pending_duplicate_reviews} color="#14b8a6" />
      </div>

      {/* Charts Row */}
      <div className={styles.chartsGrid}>
        {/* Category Bar Chart */}
        <div className={styles.chartCard}>
          <div className={styles.chartTitle}>Issues by Category</div>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={stats.category_breakdown} margin={{ top: 5, right: 10, bottom: 40, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.08)" />
              <XAxis
                dataKey="category"
                tick={{ fill: '#64748b', fontSize: 10 }}
                angle={-30}
                textAnchor="end"
                interval={0}
              />
              <YAxis tick={{ fill: '#64748b', fontSize: 11 }} />
              <Tooltip
                contentStyle={{ background: '#1e293b', border: '1px solid rgba(148,163,184,0.15)', borderRadius: 8, color: '#f8fafc' }}
                cursor={{ fill: 'rgba(59,130,246,0.08)' }}
              />
              <Bar dataKey="count" fill="#3b82f6" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Priority Pie */}
        <div className={styles.chartCard}>
          <div className={styles.chartTitle}>Priority Distribution</div>
          <ResponsiveContainer width="100%" height={220}>
            <PieChart>
              <Pie data={priorityData} cx="50%" cy="50%" innerRadius={55} outerRadius={85} paddingAngle={4} dataKey="value">
                {priorityData.map((entry) => (
                  <Cell key={entry.name} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid rgba(148,163,184,0.15)', borderRadius: 8, color: '#f8fafc' }} />
              <Legend
                formatter={(value) => <span style={{ fontSize: '0.8rem', color: '#94a3b8' }}>{value}</span>}
              />
            </PieChart>
          </ResponsiveContainer>
        </div>

        {/* Status Breakdown */}
        <div className={styles.chartCard}>
          <div className={styles.chartTitle}>Incident Status</div>
          <div className={styles.statusList}>
            {stats.status_breakdown.map((s) => (
              <div key={s.status} className={styles.statusRow}>
                <div className={styles.statusDot} style={{ background: STATUS_COLORS[s.status] || '#94a3b8' }} />
                <span className={styles.statusName}>{s.status}</span>
                <span className={styles.statusCount}>{s.count}</span>
                <div className={styles.statusBarOuter}>
                  <div
                    className={styles.statusBarInner}
                    style={{
                      width: `${Math.round((s.count / stats.total_incidents) * 100)}%`,
                      background: STATUS_COLORS[s.status] || '#94a3b8',
                    }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Trend Chart */}
      <div className={styles.trendCard}>
        <div className={styles.chartTitle}>Complaint Trend — Last 30 Days</div>
        <ResponsiveContainer width="100%" height={200}>
          <LineChart data={stats.complaint_trend} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.08)" />
            <XAxis
              dataKey="date"
              tick={{ fill: '#64748b', fontSize: 10 }}
              tickFormatter={(v) => v.slice(5)}
              interval={4}
            />
            <YAxis tick={{ fill: '#64748b', fontSize: 11 }} allowDecimals={false} />
            <Tooltip
              contentStyle={{ background: '#1e293b', border: '1px solid rgba(148,163,184,0.15)', borderRadius: 8, color: '#f8fafc' }}
              labelFormatter={(l) => `Date: ${l}`}
            />
            <Line type="monotone" dataKey="count" stroke="#14b8a6" strokeWidth={2} dot={false} name="Complaints" />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}

function KPICard({ icon, label, value, color }) {
  return (
    <div className={styles.kpiCard}>
      <div className={styles.kpiIcon} style={{ background: color + '22', color }}>{icon}</div>
      <div className={styles.kpiBody}>
        <div className={styles.kpiValue} style={{ color }}>{value}</div>
        <div className={styles.kpiLabel}>{label}</div>
      </div>
    </div>
  )
}

