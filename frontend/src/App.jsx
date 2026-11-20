import { Routes, Route } from 'react-router-dom'
import { Toaster } from 'react-hot-toast'
import CitizenPortal from './pages/CitizenPortal'
import TrackComplaint from './pages/TrackComplaint'
import AdminDashboard from './pages/AdminDashboard'
import LandingPage from './pages/LandingPage'

export default function App() {
  return (
    <>
      <Toaster
        position="top-right"
        toastOptions={{
          style: {
            background: '#1e293b',
            color: '#f8fafc',
            border: '1px solid rgba(148,163,184,0.15)',
            borderRadius: '10px',
          },
        }}
      />
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/report" element={<CitizenPortal />} />
        <Route path="/track" element={<TrackComplaint />} />
        <Route path="/admin/*" element={<AdminDashboard />} />
      </Routes>
    </>
  )
}
