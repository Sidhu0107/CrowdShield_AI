import { Routes, Route, Navigate } from 'react-router-dom';
import Layout from './components/Layout.jsx';
import LiveMonitor from './pages/LiveMonitor.jsx';
import AlertsTable from './pages/AlertsTable.jsx';
import Analytics from './pages/Analytics.jsx';

/**
 * Root router. All protected pages live under the shared Layout shell.
 * Add new <Route> entries here as pages are built out.
 */
export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Layout />}>
        <Route index element={<Navigate to="/live-monitor" replace />} />
        <Route path="dashboard" element={<Navigate to="/live-monitor" replace />} />
        <Route path="live-monitor" element={<LiveMonitor />} />
        <Route path="alerts" element={<AlertsTable />} />
        <Route path="analytics" element={<Analytics />} />
      </Route>
    </Routes>
  );
}
