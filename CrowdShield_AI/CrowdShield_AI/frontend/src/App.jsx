import { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { Routes, Route, Navigate, NavLink, useLocation, Link } from 'react-router-dom';
import {
  Shield,
  Video,
  Upload,
  Table2,
  BarChart2,
  FileText,
  Settings,
  ChevronLeft,
  ChevronRight,
  Bell,
  Cpu,
  HardDrive,
  CheckCheck,
  BellOff,
} from 'lucide-react';
import DashboardPage from './pages/DashboardPage.jsx';
import LiveMonitorPage from './pages/LiveMonitorPage.jsx';
import AnalyzeVideoPage from './pages/AnalyzeVideoPage.jsx';
import AlertsPage from './pages/AlertsPage.jsx';
import AnalyticsPage from './pages/AnalyticsPage.jsx';
import ReportsPage from './pages/ReportsPage.jsx';
import ConfigurationPage from './pages/ConfigurationPage.jsx';
import { getHealth, listLiveEvents } from './services/api.js';

const NAV_ITEMS = [
  { path: '/dashboard', label: 'Dashboard', icon: Shield },
  { path: '/monitor', label: 'Live Monitor', icon: Video },
  { path: '/analyze', label: 'Analyze Video', icon: Upload },
  { path: '/alerts', label: 'Alerts Table', icon: Table2 },
  { path: '/analytics', label: 'Analytics', icon: BarChart2 },
  { path: '/reports', label: 'Reports', icon: FileText },
  { path: '/config', label: 'Configuration', icon: Settings },
];

const PAGE_TITLES = {
  '/dashboard': 'Dashboard',
  '/monitor': 'Live Monitor',
  '/analyze': 'Analyze Video',
  '/alerts': 'Alerts Table',
  '/analytics': 'Analytics',
  '/reports': 'Reports',
  '/config': 'Configuration',
};

function useClock() {
  const [time, setTime] = useState(new Date());

  useEffect(() => {
    const interval = setInterval(() => setTime(new Date()), 1000);
    return () => clearInterval(interval);
  }, []);

  return time.toLocaleTimeString('en-US', { hour12: false });
}

function Sidebar({ collapsed, onToggle }) {
  return (
    <aside className={`sidebar ${collapsed ? 'sidebar--collapsed' : ''}`}>
      <div className="sidebar__logo">
        <div className="sidebar__logo-icon">
          <Shield size={22} strokeWidth={2.5} />
        </div>
        {!collapsed && <span className="sidebar__logo-text">CrowdShield</span>}
      </div>

      <nav className="sidebar__nav">
        {NAV_ITEMS.map(({ path, label, icon: Icon }) => (
          <NavLink
            key={path}
            to={path}
            className={({ isActive }) =>
              `sidebar__link ${isActive ? 'sidebar__link--active' : ''}`
            }
            title={collapsed ? label : undefined}
          >
            <Icon size={18} />
            {!collapsed && <span>{label}</span>}
          </NavLink>
        ))}
      </nav>

      <div className="sidebar__footer">
        {!collapsed && (
          <div className="sidebar__status">
            <span className="sidebar__status-dot pulse-green" />
            <span className="text-secondary" style={{ fontSize: '0.75rem' }}>
              System Online
            </span>
          </div>
        )}
        {!collapsed && <div className="sidebar__version">v2.0</div>}
        <button className="sidebar__toggle" onClick={onToggle} title="Toggle sidebar">
          {collapsed ? <ChevronRight size={16} /> : <ChevronLeft size={16} />}
        </button>
      </div>
    </aside>
  );
}

function Header() {
  const location = useLocation();
  const clock = useClock();
  const pageTitle = PAGE_TITLES[location.pathname] || 'CrowdShield AI';
  const [isNotifOpen, setIsNotifOpen] = useState(false);
  const [notifFilter, setNotifFilter] = useState('unread');
  const [notifications, setNotifications] = useState([]);
  const [isMuted, setIsMuted] = useState(false);
  const [health, setHealth] = useState({
    live_status: 'stopped',
    analysis_status: 'idle',
    events_buffer: 0,
  });
  const notifRef = useRef(null);

  const unreadCount = notifications.filter((item) => !item.read).length;

  const visibleNotifications = useMemo(() => {
    if (notifFilter === 'all') {
      return notifications;
    }
    return notifications.filter((item) => !item.read);
  }, [notifFilter, notifications]);

  useEffect(() => {
    if (!isNotifOpen) {
      return undefined;
    }

    const handlePointerDown = (event) => {
      if (notifRef.current && !notifRef.current.contains(event.target)) {
        setIsNotifOpen(false);
      }
    };

    window.addEventListener('pointerdown', handlePointerDown);
    return () => window.removeEventListener('pointerdown', handlePointerDown);
  }, [isNotifOpen]);

  useEffect(() => {
    let mounted = true;

    const syncHealth = async () => {
      try {
        const data = await getHealth();
        if (mounted && data) {
          setHealth(data);
        }
      } catch {
        // Preserve last health snapshot if API is temporarily unavailable.
      }
    };

    syncHealth();
    const interval = setInterval(syncHealth, 5000);

    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, []);

  useEffect(() => {
    let mounted = true;

    const syncNotifications = async () => {
      try {
        const events = await listLiveEvents();
        if (!mounted || isMuted) return;

        const next = events
          .slice()
          .reverse()
          .slice(0, 20)
          .map((event, index) => {
            const sev = event.severity === 'RED' ? 'critical' : event.severity === 'YELLOW' ? 'warn' : 'info';
            return {
              id: `${event.id}-${index}`,
              title: `${event.detector} alert`,
              message: event.description,
              time: event.timestamp,
              severity: sev,
              read: false,
              link: '/alerts',
            };
          });

        setNotifications((previous) => {
          const readByMessage = new Map(
            previous.map((item) => [`${item.title}|${item.message}|${item.time}`, item.read])
          );

          return next.map((item) => ({
            ...item,
            read: readByMessage.get(`${item.title}|${item.message}|${item.time}`) ?? false,
          }));
        });
      } catch {
        // Keep current notifications on polling errors.
      }
    };

    syncNotifications();
    const interval = setInterval(syncNotifications, 3000);

    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, [isMuted]);

  const markAllRead = useCallback(() => {
    setNotifications((prev) => prev.map((item) => ({ ...item, read: true })));
  }, []);

  const clearRead = useCallback(() => {
    setNotifications((prev) => prev.filter((item) => !item.read));
  }, []);

  const markAsRead = useCallback((id) => {
    setNotifications((prev) =>
      prev.map((item) => (item.id === id ? { ...item, read: true } : item))
    );
  }, []);

  return (
    <header className="header">
      <div className="header__left">
        <h1 className="header__title">{pageTitle}</h1>
      </div>
      <div className="header__right">
        <div className="header__chip">
          <Cpu size={13} />
          <span>Live: {health.live_status || 'unknown'}</span>
        </div>
        <div className="header__chip">
          <HardDrive size={13} />
          <span>Events: {health.events_buffer ?? 0}</span>
        </div>

        <div className="header__clock">{clock}</div>

        <div className="header__notifications" ref={notifRef}>
          <button
            className={`header__bell ${isNotifOpen ? 'header__bell--active' : ''}`}
            onClick={() => setIsNotifOpen((prev) => !prev)}
            aria-label="Open notifications"
          >
            <Bell size={18} />
            {unreadCount > 0 && (
              <span className="header__bell-badge">
                {unreadCount > 99 ? '99+' : unreadCount}
              </span>
            )}
          </button>

          {isNotifOpen && (
            <div className="header__notif-menu card">
              <div className="header__notif-head">
                <div>
                  <h3>Notifications</h3>
                  <span>{unreadCount} unread</span>
                </div>
                <button className="header__notif-link" onClick={markAllRead}>
                  <CheckCheck size={14} /> Mark all read
                </button>
              </div>

              <div className="header__notif-actions">
                <div className="header__notif-toggle-group">
                  <button
                    className={`header__notif-toggle ${notifFilter === 'unread' ? 'header__notif-toggle--active' : ''}`}
                    onClick={() => setNotifFilter('unread')}
                  >
                    Unread
                  </button>
                  <button
                    className={`header__notif-toggle ${notifFilter === 'all' ? 'header__notif-toggle--active' : ''}`}
                    onClick={() => setNotifFilter('all')}
                  >
                    All
                  </button>
                </div>

                <button className="header__notif-link" onClick={() => setIsMuted((prev) => !prev)}>
                  <BellOff size={14} /> {isMuted ? 'Unmute' : 'Mute 30m'}
                </button>
                <button className="header__notif-link" onClick={clearRead}>
                  Clear read
                </button>
              </div>

              <div className="header__notif-list">
                {visibleNotifications.length > 0 ? (
                  visibleNotifications.map((item) => (
                    <Link
                      key={item.id}
                      to={item.link}
                      className={`header__notif-item ${item.read ? '' : 'header__notif-item--unread'}`}
                      onClick={() => {
                        markAsRead(item.id);
                        setIsNotifOpen(false);
                      }}
                    >
                      <span className={`header__notif-dot header__notif-dot--${item.severity}`} />
                      <div className="header__notif-copy">
                        <div className="header__notif-title-row">
                          <strong>{item.title}</strong>
                          <time>{item.time}</time>
                        </div>
                        <p>{item.message}</p>
                      </div>
                    </Link>
                  ))
                ) : (
                  <div className="header__notif-empty">No notifications for this filter.</div>
                )}
              </div>

              <div className="header__notif-footer">
                <Link to="/alerts" onClick={() => setIsNotifOpen(false)}>
                  View alerts table
                </Link>
              </div>
            </div>
          )}
        </div>
      </div>
    </header>
  );
}

export default function App() {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const toggleSidebar = useCallback(() => setSidebarCollapsed((prev) => !prev), []);

  return (
    <div className="app-shell">
      <Sidebar collapsed={sidebarCollapsed} onToggle={toggleSidebar} />
      <div
        className="app-main"
        style={{
          marginLeft: sidebarCollapsed ? 'var(--sidebar-collapsed)' : 'var(--sidebar-width)',
        }}
      >
        <Header />
        <main className="app-content">
          <Routes>
            <Route path="/" element={<Navigate to="/dashboard" replace />} />
            <Route path="/dashboard" element={<DashboardPage />} />
            <Route path="/monitor" element={<LiveMonitorPage />} />
            <Route path="/analyze" element={<AnalyzeVideoPage />} />
            <Route path="/alerts" element={<AlertsPage />} />
            <Route path="/analytics" element={<AnalyticsPage />} />
            <Route path="/reports" element={<ReportsPage />} />
            <Route path="/config" element={<ConfigurationPage />} />
          </Routes>
        </main>
      </div>
    </div>
  );
}
