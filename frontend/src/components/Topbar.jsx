import { useLocation } from 'react-router-dom';
import styles from './Topbar.module.css';

/** Maps route paths to human-readable page titles */
const PAGE_TITLES = {
  '/dashboard': 'Live Monitor',
  '/live-monitor': 'Live Monitor',
  '/alerts': 'Alerts Table',
  '/analytics': 'Analytics',
};

/**
 * Topbar — shows the current page title and a live system status badge.
 * The status badge will be wired to the WebSocket service in a later phase.
 */
export default function Topbar() {
  const { pathname } = useLocation();
  const title = PAGE_TITLES[pathname] ?? 'CrowdShield AI';

  return (
    <header className={styles.topbar}>
      <h1 className={styles.title}>{title}</h1>

      <div className={styles.right}>
        {/* System status indicator */}
        <div className={styles.statusBadge}>
          <span className={styles.dot} />
          <span className={styles.statusText}>System Online</span>
        </div>
      </div>
    </header>
  );
}
