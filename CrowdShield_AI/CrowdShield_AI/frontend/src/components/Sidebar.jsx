import { NavLink } from 'react-router-dom';
import styles from './Sidebar.module.css';

/**
 * Navigation entries for the sidebar.
 * Extend this array to add new sections without touching JSX.
 */
const NAV_ITEMS = [
  { label: 'Live Monitor', to: '/live-monitor', icon: '⊞' },
  { label: 'Alerts Table', to: '/alerts',       icon: '🔔' },
  { label: 'Analytics',    to: '/analytics',    icon: '📊' },
];

export default function Sidebar() {
  return (
    <aside className={styles.sidebar}>
      {/* Brand */}
      <div className={styles.brand}>
        <span className={styles.brandIcon}>🛡</span>
        <span className={styles.brandName}>CrowdShield</span>
        <span className={styles.brandTag}>AI</span>
      </div>

      {/* Navigation */}
      <nav className={styles.nav}>
        <p className={styles.sectionLabel}>MONITORING</p>
        {NAV_ITEMS.map(({ label, to, icon }) => (
          <NavLink
            key={to}
            to={to}
            className={({ isActive }) =>
              `${styles.navItem} ${isActive ? styles.active : ''}`
            }
          >
            <span className={styles.navIcon}>{icon}</span>
            <span>{label}</span>
          </NavLink>
        ))}
      </nav>

      {/* Footer */}
      <div className={styles.footer}>
        <span className={styles.version}>v0.1.0</span>
      </div>
    </aside>
  );
}
