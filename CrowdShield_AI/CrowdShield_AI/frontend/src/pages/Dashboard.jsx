import styles from './Dashboard.module.css';

/**
 * Stat card for the summary row.
 */
function StatCard({ label, value, color }) {
  return (
    <div className={styles.statCard}>
      <span className={styles.statLabel}>{label}</span>
      <span className={styles.statValue} style={{ color }}>{value}</span>
    </div>
  );
}

/**
 * Dashboard page — initial display scaffold.
 * Data will be wired via services/api.js and services/websocket.js in a later phase.
 */
export default function Dashboard() {
  return (
    <div className={styles.page}>
      {/* Summary row */}
      <section className={styles.statsRow}>
        <StatCard label="Active Cameras"  value="—"      color="var(--info)" />
        <StatCard label="Alerts Today"    value="—"      color="var(--danger)" />
        <StatCard label="Avg Confidence"  value="—"      color="var(--accent)" />
        <StatCard label="Crowd Density"   value="—"      color="var(--warning)" />
      </section>

      {/* Main content area */}
      <section className={styles.grid}>
        {/* Live feed placeholder */}
        <div className={`${styles.panel} ${styles.wide}`}>
          <p className={styles.panelTitle}>Live Feed</p>
          <div className={styles.placeholder}>
            <span className={styles.placeholderIcon}>📹</span>
            <p>No active camera streams</p>
          </div>
        </div>

        {/* Alert timeline placeholder */}
        <div className={styles.panel}>
          <p className={styles.panelTitle}>Recent Alerts</p>
          <div className={styles.placeholder}>
            <span className={styles.placeholderIcon}>🔔</span>
            <p>No alerts recorded</p>
          </div>
        </div>
      </section>
    </div>
  );
}
