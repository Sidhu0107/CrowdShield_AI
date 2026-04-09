import { useMemo } from 'react';
import { useAlertsData } from '../services/alerts-context.jsx';
import styles from './LiveMonitor.module.css';

function StatCard({ label, value, tone }) {
  return (
    <div className={styles.statCard}>
      <span className={styles.statLabel}>{label}</span>
      <span className={`${styles.statValue} ${styles[tone]}`}>{value}</span>
    </div>
  );
}

export default function LiveMonitor() {
  const { alerts, services, connected } = useAlertsData();

  const latest = alerts[0] ?? null;
  const highSeverity = useMemo(
    () => alerts.filter((a) => a.severity === 'high').length,
    [alerts]
  );

  return (
    <div className={styles.page}>
      <section className={styles.statsRow}>
        <StatCard label="WebSocket" value={connected ? 'Connected' : 'Disconnected'} tone={connected ? 'ok' : 'danger'} />
        <StatCard label="Total Alerts" value={alerts.length} tone="info" />
        <StatCard label="High Severity" value={highSeverity} tone="danger" />
        <StatCard label="Gateway Status" value={services['api-gateway'] ?? 'unknown'} tone="ok" />
      </section>

      <section className={styles.grid}>
        <article className={`${styles.panel} ${styles.wide}`}>
          <h2 className={styles.panelTitle}>Live Monitor</h2>
          <div className={styles.feedPlaceholder}>
            <p className={styles.feedTitle}>Live stream placeholder</p>
            <p className={styles.feedSub}>Connect camera render source from API gateway stream.</p>
          </div>
        </article>

        <article className={styles.panel}>
          <h2 className={styles.panelTitle}>Latest Alert</h2>
          {latest ? (
            <div className={styles.latestAlert}>
              <p><strong>Type:</strong> {latest.type}</p>
              <p><strong>Severity:</strong> {latest.severity}</p>
              <p><strong>Confidence:</strong> {(latest.confidence * 100).toFixed(1)}%</p>
              <p><strong>Camera:</strong> {latest.camera_id}</p>
              <p><strong>Time:</strong> {new Date(latest.timestamp).toLocaleString()}</p>
            </div>
          ) : (
            <div className={styles.empty}>No live alerts yet.</div>
          )}
        </article>
      </section>
    </div>
  );
}
