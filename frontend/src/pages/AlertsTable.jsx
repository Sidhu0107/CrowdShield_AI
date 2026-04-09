import { useMemo, useState } from 'react';
import { useAlertsData } from '../services/alerts-context.jsx';
import styles from './AlertsTable.module.css';

export default function AlertsTable() {
  const { alerts } = useAlertsData();
  const [filter, setFilter] = useState('all');

  const filtered = useMemo(() => {
    if (filter === 'all') return alerts;
    return alerts.filter((a) => a.severity === filter);
  }, [alerts, filter]);

  return (
    <div className={styles.page}>
      <div className={styles.toolbar}>
        <h2 className={styles.title}>Alerts Table</h2>
        <select
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          className={styles.select}
        >
          <option value="all">All Severity</option>
          <option value="high">High</option>
          <option value="medium">Medium</option>
          <option value="low">Low</option>
        </select>
      </div>

      <div className={styles.tableWrap}>
        <table className={styles.table}>
          <thead>
            <tr>
              <th>Alert ID</th>
              <th>Type</th>
              <th>Severity</th>
              <th>Confidence</th>
              <th>Camera</th>
              <th>Timestamp</th>
            </tr>
          </thead>
          <tbody>
            {filtered.length === 0 ? (
              <tr>
                <td colSpan={6} className={styles.empty}>No alerts available</td>
              </tr>
            ) : (
              filtered.map((alert) => (
                <tr key={alert.alert_id}>
                  <td className={styles.mono}>{alert.alert_id}</td>
                  <td>{alert.type}</td>
                  <td>
                    <span className={`${styles.badge} ${styles[alert.severity]}`}>
                      {alert.severity}
                    </span>
                  </td>
                  <td>{(alert.confidence * 100).toFixed(1)}%</td>
                  <td>{alert.camera_id}</td>
                  <td>{new Date(alert.timestamp).toLocaleString()}</td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
