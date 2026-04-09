import { useMemo } from 'react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { useAlertsData } from '../services/alerts-context.jsx';
import styles from './Analytics.module.css';

const COLORS = {
  high: '#ef4444',
  medium: '#f59e0b',
  low: '#10b981',
};

export default function Analytics() {
  const { alerts } = useAlertsData();

  const severityData = useMemo(() => {
    const counts = { high: 0, medium: 0, low: 0 };
    for (const a of alerts) {
      if (counts[a.severity] !== undefined) counts[a.severity] += 1;
    }
    return Object.entries(counts).map(([name, value]) => ({ name, value }));
  }, [alerts]);

  const typeData = useMemo(() => {
    const map = new Map();
    for (const a of alerts) {
      map.set(a.type, (map.get(a.type) ?? 0) + 1);
    }
    return Array.from(map.entries()).map(([type, count]) => ({ type, count }));
  }, [alerts]);

  return (
    <div className={styles.page}>
      <section className={styles.panel}>
        <h2 className={styles.title}>Severity Distribution</h2>
        <div className={styles.chartBox}>
          <ResponsiveContainer width="100%" height={320}>
            <PieChart>
              <Pie data={severityData} dataKey="value" nameKey="name" outerRadius={110} label>
                {severityData.map((entry) => (
                  <Cell key={entry.name} fill={COLORS[entry.name] ?? '#6366f1'} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </section>

      <section className={styles.panel}>
        <h2 className={styles.title}>Alert Type Frequency</h2>
        <div className={styles.chartBox}>
          <ResponsiveContainer width="100%" height={320}>
            <BarChart data={typeData}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
              <XAxis dataKey="type" stroke="#94a3b8" />
              <YAxis stroke="#94a3b8" />
              <Tooltip />
              <Bar dataKey="count" fill="#6366f1" radius={[8, 8, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </section>
    </div>
  );
}
