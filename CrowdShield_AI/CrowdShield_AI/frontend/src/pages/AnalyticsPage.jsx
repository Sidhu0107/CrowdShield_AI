import { useState, useEffect, useMemo, useRef } from 'react';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RTooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  Legend,
  BarChart,
  Bar,
  LabelList,
} from 'recharts';
import { Lightbulb, TrendingUp, Calendar } from 'lucide-react';
import './AnalyticsPage.css';
import { listLiveEvents } from '../services/api.js';

const SEV_COLORS = { RED: '#FF3B3B', YELLOW: '#FFB800', GREEN: '#00E676' };
const CONF_COLORS = [
  '#FF3B3B', '#FF5252', '#FF6B35', '#FF8F00', '#FFB800',
  '#FFCA28', '#C6E040', '#66BB6A', '#00C853', '#00E676',
];

function TimelineTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="analytics-tooltip card">
      <div className="analytics-tooltip__title">{label}</div>
      {payload.map((point) => (
        <div key={point.dataKey} className="analytics-tooltip__row">
          <span className="analytics-tooltip__dot" style={{ background: point.stroke || point.fill }} />
          <span>{point.dataKey === 'red' ? 'Critical' : point.dataKey === 'yellow' ? 'Warning' : 'Normal'}</span>
          <strong>{point.value}</strong>
        </div>
      ))}
    </div>
  );
}

function renderPieLabel({ name, percent }) {
  return `${name} ${(percent * 100).toFixed(0)}%`;
}

function HeatmapCanvas({ values }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    const cols = 4;
    const rows = 4;
    const cellW = width / cols;
    const cellH = height / rows;
    const max = Math.max(1, ...values.flat());
    const rowLabels = ['A', 'B', 'C', 'D'];

    ctx.clearRect(0, 0, width, height);

    for (let r = 0; r < rows; r += 1) {
      for (let c = 0; c < cols; c += 1) {
        const value = values[r][c];
        const t = Math.min(1, value / max);
        const color = t < 0.5
          ? `rgb(${Math.round(255 * (t * 2))}, ${Math.round(200 + 30 * (1 - t * 2))}, 50)`
          : `rgb(255, ${Math.round(200 * (1 - (t - 0.5) * 2))}, 50)`;

        const x = c * cellW;
        const y = r * cellH;

        ctx.fillStyle = color;
        ctx.fillRect(x + 1, y + 1, cellW - 2, cellH - 2);

        ctx.strokeStyle = 'rgba(30, 33, 48, 0.8)';
        ctx.lineWidth = 1;
        ctx.strokeRect(x + 1, y + 1, cellW - 2, cellH - 2);

        ctx.fillStyle = t > 0.5 ? '#0A0B0F' : '#C8CDD8';
        ctx.font = 'bold 11px Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'top';
        ctx.fillText(`${rowLabels[r]}${c + 1}`, x + cellW / 2, y + 6);

        ctx.font = '600 14px Inter, sans-serif';
        ctx.textBaseline = 'middle';
        ctx.fillText(`${value.toFixed(1)}`, x + cellW / 2, y + cellH / 2 + 4);
      }
    }
  }, [values]);

  return <canvas ref={canvasRef} width={280} height={280} className="heatmap-canvas" />;
}

export default function AnalyticsPage() {
  const [timeRange, setTimeRange] = useState('24h');
  const [events, setEvents] = useState([]);

  useEffect(() => {
    let mounted = true;

    const load = async () => {
      try {
        const rows = await listLiveEvents();
        if (mounted) {
          setEvents(rows);
        }
      } catch {
        // keep last snapshot
      }
    };

    load();
    const interval = setInterval(load, 3000);

    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, []);

  const timelineData = useMemo(() => {
    const buckets = Array.from({ length: 24 }, (_, hour) => ({
      hour: `${String(hour).padStart(2, '0')}:00`,
      red: 0,
      yellow: 0,
      green: 0,
      total: 0,
    }));

    for (const event of events) {
      const hour = Math.max(0, Math.min(23, Math.floor((event.timestampSeconds || 0) / 3600)));
      buckets[hour].total += 1;
      if (event.severity === 'RED') buckets[hour].red += 1;
      else if (event.severity === 'YELLOW') buckets[hour].yellow += 1;
      else buckets[hour].green += 1;
    }

    return buckets;
  }, [events]);

  const severityPie = useMemo(() => {
    const red = events.filter((event) => event.severity === 'RED').length;
    const yellow = events.filter((event) => event.severity === 'YELLOW').length;
    const green = Math.max(0, events.length - red - yellow);

    return [
      { name: 'RED', value: red, color: '#FF3B3B' },
      { name: 'YELLOW', value: yellow, color: '#FFB800' },
      { name: 'GREEN', value: green, color: '#00E676' },
    ].filter((item) => item.value > 0);
  }, [events]);

  const totalAlerts = useMemo(() => severityPie.reduce((sum, item) => sum + item.value, 0), [severityPie]);

  const detectorHits = useMemo(() => {
    const map = new Map();

    for (const event of events) {
      if (!map.has(event.detector)) {
        map.set(event.detector, { name: event.detector, hits: 0, severity: 'GREEN' });
      }

      const row = map.get(event.detector);
      row.hits += 1;
      if (event.severity === 'RED') row.severity = 'RED';
      else if (event.severity === 'YELLOW' && row.severity !== 'RED') row.severity = 'YELLOW';
    }

    return Array.from(map.values()).sort((a, b) => b.hits - a.hits).slice(0, 9);
  }, [events]);

  const confidenceBuckets = useMemo(() => {
    const buckets = Array.from({ length: 10 }, (_, i) => ({ range: `${i * 10}-${(i + 1) * 10}`, count: 0 }));

    for (const event of events) {
      const percent = Math.round(event.confidence * 100);
      const index = Math.max(0, Math.min(9, Math.floor(percent / 10)));
      buckets[index].count += 1;
    }

    return buckets;
  }, [events]);

  const insights = useMemo(() => {
    const topDetector = detectorHits[0];
    const redCount = events.filter((event) => event.severity === 'RED').length;
    const avgConfidence =
      events.length > 0
        ? ((events.reduce((sum, event) => sum + event.confidence, 0) / events.length) * 100).toFixed(1)
        : '0.0';

    const peakHour = timelineData.reduce(
      (best, item) => (item.total > best.total ? item : best),
      { hour: '00:00', total: 0 }
    );

    return [
      {
        title: 'Peak Activity Window',
        text: `${peakHour.hour} shows the highest activity window with ${peakHour.total} alert(s).`,
      },
      {
        title: 'Dominant Detector',
        text: topDetector
          ? `${topDetector.name} produced the highest volume with ${topDetector.hits} alert(s).`
          : 'No detector activity captured yet.',
      },
      {
        title: 'Critical Alert Pressure',
        text: `${redCount} critical event(s) currently in the active dataset.`,
      },
      {
        title: 'Model Confidence Snapshot',
        text: `Average model confidence across active events is ${avgConfidence}%.`,
      },
    ];
  }, [events, detectorHits, timelineData]);

  const heatmapData = useMemo(() => {
    const grid = Array.from({ length: 4 }, () => Array.from({ length: 4 }, () => 0));

    for (const event of events) {
      const index = Math.abs((event.frame || 0) + String(event.detector).length) % 16;
      const row = Math.floor(index / 4);
      const col = index % 4;
      grid[row][col] += 1 + (event.severity === 'RED' ? 0.6 : 0.3);
    }

    return grid;
  }, [events]);

  const heatmapMax = useMemo(() => Math.max(1, ...heatmapData.flat()), [heatmapData]);

  return (
    <div className="analytics-page fade-in">
      <div className="analytics-header">
        <div className="analytics-header__left">
          <TrendingUp size={20} className="text-cyan" />
          <h1>Analytics & Insights</h1>
        </div>
        <div className="time-range-pills">
          {[
            { key: '1h', label: 'Last Hour' },
            { key: '24h', label: 'Last 24h' },
            { key: '7d', label: 'Last 7d' },
          ].map((item) => (
            <button
              key={item.key}
              className={`time-pill ${timeRange === item.key ? 'time-pill--active' : ''}`}
              onClick={() => setTimeRange(item.key)}
            >
              <Calendar size={12} />
              {item.label}
            </button>
          ))}
        </div>
      </div>

      <section className="analytics-row-full">
        <div className="chart-card card">
          <h2 className="chart-card__title">Alert Timeline</h2>
          <ResponsiveContainer width="100%" height={260}>
            <AreaChart data={timelineData} margin={{ top: 8, right: 12, left: -12, bottom: 0 }}>
              <defs>
                <linearGradient id="aGradRed" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#FF3B3B" stopOpacity={0.35} />
                  <stop offset="100%" stopColor="#FF3B3B" stopOpacity={0.02} />
                </linearGradient>
                <linearGradient id="aGradYellow" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#FFB800" stopOpacity={0.3} />
                  <stop offset="100%" stopColor="#FFB800" stopOpacity={0.02} />
                </linearGradient>
                <linearGradient id="aGradGreen" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#00E676" stopOpacity={0.2} />
                  <stop offset="100%" stopColor="#00E676" stopOpacity={0.01} />
                </linearGradient>
              </defs>
              <CartesianGrid stroke="rgba(255,255,255,0.04)" strokeDasharray="3 3" />
              <XAxis
                dataKey="hour"
                stroke="#4B5060"
                tick={{ fill: '#6B7280', fontSize: 11 }}
                interval={3}
                tickLine={false}
                axisLine={{ stroke: '#1E2130' }}
              />
              <YAxis
                stroke="#4B5060"
                tick={{ fill: '#6B7280', fontSize: 11 }}
                axisLine={false}
                tickLine={false}
              />
              <RTooltip content={<TimelineTooltip />} />
              <Area type="monotone" dataKey="green" stroke="#00E676" fill="url(#aGradGreen)" strokeWidth={1.5} />
              <Area type="monotone" dataKey="yellow" stroke="#FFB800" fill="url(#aGradYellow)" strokeWidth={1.5} />
              <Area type="monotone" dataKey="red" stroke="#FF3B3B" fill="url(#aGradRed)" strokeWidth={2} />
              <Legend
                verticalAlign="top"
                align="right"
                wrapperStyle={{ fontSize: '0.72rem', color: '#6B7280' }}
                formatter={(value) => (value === 'red' ? 'Critical' : value === 'yellow' ? 'Warning' : 'Normal')}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </section>

      <section className="analytics-row-3">
        <div className="chart-card card">
          <h2 className="chart-card__title">Severity Distribution</h2>
          <div className="donut-wrap">
            <ResponsiveContainer width="100%" height={240}>
              <PieChart>
                <Pie
                  data={severityPie}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={90}
                  paddingAngle={3}
                  dataKey="value"
                  animationBegin={0}
                  animationDuration={900}
                  label={renderPieLabel}
                >
                  {severityPie.map((entry) => (
                    <Cell key={entry.name} fill={entry.color} stroke="none" />
                  ))}
                </Pie>
                <RTooltip
                  contentStyle={{
                    background: '#12141A',
                    border: '1px solid #1E2130',
                    borderRadius: 8,
                    fontSize: '0.75rem',
                    color: '#C8CDD8',
                  }}
                />
              </PieChart>
            </ResponsiveContainer>
            <div className="donut-center">
              <span className="donut-center__num">{totalAlerts}</span>
              <span className="donut-center__label">Total</span>
            </div>
          </div>
          <div className="donut-legend">
            {severityPie.map((item) => (
              <div key={item.name} className="donut-legend__item">
                <span className="donut-legend__dot" style={{ background: item.color }} />
                <span>{item.name}</span>
                <strong>{item.value}</strong>
              </div>
            ))}
          </div>
        </div>

        <div className="chart-card card">
          <h2 className="chart-card__title">Detector Hit Rate</h2>
          <ResponsiveContainer width="100%" height={320}>
            <BarChart data={detectorHits} layout="vertical" margin={{ top: 4, right: 40, left: 8, bottom: 4 }}>
              <CartesianGrid stroke="rgba(255,255,255,0.03)" horizontal={false} />
              <XAxis type="number" hide />
              <YAxis
                dataKey="name"
                type="category"
                width={140}
                tick={{ fill: '#C8CDD8', fontSize: 11 }}
                axisLine={false}
                tickLine={false}
              />
              <RTooltip
                contentStyle={{
                  background: '#12141A',
                  border: '1px solid #1E2130',
                  borderRadius: 8,
                  fontSize: '0.75rem',
                  color: '#C8CDD8',
                }}
              />
              <Bar dataKey="hits" radius={[0, 4, 4, 0]} animationDuration={800}>
                {detectorHits.map((entry) => (
                  <Cell key={entry.name} fill={SEV_COLORS[entry.severity]} fillOpacity={0.75} />
                ))}
                <LabelList
                  dataKey="hits"
                  position="right"
                  style={{ fill: '#6B7280', fontSize: 11, fontFamily: 'JetBrains Mono, monospace' }}
                />
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="chart-card card">
          <h2 className="chart-card__title">Confidence Distribution</h2>
          <ResponsiveContainer width="100%" height={320}>
            <BarChart data={confidenceBuckets} margin={{ top: 8, right: 8, left: -12, bottom: 0 }}>
              <CartesianGrid stroke="rgba(255,255,255,0.04)" strokeDasharray="3 3" />
              <XAxis
                dataKey="range"
                stroke="#4B5060"
                tick={{ fill: '#6B7280', fontSize: 10 }}
                tickLine={false}
                axisLine={{ stroke: '#1E2130' }}
              />
              <YAxis
                stroke="#4B5060"
                tick={{ fill: '#6B7280', fontSize: 11 }}
                axisLine={false}
                tickLine={false}
              />
              <RTooltip
                contentStyle={{
                  background: '#12141A',
                  border: '1px solid #1E2130',
                  borderRadius: 8,
                  fontSize: '0.75rem',
                  color: '#C8CDD8',
                }}
                labelFormatter={(value) => `${value}% confidence`}
              />
              <Bar dataKey="count" radius={[4, 4, 0, 0]} animationDuration={700}>
                {confidenceBuckets.map((_, i) => (
                  <Cell key={i} fill={CONF_COLORS[i]} fillOpacity={0.8} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </section>

      <section className="analytics-row-2">
        <div className="chart-card card insights-card">
          <h2 className="chart-card__title">
            <Lightbulb size={15} className="text-yellow" />
            Key Observations
          </h2>
          <div className="insights-list">
            {insights.map((insight, index) => (
              <div key={index} className="insight-item">
                <div className="insight-item__dot" />
                <div>
                  <strong className="insight-item__title">{insight.title}</strong>
                  <p className="insight-item__text">{insight.text}</p>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="chart-card card heatmap-card">
          <h2 className="chart-card__title">Crowd Density Heatmap</h2>
          <p className="heatmap-subtitle text-secondary">4x4 grid derived from active alert distribution.</p>
          <div className="heatmap-wrap">
            <HeatmapCanvas values={heatmapData} />
          </div>
          <div className="heatmap-legend">
            <span className="heatmap-legend__label">0</span>
            <div className="heatmap-legend__bar" />
            <span className="heatmap-legend__label">{heatmapMax.toFixed(1)}</span>
          </div>
          <div className="heatmap-thresholds">
            <span className="badge-green">Safe &lt; 40%</span>
            <span className="badge-yellow">Warning 40-70%</span>
            <span className="badge-red">Critical &gt; 70%</span>
          </div>
        </div>
      </section>
    </div>
  );
}
