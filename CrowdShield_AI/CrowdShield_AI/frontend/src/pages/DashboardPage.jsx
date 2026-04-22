import { useEffect, useMemo, useState } from 'react';
import {
  Video,
  AlertTriangle,
  Zap,
  Cpu,
  TrendingUp,
  ArrowUpRight,
  Play,
  Download,
  Trash2,
  Radio,
  Eye,
  PersonStanding,
  Brain,
  Filter,
  ChevronRight,
  Clock,
  Crosshair,
} from 'lucide-react';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  LineChart,
  Line,
} from 'recharts';
import { Link } from 'react-router-dom';
import './DashboardPage.css';
import { getLiveStatus, getProgress, getLatestReport, listLiveEvents } from '../services/api.js';

const DETECTOR_TOTAL = 11;

function severityRank(severity) {
  if (severity === 'RED') return 3;
  if (severity === 'YELLOW') return 2;
  return 1;
}

function KpiCard({ icon: Icon, label, value, sub, color, glow }) {
  return (
    <div className={`kpi-card card ${glow ? 'glow-red' : ''}`}>
      <div className="kpi-card__header">
        <div className={`kpi-card__icon kpi-card__icon--${color}`}>
          <Icon size={16} />
        </div>
        <ArrowUpRight size={14} className="kpi-card__trend" />
      </div>
      <div className={`kpi-card__value kpi-card__value--${color}`}>{value}</div>
      <div className="kpi-card__label">{label}</div>
      {sub && <div className="kpi-card__sub">{sub}</div>}
    </div>
  );
}

function ChartTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="chart-tooltip card">
      <div className="chart-tooltip__time">{label}</div>
      {payload.map((point) => (
        <div key={point.dataKey} className="chart-tooltip__row">
          <span className="chart-tooltip__dot" style={{ background: point.stroke || point.fill }} />
          <span className="chart-tooltip__label">
            {point.dataKey === 'total' ? 'Total' : point.dataKey === 'red' ? 'Critical' : 'Warning'}
          </span>
          <span className="chart-tooltip__val">{point.value}</span>
        </div>
      ))}
    </div>
  );
}

function DetectorRow({ detector, maxHits }) {
  const barWidth = maxHits > 0 ? (detector.hits / maxHits) * 100 : 0;
  const sevClass =
    detector.severity === 'RED'
      ? 'badge-red'
      : detector.severity === 'YELLOW'
      ? 'badge-yellow'
      : 'badge-green';
  const barColor =
    detector.severity === 'RED'
      ? 'var(--color-red)'
      : detector.severity === 'YELLOW'
      ? 'var(--color-yellow)'
      : 'var(--color-green)';
  const sparkData = detector.spark.map((value, i) => ({ i, v: value }));

  return (
    <div className="detector-row">
      <div className="detector-row__info">
        <span className="detector-row__name">{detector.name}</span>
        <span className={sevClass}>{detector.severity}</span>
      </div>
      <div className="detector-row__bar-track">
        <div className="detector-row__bar-fill" style={{ width: `${barWidth}%`, background: barColor }} />
        <span className="detector-row__hits">{detector.hits}</span>
      </div>
      <div className="detector-row__spark">
        <ResponsiveContainer width={60} height={24}>
          <LineChart data={sparkData}>
            <Line type="monotone" dataKey="v" stroke={barColor} strokeWidth={1.5} dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function CriticalEventCard({ event, index }) {
  return (
    <div className="critical-card card" style={{ animationDelay: `${index * 80}ms` }}>
      <div className="critical-card__left" />
      <div className="critical-card__body">
        <div className="critical-card__header">
          <span className="critical-card__detector">
            <Crosshair size={13} />
            {event.detector}
          </span>
          <span className="badge-red">{Math.round(event.confidence * 100)}%</span>
        </div>
        <p className="critical-card__desc">{event.description}</p>
        <div className="critical-card__footer">
          <span className="critical-card__time">
            <Clock size={12} />
            {event.timestamp}
          </span>
          <Link to="/alerts" className="critical-card__btn">
            View Frame <ChevronRight size={13} />
          </Link>
        </div>
      </div>
    </div>
  );
}

function PipelineStage({ stage, isLast }) {
  const Icon = stage.icon;
  return (
    <div className="pipeline-stage">
      <div className="pipeline-stage__row">
        <div className={`pipeline-stage__dot ${stage.online ? 'pulse-green' : ''}`} />
        <Icon size={15} className="pipeline-stage__icon" />
        <span className="pipeline-stage__name">{stage.name}</span>
        <span className="pipeline-stage__latency">{stage.latency}</span>
      </div>
      {!isLast && <div className="pipeline-stage__connector" />}
    </div>
  );
}

export default function DashboardPage() {
  const [events, setEvents] = useState([]);
  const [report, setReport] = useState(null);
  const [liveStatus, setLiveStatus] = useState({ running: false, status: 'stopped' });
  const [progress, setProgress] = useState({ frame: 0, total: 0, status: 'idle', percent: 0 });

  useEffect(() => {
    let mounted = true;

    const load = async () => {
      try {
        const [eventRows, latestReport, status, progressData] = await Promise.all([
          listLiveEvents(),
          getLatestReport().catch(() => null),
          getLiveStatus().catch(() => ({ running: false, status: 'stopped' })),
          getProgress().catch(() => ({ frame: 0, total: 0, status: 'idle', percent: 0 })),
        ]);

        if (!mounted) return;

        setEvents(eventRows);
        if (latestReport && typeof latestReport === 'object') {
          setReport(latestReport);
        }
        setLiveStatus(status || { running: false, status: 'stopped' });
        setProgress(progressData || { frame: 0, total: 0, status: 'idle', percent: 0 });
      } catch {
        // keep last good state
      }
    };

    load();
    const interval = setInterval(load, 2500);

    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, []);

  const hourlyData = useMemo(() => {
    const buckets = Array.from({ length: 24 }, (_, i) => ({
      hour: `${String(i).padStart(2, '0')}:00`,
      total: 0,
      red: 0,
      yellow: 0,
    }));

    for (const event of events) {
      const index = Math.max(0, Math.min(23, Math.floor((event.timestampSeconds || 0) / 3600)));
      buckets[index].total += 1;
      if (event.severity === 'RED') buckets[index].red += 1;
      if (event.severity === 'YELLOW') buckets[index].yellow += 1;
    }

    return buckets;
  }, [events]);

  const detectors = useMemo(() => {
    const map = new Map();

    for (const event of events) {
      if (!map.has(event.detector)) {
        map.set(event.detector, {
          name: event.detector,
          hits: 0,
          severity: 'GREEN',
          spark: [0, 0, 0, 0, 0],
        });
      }
      const row = map.get(event.detector);
      row.hits += 1;
      if (severityRank(event.severity) > severityRank(row.severity)) {
        row.severity = event.severity;
      }

      const bucket = Math.max(0, Math.min(4, Math.floor((event.timestampSeconds || 0) / 720)));
      row.spark[bucket] += 1;
    }

    return Array.from(map.values()).sort((a, b) => b.hits - a.hits).slice(0, 9);
  }, [events]);

  const criticalEvents = useMemo(() => {
    return events
      .filter((event) => event.severity === 'RED')
      .sort((a, b) => b.confidence - a.confidence)
      .slice(0, 5);
  }, [events]);

  const maxHits = useMemo(() => Math.max(1, ...detectors.map((detector) => detector.hits)), [detectors]);

  const redCount = events.filter((event) => event.severity === 'RED').length;
  const avgConfidence =
    events.length > 0
      ? `${((events.reduce((sum, event) => sum + event.confidence, 0) / events.length) * 100).toFixed(1)}%`
      : '0.0%';

  const pipelineStages = [
    { name: 'Ingestion', icon: Radio, latency: liveStatus.running ? 'active' : 'idle', online: liveStatus.running },
    { name: 'Vision', icon: Eye, latency: report ? 'online' : 'idle', online: Boolean(report) },
    {
      name: 'Pose',
      icon: PersonStanding,
      latency: progress.status === 'processing' ? `${Math.round(progress.percent || 0)}%` : 'ready',
      online: progress.status === 'processing' || progress.status === 'completed',
    },
    {
      name: 'Behavior',
      icon: Brain,
      latency: report?.highest_severity || 'idle',
      online: Boolean(report),
    },
    {
      name: 'Triage',
      icon: Filter,
      latency: report?.alert_summary ? 'online' : 'idle',
      online: Boolean(report?.alert_summary),
    },
  ];

  return (
    <div className="dashboard fade-in">
      <section className="dashboard__kpis">
        <KpiCard icon={Video} label="Active Feeds" value={liveStatus.running ? '1' : '0'} color="cyan" />
        <KpiCard
          icon={AlertTriangle}
          label="Alerts Captured"
          value={String(events.length)}
          color="amber"
          sub={events.length > 0 ? `${events.length} buffered events` : 'Waiting for events'}
        />
        <KpiCard icon={Zap} label="Critical Events" value={String(redCount)} color="red" glow={redCount > 0} />
        <KpiCard
          icon={Cpu}
          label="Detectors Active"
          value={`${detectors.length}/${DETECTOR_TOTAL}`}
          color="green"
        />
        <KpiCard icon={TrendingUp} label="Avg Confidence" value={avgConfidence} color="cyan" />
      </section>

      <section className="dashboard__mid">
        <div className="dashboard__chart card">
          <h2 className="section-title">Threat Activity - Last 24 Hours</h2>
          <div className="dashboard__chart-area">
            <ResponsiveContainer width="100%" height={280}>
              <AreaChart data={hourlyData} margin={{ top: 8, right: 8, left: -16, bottom: 0 }}>
                <defs>
                  <linearGradient id="gradRed" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#FF3B3B" stopOpacity={0.35} />
                    <stop offset="100%" stopColor="#FF3B3B" stopOpacity={0.02} />
                  </linearGradient>
                  <linearGradient id="gradYellow" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#FFB800" stopOpacity={0.3} />
                    <stop offset="100%" stopColor="#FFB800" stopOpacity={0.02} />
                  </linearGradient>
                  <linearGradient id="gradCyan" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#00D4FF" stopOpacity={0.18} />
                    <stop offset="100%" stopColor="#00D4FF" stopOpacity={0.01} />
                  </linearGradient>
                </defs>
                <CartesianGrid stroke="rgba(255,255,255,0.04)" strokeDasharray="3 3" />
                <XAxis
                  dataKey="hour"
                  stroke="#4B5060"
                  tick={{ fill: '#6B7280', fontSize: 11 }}
                  interval={3}
                  axisLine={{ stroke: '#1E2130' }}
                  tickLine={false}
                />
                <YAxis
                  stroke="#4B5060"
                  tick={{ fill: '#6B7280', fontSize: 11 }}
                  axisLine={false}
                  tickLine={false}
                />
                <Tooltip content={<ChartTooltip />} />
                <Area type="monotone" dataKey="total" stroke="#00D4FF" fill="url(#gradCyan)" strokeWidth={2} />
                <Area type="monotone" dataKey="yellow" stroke="#FFB800" fill="url(#gradYellow)" strokeWidth={1.5} />
                <Area type="monotone" dataKey="red" stroke="#FF3B3B" fill="url(#gradRed)" strokeWidth={1.5} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="dashboard__detectors card">
          <h2 className="section-title">Detector Performance</h2>
          <div className="dashboard__detectors-list">
            {detectors.length > 0 ? (
              detectors.map((detector) => <DetectorRow key={detector.name} detector={detector} maxHits={maxHits} />)
            ) : (
              <p className="text-secondary" style={{ padding: '8px 0' }}>
                No detector activity yet. Start live monitoring or analyze a video.
              </p>
            )}
          </div>
        </div>
      </section>

      <section className="dashboard__bottom">
        <div className="dashboard__events card">
          <h2 className="section-title">
            <span className="section-title__dot pulse-red" />
            Recent Critical Events
          </h2>
          <div className="dashboard__events-list">
            {criticalEvents.length > 0 ? (
              criticalEvents.map((event, index) => <CriticalEventCard key={event.id} event={event} index={index} />)
            ) : (
              <p className="text-secondary">No critical events captured yet.</p>
            )}
          </div>
        </div>

        <div className="dashboard__system card">
          <h2 className="section-title">System Status</h2>

          <div className="dashboard__pipeline">
            {pipelineStages.map((stage, index) => (
              <PipelineStage key={stage.name} stage={stage} isLast={index === pipelineStages.length - 1} />
            ))}
          </div>

          <div className="dashboard__storage">
            <div className="dashboard__storage-row">
              <span className="text-secondary">Live Status</span>
              <span className="text-bright">{liveStatus.status || 'stopped'}</span>
            </div>
            <div className="dashboard__storage-row">
              <span className="text-secondary">Latest Frame</span>
              <span className="text-green">{Number(liveStatus.frame || 0).toLocaleString()}</span>
            </div>
          </div>

          <div className="dashboard__actions">
            <Link to="/analyze" className="action-btn action-btn--primary" style={{ textDecoration: 'none' }}>
              <Play size={14} /> Start Analysis
            </Link>
            <Link to="/reports" className="action-btn action-btn--ghost" style={{ textDecoration: 'none' }}>
              <Download size={14} /> Export Report
            </Link>
            <Link to="/alerts" className="action-btn action-btn--danger" style={{ textDecoration: 'none' }}>
              <Trash2 size={14} /> Review Alerts
            </Link>
          </div>
        </div>
      </section>
    </div>
  );
}
