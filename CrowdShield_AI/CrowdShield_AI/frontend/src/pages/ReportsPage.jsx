import { useMemo, useEffect, useRef, useState, useCallback } from 'react';
import {
  Shield,
  Flame,
  AlertTriangle,
  CheckCircle,
  Clock,
  Printer,
  FileJson,
  FileSpreadsheet,
  ArrowLeft,
  Camera,
} from 'lucide-react';
import { Link } from 'react-router-dom';
import './ReportsPage.css';
import { listLiveEvents, getLatestReport, getProgress } from '../services/api.js';

const DETECTOR_NAMES = [
  'Loitering',
  'Falling',
  'Sudden Dispersal',
  'Crowd Density',
  'Counter Flow',
  'Exit Blocking',
  'Fire / Smoke',
  'Abandoned Object',
  'Vandalism',
  'Weapon',
  'Suspicious Roaming',
];

const VERDICT_LABELS = {
  RED: 'CRITICAL THREATS DETECTED',
  YELLOW: 'SUSPICIOUS ACTIVITY',
  GREEN: 'ALL CLEAR - No Anomalies',
};

function EventCard({ event, accentColor }) {
  return (
    <div className="report-event-card" style={{ borderLeftColor: accentColor }}>
      <div className="report-event-card__top">
        <span className="report-event-card__detector">{event.detector}</span>
        <span className="report-event-card__time">
          <Clock size={11} /> {event.timestamp}
        </span>
      </div>
      <p className="report-event-card__desc">{event.description}</p>
      <div className="report-event-card__bottom">
        <div className="report-conf-bar">
          <div
            className="report-conf-bar__fill"
            style={{ width: `${event.confidence * 100}%`, background: accentColor }}
          />
        </div>
        <span className="report-event-card__conf">{Math.round(event.confidence * 100)}%</span>
        <div className="report-frame-thumb">
          <Camera size={10} />
          <span>F:{event.frame}</span>
        </div>
      </div>
    </div>
  );
}

function TimelineCanvas({ events, totalSeconds }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    ctx.clearRect(0, 0, width, height);

    ctx.fillStyle = '#12141A';
    ctx.fillRect(0, 20, width, 40);
    ctx.strokeStyle = '#1E2130';
    ctx.strokeRect(0, 20, width, 40);

    const tickEvery = Math.max(30, Math.floor(totalSeconds / 15));
    ctx.fillStyle = '#4B5060';
    ctx.font = '10px Inter, sans-serif';
    ctx.textAlign = 'center';

    for (let sec = 0; sec <= totalSeconds; sec += tickEvery) {
      const x = (sec / Math.max(totalSeconds, 1)) * width;
      ctx.fillStyle = '#1E2130';
      ctx.fillRect(x, 60, 1, 6);
      ctx.fillStyle = '#4B5060';
      const m = Math.floor(sec / 60);
      const s = Math.floor(sec % 60);
      ctx.fillText(`${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`, x, 76);
    }

    for (const event of events) {
      const x = (event.timestampSeconds / Math.max(totalSeconds, 1)) * width;
      const isRed = event.severity === 'RED';
      const isYellow = event.severity === 'YELLOW';
      const color = isRed ? '#FF3B3B' : isYellow ? '#FFB800' : '#00E676';
      const barHeight = isRed ? 20 : isYellow ? 14 : 8;
      const yTop = 40 - barHeight / 2;

      ctx.fillStyle = color;
      ctx.globalAlpha = 0.85;
      ctx.fillRect(x - 1, yTop, 3, barHeight);
      ctx.globalAlpha = 1;
    }

    const playheadX = width * 0.65;
    ctx.fillStyle = '#FF6B00';
    ctx.beginPath();
    ctx.moveTo(playheadX - 5, 16);
    ctx.lineTo(playheadX + 5, 16);
    ctx.lineTo(playheadX, 22);
    ctx.closePath();
    ctx.fill();
    ctx.fillRect(playheadX - 0.5, 22, 1, 38);
  }, [events, totalSeconds]);

  return (
    <div className="timeline-canvas-wrap">
      <canvas ref={canvasRef} width={900} height={80} className="timeline-canvas" />
      <div className="timeline-hover-layer">
        {events.map((event) => {
          const left = `${(event.timestampSeconds / Math.max(totalSeconds, 1)) * 100}%`;
          const sevClass =
            event.severity === 'RED' ? 'red' : event.severity === 'YELLOW' ? 'yellow' : 'green';

          return (
            <div
              key={event.id}
              className={`timeline-tick timeline-tick--${sevClass}`}
              style={{ left }}
              title={`${event.detector}: ${event.description}`}
            />
          );
        })}
      </div>
    </div>
  );
}

export default function ReportsPage() {
  const [events, setEvents] = useState([]);
  const [report, setReport] = useState(null);
  const [progress, setProgress] = useState({ frame: 0, total: 0, status: 'idle', percent: 0 });

  useEffect(() => {
    let mounted = true;

    const load = async () => {
      try {
        const [eventRows, latestReport, progressData] = await Promise.all([
          listLiveEvents(),
          getLatestReport().catch(() => null),
          getProgress().catch(() => ({ frame: 0, total: 0, status: 'idle', percent: 0 })),
        ]);

        if (!mounted) return;

        setEvents(eventRows);
        if (latestReport && typeof latestReport === 'object') {
          setReport(latestReport);
        }
        setProgress(progressData || { frame: 0, total: 0, status: 'idle', percent: 0 });
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

  const redEvents = useMemo(() => events.filter((event) => event.severity === 'RED'), [events]);
  const yellowEvents = useMemo(() => events.filter((event) => event.severity === 'YELLOW'), [events]);

  const detectorStats = useMemo(() => {
    const detectorUniverse = new Set(DETECTOR_NAMES);
    for (const event of events) detectorUniverse.add(event.detector);

    return Array.from(detectorUniverse)
      .map((name) => {
        const rows = events.filter((event) => event.detector === name);
        const red = rows.filter((event) => event.severity === 'RED').length;
        const yellow = rows.filter((event) => event.severity === 'YELLOW').length;
        const avg =
          rows.length > 0
            ? (rows.reduce((sum, event) => sum + event.confidence, 0) / rows.length).toFixed(2)
            : '-';

        return { name, total: rows.length, red, yellow, avgConf: avg };
      })
      .sort((a, b) => b.total - a.total);
  }, [events]);

  const overallVerdict = useMemo(() => {
    if (redEvents.length > 0) return 'RED';
    if (yellowEvents.length > 0) return 'YELLOW';
    return 'GREEN';
  }, [redEvents.length, yellowEvents.length]);

  const sessionMeta = useMemo(() => {
    const now = new Date();
    const sessionId = `CS-${now.toISOString().slice(0, 10).replace(/-/g, '')}-${String(events.length).padStart(4, '0')}`;

    const durationSec = report?.video_duration_s
      ? Number(report.video_duration_s)
      : events.length > 0
      ? Math.max(...events.map((event) => event.timestampSeconds))
      : 0;

    const duration = `${String(Math.floor(durationSec / 3600)).padStart(2, '0')}:${String(
      Math.floor((durationSec % 3600) / 60)
    ).padStart(2, '0')}:${String(Math.floor(durationSec % 60)).padStart(2, '0')}`;

    return {
      sessionId,
      date: now.toISOString().slice(0, 10),
      duration,
      totalFrames: Number(progress.total || progress.frame || 0),
      fps: Number(report?.fps || 25),
      source: progress.status === 'processing' || progress.status === 'completed' ? 'uploaded video' : 'live monitor',
      detectorsActive: detectorStats.filter((item) => item.total > 0).length,
    };
  }, [events, progress, report, detectorStats]);

  const totalSeconds = useMemo(() => {
    const fromReport = Number(report?.video_duration_s || 0);
    if (fromReport > 0) return fromReport;
    if (events.length > 0) return Math.max(...events.map((event) => event.timestampSeconds));
    return 30 * 60;
  }, [events, report]);

  const exportJSON = useCallback(() => {
    const data = {
      session: sessionMeta,
      report,
      events,
      generatedAt: new Date().toISOString(),
    };

    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const anchor = document.createElement('a');
    anchor.href = URL.createObjectURL(blob);
    anchor.download = `crowdshield_report_${sessionMeta.sessionId}.json`;
    anchor.click();
    URL.revokeObjectURL(anchor.href);
  }, [events, report, sessionMeta]);

  const exportCSV = useCallback(() => {
    const header = 'ID,Severity,Detector,Description,Confidence,PersonIDs,Timestamp,Frame\n';
    const rows = events
      .map(
        (event) =>
          `${event.id},${event.severity},"${event.detector}","${event.description}",${event.confidence},"${event.personIds.join(';')}",${event.timestamp},${event.frame}`
      )
      .join('\n');

    const blob = new Blob([header + rows], { type: 'text/csv;charset=utf-8;' });
    const anchor = document.createElement('a');
    anchor.href = URL.createObjectURL(blob);
    anchor.download = `crowdshield_events_${sessionMeta.sessionId}.csv`;
    anchor.click();
    URL.revokeObjectURL(anchor.href);
  }, [events, sessionMeta.sessionId]);

  return (
    <div className="reports-page fade-in">
      <style>{`
        @media print {
          .sidebar, .header, .reports-export-bar { display: none !important; }
          .app-main { margin-left: 0 !important; }
          .app-content { max-height: none; overflow: visible; padding: 0; }
          .reports-page { color: #111 !important; background: #fff !important; }
          .reports-page * { color: #111 !important; border-color: #ccc !important; }
          .card { background: #fff !important; box-shadow: none !important; border: 1px solid #ddd !important; }
          .report-header, .report-severity-cols, .report-timeline-section,
          .report-breakdown { page-break-inside: avoid; }
          .report-severity-cols { page-break-before: always; }
          .report-timeline-section { page-break-before: always; }
        }
      `}</style>

      <section className="report-header card">
        <div className="report-header__top">
          <div className="report-header__logo">
            <Shield size={28} strokeWidth={2.5} />
            <div>
              <h1>CrowdShield AI - Incident Report</h1>
              <span className="text-secondary">Automated Threat Analysis Summary</span>
            </div>
          </div>
          <div className={`report-verdict report-verdict--${overallVerdict.toLowerCase()}`}>
            {VERDICT_LABELS[overallVerdict]}
          </div>
        </div>

        <div className="report-meta">
          {[
            ['Session ID', sessionMeta.sessionId],
            ['Date', sessionMeta.date],
            ['Duration', sessionMeta.duration],
            ['Total Frames', sessionMeta.totalFrames.toLocaleString()],
            ['Source', sessionMeta.source],
            ['Detectors Active', `${sessionMeta.detectorsActive}/${DETECTOR_NAMES.length}`],
          ].map(([key, value]) => (
            <div key={key} className="report-meta__item">
              <span className="report-meta__label">{key}</span>
              <span className="report-meta__value">{value}</span>
            </div>
          ))}
        </div>
      </section>

      <section className="report-severity-cols">
        <div className="sev-col sev-col--red card">
          <div className="sev-col__header">
            <Flame size={16} />
            <h2>Critical Threats</h2>
            <span className="badge-red">{redEvents.length}</span>
          </div>
          <div className="sev-col__list">
            {redEvents.length > 0 ? (
              redEvents.map((event) => <EventCard key={event.id} event={event} accentColor="#FF3B3B" />)
            ) : (
              <p className="sev-col__empty">No critical threats detected</p>
            )}
          </div>
        </div>

        <div className="sev-col sev-col--yellow card">
          <div className="sev-col__header">
            <AlertTriangle size={16} />
            <h2>Suspicious Activity</h2>
            <span className="badge-yellow">{yellowEvents.length}</span>
          </div>
          <div className="sev-col__list">
            {yellowEvents.length > 0 ? (
              yellowEvents.slice(0, 15).map((event) => (
                <EventCard key={event.id} event={event} accentColor="#FFB800" />
              ))
            ) : (
              <p className="sev-col__empty">No suspicious activity</p>
            )}
            {yellowEvents.length > 15 && (
              <p className="sev-col__more">+ {yellowEvents.length - 15} more events</p>
            )}
          </div>
        </div>

        <div className="sev-col sev-col--green card">
          <div className="sev-col__header">
            <CheckCircle size={16} />
            <h2>Clear Detectors</h2>
          </div>
          <div className="sev-col__list">
            {detectorStats.map((detector) => {
              const isClear = detector.total === 0;
              return (
                <div key={detector.name} className={`clear-detector-row ${isClear ? 'clear-detector-row--clear' : ''}`}>
                  <span>{detector.name}</span>
                  <span className={isClear ? 'text-green' : 'text-secondary'}>
                    {detector.total} events {isClear ? 'ok' : ''}
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      </section>

      <section className="report-timeline-section card">
        <h2 className="section-title">Event Timeline</h2>
        <p className="text-secondary" style={{ fontSize: '0.72rem', marginBottom: 12 }}>
          Vertical ticks represent detected events. Height indicates severity (RED = tallest).
        </p>
        <TimelineCanvas events={events} totalSeconds={totalSeconds} />
      </section>

      <section className="report-breakdown card">
        <h2 className="section-title">Detector Breakdown</h2>
        <table className="breakdown-table">
          <thead>
            <tr>
              <th>Detector</th>
              <th>Total Events</th>
              <th>RED</th>
              <th>YELLOW</th>
              <th>Avg Confidence</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            {detectorStats.map((detector) => (
              <tr key={detector.name}>
                <td className="breakdown-name">{detector.name}</td>
                <td>{detector.total}</td>
                <td className={detector.red > 0 ? 'text-red' : ''}>{detector.red}</td>
                <td className={detector.yellow > 0 ? 'text-yellow' : ''}>{detector.yellow}</td>
                <td>
                  {detector.avgConf !== '-' ? (
                    <span className="font-mono">{detector.avgConf}</span>
                  ) : (
                    <span className="text-secondary">-</span>
                  )}
                </td>
                <td>
                  {detector.total > 0 ? (
                    <span className="status-chip status-chip--ok">Operational</span>
                  ) : (
                    <span className="status-chip status-chip--warn">No Data</span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>

      <div className="reports-export-bar">
        <button className="action-btn action-btn--primary" onClick={() => window.print()}>
          <Printer size={14} /> Print Report
        </button>
        <button className="action-btn action-btn--ghost" onClick={exportJSON}>
          <FileJson size={14} /> Export JSON
        </button>
        <button className="action-btn action-btn--ghost" onClick={exportCSV}>
          <FileSpreadsheet size={14} /> Export CSV
        </button>
        <Link to="/analyze" className="action-btn action-btn--ghost" style={{ textDecoration: 'none' }}>
          <ArrowLeft size={14} /> New Analysis
        </Link>
      </div>
    </div>
  );
}
