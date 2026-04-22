import { useState, useEffect, useCallback, useMemo } from 'react';
import {
  Activity,
  Wifi,
  WifiOff,
  Eye,
  Camera,
  CircleDot,
  Clock,
  Users,
  Pin,
  Volume2,
  VolumeX,
  Radio,
} from 'lucide-react';
import './LiveMonitorPage.css';
import { getLiveStatus, listLiveEvents, startLive, stopLive } from '../services/api.js';

function AnomalyCard({ event }) {
  const sevClass =
    event.severity === 'RED' ? 'red' : event.severity === 'YELLOW' ? 'yellow' : 'green';

  return (
    <div className={`anomaly-card anomaly-card--${sevClass}`}>
      <div className={`anomaly-card__bar anomaly-card__bar--${sevClass}`} />
      <div className="anomaly-card__body">
        <div className="anomaly-card__top">
          <span className="anomaly-card__detector">{event.detector}</span>
          <span className={`badge-${sevClass}`}>{event.severity}</span>
          <span className="anomaly-card__conf">{Math.round(event.confidence * 100)}%</span>
        </div>
        <p className="anomaly-card__desc">{event.description}</p>
        <div className="anomaly-card__bottom">
          <span className="anomaly-card__time">
            <Clock size={11} /> {event.timestamp || event.time}
          </span>
          {event.personIds?.length > 0 && (
            <span className="anomaly-card__ids">
              <Users size={11} /> ID: {event.personIds.slice(0, 3).join(', ')}
              {event.personIds.length > 3 && ` +${event.personIds.length - 3}`}
            </span>
          )}
        </div>
      </div>
      <button className="anomaly-card__pin" title="Pin alert">
        <Pin size={13} />
      </button>
    </div>
  );
}

function AnomalyFeed({ events, isLive }) {
  const [sevFilter, setSevFilter] = useState('ALL');
  const [muted, setMuted] = useState(true);

  const filtered =
    sevFilter === 'ALL' ? events : events.filter((event) => event.severity === sevFilter);

  return (
    <div className="anomaly-feed card">
      <div className="anomaly-feed__header">
        <h3>
          {isLive && <span className="anomaly-feed__live-dot pulse-red" />}
          Live Anomaly Feed
        </h3>
        <span className="anomaly-feed__count badge-red">{events.length}</span>
      </div>

      <div className="anomaly-feed__filters">
        {['ALL', 'RED', 'YELLOW'].map((filterName) => (
          <button
            key={filterName}
            className={`feed-filter ${sevFilter === filterName ? 'feed-filter--active' : ''} ${
              filterName === 'RED'
                ? 'feed-filter--red'
                : filterName === 'YELLOW'
                ? 'feed-filter--yellow'
                : ''
            }`}
            onClick={() => setSevFilter(filterName)}
          >
            {filterName}
          </button>
        ))}
        <button
          className={`feed-mute ${muted ? '' : 'feed-mute--on'}`}
          onClick={() => setMuted((prev) => !prev)}
          title={muted ? 'Unmute alerts' : 'Mute alerts'}
        >
          {muted ? <VolumeX size={14} /> : <Volume2 size={14} />}
        </button>
      </div>

      <div className="anomaly-feed__list">
        {filtered.length > 0 ? (
          filtered.map((event) => <AnomalyCard key={event.id} event={event} />)
        ) : (
          <div className="anomaly-feed__empty">
            <div className="radar-sweep" />
            <span>Monitoring...</span>
          </div>
        )}
      </div>
    </div>
  );
}

function CctvViewport({ streamActive, fps, frameCount, detectionCount, cameraLabel }) {
  const [now, setNow] = useState(new Date());

  useEffect(() => {
    const interval = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(interval);
  }, []);

  const timeLabel = now.toLocaleTimeString('en-US', { hour12: false });
  const dateLabel = now.toLocaleDateString('en-US', {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
  });

  return (
    <div className="cctv-viewport">
      <div className="cctv-viewport__scanlines" />
      <div className="cctv-viewport__corner cctv-viewport__corner--tl" />
      <div className="cctv-viewport__corner cctv-viewport__corner--tr" />
      <div className="cctv-viewport__corner cctv-viewport__corner--bl" />
      <div className="cctv-viewport__corner cctv-viewport__corner--br" />

      <div className="cctv-hud cctv-hud--tl">
        <span className="cctv-rec">
          <span className="cctv-rec__dot pulse-red" />
          REC
        </span>
        <span className="cctv-ts">
          {dateLabel} {timeLabel}
        </span>
      </div>

      <div className="cctv-hud cctv-hud--tr">
        <span className="cctv-cam">{cameraLabel}</span>
        <span className="cctv-res badge-green">1080p</span>
      </div>

      {streamActive ? (
        <img className="cctv-viewport__stream" src="/api/video_feed" alt="Live MJPEG stream" />
      ) : (
        <div className="cctv-viewport__nosignal">
          <div className="static-noise" />
          <div className="nosignal-text">
            <WifiOff size={36} />
            <span>NO SIGNAL</span>
          </div>
        </div>
      )}

      <div className="cctv-hud-bar">
        <div className="cctv-hud-bar__item">
          <Activity size={12} />
          <span>{fps} FPS</span>
        </div>
        <div className="cctv-hud-bar__item">
          <Camera size={12} />
          <span>Frame {frameCount.toLocaleString()}</span>
        </div>
        <div className="cctv-hud-bar__item">
          <Eye size={12} />
          <span>{detectionCount} Detections</span>
        </div>
      </div>
    </div>
  );
}

export default function LiveMonitorPage() {
  const [events, setEvents] = useState([]);
  const [streamActive, setStreamActive] = useState(false);
  const [streamBusy, setStreamBusy] = useState(false);
  const [streamSource, setStreamSource] = useState('0');
  const [feedPaused, setFeedPaused] = useState(false);
  const [fps] = useState(25);
  const [frameCount, setFrameCount] = useState(0);
  const [streamStatusText, setStreamStatusText] = useState('');

  const cameraLabel = useMemo(() => {
    if (streamSource === '0') return 'CAM-00';
    if (streamSource === '1') return 'CAM-01';
    return 'CAM-CUSTOM';
  }, [streamSource]);

  const fetchLiveStatus = useCallback(async () => {
    try {
      const data = await getLiveStatus();
      setStreamActive(Boolean(data.running));
      setFrameCount(Number(data.frame || 0));
      if (data.source !== undefined && data.source !== null) {
        setStreamSource(String(data.source));
      }
      if (data.status) {
        setStreamStatusText(String(data.status));
      }
    } catch {
      // Keep previous UI state.
    }
  }, []);

  useEffect(() => {
    fetchLiveStatus();
  }, [fetchLiveStatus]);

  useEffect(() => {
    const interval = setInterval(fetchLiveStatus, 1500);
    return () => clearInterval(interval);
  }, [fetchLiveStatus]);

  useEffect(() => {
    if (feedPaused) {
      return undefined;
    }

    const interval = setInterval(async () => {
      try {
        const data = await listLiveEvents();
        if (Array.isArray(data) && data.length > 0) {
          setEvents(data);
        }
      } catch {
        // Keep existing events as fallback.
      }
    }, 1000);

    return () => clearInterval(interval);
  }, [feedPaused]);

  const toggleStream = useCallback(async () => {
    if (streamBusy) {
      return;
    }

    setStreamBusy(true);

    try {
      if (streamActive) {
        await stopLive();
      } else {
        await startLive(streamSource, true);
      }

      await fetchLiveStatus();
    } finally {
      setStreamBusy(false);
    }
  }, [streamActive, streamBusy, streamSource, fetchLiveStatus]);

  const toggleFeed = useCallback(() => {
    setFeedPaused((prev) => !prev);
  }, []);

  return (
    <div className="live-monitor fade-in">
      <div className="monitor-topbar">
        <div className="monitor-topbar__left">
          <span className={`topbar-status ${streamActive ? 'topbar-status--active' : ''}`}>
            <CircleDot size={12} />
            {streamActive ? 'Live stream active' : 'Stream standby'}
          </span>

          <select
            className="monitor-source-select"
            value={streamSource}
            onChange={(event) => setStreamSource(event.target.value)}
            disabled={streamActive || streamBusy}
            title="Select camera source"
          >
            <option value="0">Webcam 0</option>
            <option value="1">Webcam 1</option>
          </select>

          <button
            className="action-btn action-btn--ghost action-btn--sm"
            onClick={toggleStream}
            disabled={streamBusy}
          >
            {streamActive ? <WifiOff size={12} /> : <Wifi size={12} />}
            {streamBusy
              ? 'Applying...'
              : streamActive
              ? 'Disconnect stream'
              : 'Connect stream'}
          </button>

          <button className="action-btn action-btn--ghost action-btn--sm" onClick={toggleFeed}>
            <Radio size={12} />
            {feedPaused ? 'Resume feed' : 'Pause feed'}
          </button>
        </div>

        <div className="monitor-topbar__right">
          <span className="topbar-chip">
            <Activity size={12} /> {fps} FPS
          </span>
          <span className="topbar-chip">
            <Camera size={12} /> F:{frameCount.toLocaleString()}
          </span>
          <span className={`topbar-status ${!feedPaused ? 'topbar-status--active' : ''}`}>
            <CircleDot size={12} />
            {feedPaused ? 'Feed paused' : 'Feed live'}
          </span>
          {streamStatusText && <span className="topbar-chip">Status: {streamStatusText}</span>}
        </div>
      </div>

      <div className="monitor-body">
        <div className="monitor-main">
          <CctvViewport
            streamActive={streamActive}
            fps={fps}
            frameCount={frameCount}
            detectionCount={events.length}
            cameraLabel={cameraLabel}
          />
        </div>

        <AnomalyFeed events={events} isLive={streamActive && !feedPaused} />
      </div>
    </div>
  );
}
