import { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import {
  UploadCloud,
  FileVideo,
  Rocket,
  Square,
  CircleDot,
  SlidersHorizontal,
  CheckCircle2,
  Footprints,
  PersonStanding,
  Zap,
  Users,
  ArrowDownUp,
  DoorOpen,
  Flame,
  Package,
  Paintbrush,
  RefreshCw,
} from 'lucide-react';
import './LiveMonitorPage.css';
import './AnalyzeVideoPage.css';
import { getLatestReport, getProgress, uploadAndAnalyze } from '../services/api.js';

const ALL_DETECTORS = [
  { key: 'loitering', label: 'Loitering', icon: Footprints },
  { key: 'falling', label: 'Falling', icon: PersonStanding },
  { key: 'dispersal', label: 'Dispersal', icon: Zap },
  { key: 'density', label: 'Crowd Density', icon: Users },
  { key: 'counterflow', label: 'Counter-Flow', icon: ArrowDownUp },
  { key: 'exitblock', label: 'Exit Blocking', icon: DoorOpen },
  { key: 'fire', label: 'Fire/Smoke', icon: Flame },
  { key: 'abandoned', label: 'Abandoned Object', icon: Package },
  { key: 'vandalism', label: 'Vandalism', icon: Paintbrush },
];

const INITIAL_RUNS = [
  { id: 'run-1', fileName: 'entrance_gate_04.mp4', status: 'completed', risk: 'High', time: '09:42' },
  { id: 'run-2', fileName: 'station_hall_12.mp4', status: 'completed', risk: 'Medium', time: '08:10' },
  { id: 'run-3', fileName: 'mall_wing_c.mp4', status: 'completed', risk: 'Low', time: 'Yesterday' },
];

function UploadDropZone({ onFileDrop, onBrowse, fileInputRef }) {
  const [dragOver, setDragOver] = useState(false);

  const handleDrop = (event) => {
    event.preventDefault();
    setDragOver(false);
    const file = event.dataTransfer.files?.[0];
    if (file) {
      onFileDrop(file);
    }
  };

  return (
    <div
      className={`drop-zone ${dragOver ? 'drop-zone--active' : ''}`}
      onDragOver={(event) => {
        event.preventDefault();
        setDragOver(true);
      }}
      onDragLeave={() => setDragOver(false)}
      onDrop={handleDrop}
    >
      <UploadCloud size={64} className="drop-zone__icon" />
      <h2>Drop video file here</h2>
      <p className="text-secondary">Supports MP4, AVI, MOV up to 2GB</p>
      <div className="drop-zone__divider">
        <span>OR</span>
      </div>
      <button className="drop-zone__browse" onClick={onBrowse}>
        Browse Files
      </button>
      <div className="drop-zone__formats">
        {['MP4', 'AVI', 'MOV'].map((format) => (
          <span key={format} className="drop-zone__fmt">
            <FileVideo size={13} /> .{format.toLowerCase()}
          </span>
        ))}
      </div>
      <input
        ref={fileInputRef}
        type="file"
        accept=".mp4,.avi,.mov"
        style={{ display: 'none' }}
        onChange={(event) => {
          if (event.target.files?.[0]) {
            onFileDrop(event.target.files[0]);
          }
        }}
      />
    </div>
  );
}

function FilePreview({ file, videoUrl, onStart, onClear }) {
  const videoRef = useRef(null);
  const [meta, setMeta] = useState({ duration: 0, width: 0, height: 0 });

  useEffect(() => {
    const video = videoRef.current;
    if (!video) {
      return undefined;
    }

    const handleLoadedMetadata = () => {
      setMeta({
        duration: video.duration || 0,
        width: video.videoWidth || 0,
        height: video.videoHeight || 0,
      });
    };

    video.addEventListener('loadedmetadata', handleLoadedMetadata);
    return () => video.removeEventListener('loadedmetadata', handleLoadedMetadata);
  }, [videoUrl]);

  const formatSize = (bytes) => {
    if (bytes < 1024 * 1024) {
      return `${(bytes / 1024).toFixed(1)} KB`;
    }
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  const formatDuration = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <div className="file-preview">
      <div className="file-preview__thumb">
        <video ref={videoRef} src={videoUrl} muted controls preload="metadata" />
      </div>

      <div className="file-preview__meta card">
        <h3 className="file-preview__name">{file.name}</h3>
        <div className="file-preview__stats">
          <span>{formatSize(file.size)}</span>
          {meta.duration > 0 && <span>{formatDuration(meta.duration)}</span>}
          {meta.width > 0 && <span>{meta.width} x {meta.height}</span>}
        </div>
      </div>

      <div className="file-preview__actions">
        <button className="action-btn action-btn--primary action-btn--lg" onClick={onStart}>
          <Rocket size={16} /> Start Analysis
        </button>
        <button className="action-btn action-btn--ghost" onClick={onClear}>
          Change File
        </button>
      </div>
    </div>
  );
}

function AnalyzingView({ progress, videoUrl, onStop, onReset }) {
  const percent = Math.min(100, Math.round(progress.percent || 0));
  const isDone = progress.status === 'completed' || percent >= 100;

  return (
    <div className="analyzing-view">
      <div className="analyze-video-preview card" style={{ backgroundColor: '#000' }}>
        {!isDone ? (
          <img 
            src="/api/video_feed" 
            alt="Live AI Analysis Stream" 
            style={{ width: '100%', height: '100%', objectFit: 'contain', borderRadius: 'inherit' }} 
          />
        ) : videoUrl ? (
          <video src={videoUrl} controls muted />
        ) : (
          <div className="analyze-video-preview__empty" />
        )}
      </div>

      <div className="analyze-progress">
        <div className="analyze-progress__bar">
          <div className="analyze-progress__fill" style={{ width: `${percent}%` }} />
        </div>

        <div className="analyze-progress__info">
          <span>
            Frame {(progress.frame || 0).toLocaleString()} of {(progress.total || 0).toLocaleString()} -
            <strong> {percent}%</strong>
          </span>

          {isDone ? (
            <button className="action-btn action-btn--primary action-btn--sm" onClick={onReset}>
              <RefreshCw size={12} /> Analyze Another
            </button>
          ) : (
            <button className="action-btn action-btn--danger action-btn--sm" onClick={onStop}>
              <Square size={12} /> Stop Analysis
            </button>
          )}
        </div>
      </div>

      {isDone && (
        <div className="analyze-result card">
          <CheckCircle2 size={16} />
          <div>
            <strong>Analysis completed.</strong>
            <p>Review generated events in Alerts Table and Reports.</p>
          </div>
        </div>
      )}
    </div>
  );
}

export default function AnalyzeVideoPage() {
  const fileInputRef = useRef(null);
  const [file, setFile] = useState(null);
  const [videoUrl, setVideoUrl] = useState(null);
  const [uploadState, setUploadState] = useState('idle');
  const [progress, setProgress] = useState({ frame: 0, total: 0, percent: 0, status: 'idle' });
  const [recentRuns, setRecentRuns] = useState(INITIAL_RUNS);
  const [settings, setSettings] = useState({
    confidence: 70,
    stride: '1',
    sensitivity: 'balanced',
    snapshots: true,
    archiveResult: true,
  });
  const [detectorToggles, setDetectorToggles] = useState(
    Object.fromEntries(ALL_DETECTORS.map((detector) => [detector.key, true]))
  );

  const enabledDetectorCount = useMemo(
    () => Object.values(detectorToggles).filter(Boolean).length,
    [detectorToggles]
  );

  const handleFileSelect = useCallback((nextFile) => {
    if (videoUrl) {
      URL.revokeObjectURL(videoUrl);
    }
    setFile(nextFile);
    setVideoUrl(URL.createObjectURL(nextFile));
    setUploadState('selected');
    setProgress({ frame: 0, total: 0, percent: 0, status: 'idle' });
  }, [videoUrl]);

  const handleClear = useCallback(() => {
    if (videoUrl) {
      URL.revokeObjectURL(videoUrl);
    }
    setFile(null);
    setVideoUrl(null);
    setUploadState('idle');
    setProgress({ frame: 0, total: 0, percent: 0, status: 'idle' });
  }, [videoUrl]);

  useEffect(() => {
    return () => {
      if (videoUrl) {
        URL.revokeObjectURL(videoUrl);
      }
    };
  }, [videoUrl]);

  const handleToggleDetector = useCallback((key) => {
    setDetectorToggles((prev) => ({ ...prev, [key]: !prev[key] }));
  }, []);

  const handleStartAnalysis = useCallback(async () => {
    if (!file) {
      return;
    }

    setUploadState('analyzing');
    setProgress({ frame: 0, total: 1000, percent: 0, status: 'running' });

    try {
      const selectedDetectors = Object.entries(detectorToggles)
        .filter(([, enabled]) => enabled)
        .map(([key]) => key);
      await uploadAndAnalyze(file, selectedDetectors, settings);
    } catch {
      // Progress polling handles retries and fallback progression.
    }
  }, [file, detectorToggles, settings]);

  useEffect(() => {
    if (uploadState !== 'analyzing') {
      return undefined;
    }

    const interval = setInterval(async () => {
      try {
        const data = await getProgress();

        setProgress((prev) => ({
          frame: data.frame ?? prev.frame,
          total: data.total ?? prev.total,
          percent: data.percent ?? prev.percent,
          status: data.status ?? prev.status,
        }));

        if (data.status === 'completed') {
          setUploadState('completed');
          const latestReport = await getLatestReport().catch(() => null);
          const severity = String(latestReport?.highest_severity || '').toUpperCase();
          const risk = severity === 'RED' ? 'High' : severity === 'YELLOW' ? 'Medium' : 'Low';
          setRecentRuns((prev) => [
            {
              id: `run-${Date.now()}`,
              fileName: file?.name || 'uploaded_video.mp4',
              status: 'completed',
              risk,
              time: 'Just now',
            },
            ...prev,
          ].slice(0, 6));
          clearInterval(interval);
        }
      } catch {
        setProgress((prev) => {
          const nextPercent = Math.min(prev.percent + 6, 97);
          const nextFrame = Math.min((prev.frame || 0) + 45, prev.total || 1000);
          return { ...prev, frame: nextFrame, percent: nextPercent, status: 'running' };
        });
      }
    }, 1000);

    return () => clearInterval(interval);
  }, [uploadState, file]);

  const handleStop = useCallback(() => {
    setUploadState('selected');
    setProgress({ frame: 0, total: 0, percent: 0, status: 'idle' });
  }, []);

  const handleResetAfterComplete = useCallback(() => {
    setUploadState('selected');
    setProgress({ frame: 0, total: 0, percent: 0, status: 'idle' });
  }, []);

  return (
    <div className="analyze-video fade-in">
      <div className="analyze-video__header">
        <div>
          <h2>Video Analysis Workspace</h2>
          <p className="text-secondary">Upload a recording, choose detectors, and run offline threat analysis.</p>
        </div>
        <div className="analyze-video__chips">
          <span className="topbar-chip">
            <SlidersHorizontal size={12} /> {enabledDetectorCount} detectors active
          </span>
          <span className={`topbar-status ${uploadState === 'analyzing' ? 'topbar-status--active' : ''}`}>
            <CircleDot size={12} />
            {uploadState === 'analyzing' ? 'Analyzing' : uploadState === 'completed' ? 'Completed' : 'Ready'}
          </span>
        </div>
      </div>

      <div className="analyze-video__body">
        <section className="analyze-video__workspace card">
          {uploadState === 'idle' && (
            <UploadDropZone
              onFileDrop={handleFileSelect}
              onBrowse={() => fileInputRef.current?.click()}
              fileInputRef={fileInputRef}
            />
          )}

          {uploadState === 'selected' && file && (
            <FilePreview file={file} videoUrl={videoUrl} onStart={handleStartAnalysis} onClear={handleClear} />
          )}

          {(uploadState === 'analyzing' || uploadState === 'completed') && (
            <AnalyzingView
              progress={progress}
              videoUrl={videoUrl}
              onStop={handleStop}
              onReset={handleResetAfterComplete}
            />
          )}
        </section>

        <aside className="analyze-video__sidebar">
          <div className="analyze-side-card card">
            <h3>Detectors</h3>
            <p className="text-secondary">Enable only the behaviors you need for this video run.</p>
            <div className="detector-toggles__grid">
              {ALL_DETECTORS.map((detector) => {
                const Icon = detector.icon;
                const enabled = detectorToggles[detector.key];

                return (
                  <button
                    key={detector.key}
                    className={`detector-pill ${enabled ? 'detector-pill--on' : ''}`}
                    onClick={() => handleToggleDetector(detector.key)}
                  >
                    <Icon size={13} /> {detector.label}
                  </button>
                );
              })}
            </div>
          </div>

          <div className="analyze-side-card card">
            <h3>Run Options</h3>
            <div className="analyze-option-row">
              <label>Confidence Threshold</label>
              <strong>{settings.confidence}%</strong>
            </div>
            <input
              type="range"
              min="40"
              max="95"
              value={settings.confidence}
              onChange={(event) =>
                setSettings((prev) => ({ ...prev, confidence: Number(event.target.value) }))
              }
            />

            <div className="analyze-option-row">
              <label>Frame Sampling</label>
              <select
                value={settings.stride}
                onChange={(event) => setSettings((prev) => ({ ...prev, stride: event.target.value }))}
              >
                <option value="1">Every frame</option>
                <option value="2">Every 2nd frame</option>
                <option value="4">Every 4th frame</option>
              </select>
            </div>

            <div className="analyze-option-row">
              <label>Alert Sensitivity</label>
              <select
                value={settings.sensitivity}
                onChange={(event) =>
                  setSettings((prev) => ({ ...prev, sensitivity: event.target.value }))
                }
              >
                <option value="conservative">Conservative</option>
                <option value="balanced">Balanced</option>
                <option value="aggressive">Aggressive</option>
              </select>
            </div>

            <label className="analyze-switch-row">
              <input
                type="checkbox"
                checked={settings.snapshots}
                onChange={(event) =>
                  setSettings((prev) => ({ ...prev, snapshots: event.target.checked }))
                }
              />
              Store event snapshots
            </label>

            <label className="analyze-switch-row">
              <input
                type="checkbox"
                checked={settings.archiveResult}
                onChange={(event) =>
                  setSettings((prev) => ({ ...prev, archiveResult: event.target.checked }))
                }
              />
              Archive run in reports
            </label>
          </div>

          <div className="analyze-side-card card">
            <h3>Recent Runs</h3>
            <div className="analyze-runs-list">
              {recentRuns.map((run) => (
                <div key={run.id} className="analyze-run-item">
                  <div>
                    <strong>{run.fileName}</strong>
                    <span>{run.time}</span>
                  </div>
                  <span
                    className={`analyze-run-risk ${
                      run.risk.toLowerCase() === 'high'
                        ? 'analyze-run-risk--high'
                        : run.risk.toLowerCase() === 'low'
                        ? 'analyze-run-risk--low'
                        : 'analyze-run-risk--medium'
                    }`}
                  >
                    {run.risk}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </aside>
      </div>
    </div>
  );
}
