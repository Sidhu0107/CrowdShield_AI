import { useState, useMemo, useEffect } from 'react';
import {
  Settings2,
  Save,
  RotateCcw,
  BellRing,
  SlidersHorizontal,
  Camera,
  ShieldCheck,
  Database,
} from 'lucide-react';
import './ConfigurationPage.css';
import { getConfig, saveConfig } from '../services/api.js';

const DEFAULT_CONFIG = {
  thresholds: {
    crowdDensity: 82,
    loiteringSeconds: 90,
    counterFlow: 70,
    fireSmokeConfidence: 76,
  },
  stream: {
    source: 'camera_01',
    ingestFps: 25,
    detectionStride: '1',
    retentionDays: 14,
  },
  notifications: {
    inApp: true,
    email: true,
    sms: false,
    webhook: false,
    webhookUrl: '',
  },
  governance: {
    autoEscalateCritical: true,
    blurFacesInExports: true,
    saveSnapshots: true,
  },
};

function ToggleRow({ label, value, onChange }) {
  return (
    <label className="config-toggle-row">
      <span>{label}</span>
      <input type="checkbox" checked={value} onChange={(event) => onChange(event.target.checked)} />
    </label>
  );
}

function RangeRow({ label, value, min, max, suffix, onChange }) {
  return (
    <div className="config-range-row">
      <div className="config-range-row__head">
        <label>{label}</label>
        <strong>
          {value}
          {suffix}
        </strong>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
      />
    </div>
  );
}

export default function ConfigurationPage() {
  const [config, setConfig] = useState(DEFAULT_CONFIG);
  const [savedConfig, setSavedConfig] = useState(DEFAULT_CONFIG);
  const [status, setStatus] = useState('idle');

  useEffect(() => {
    let mounted = true;

    const loadConfig = async () => {
      try {
        const remoteConfig = await getConfig();
        if (mounted && remoteConfig && typeof remoteConfig === 'object') {
          setConfig(remoteConfig);
          setSavedConfig(remoteConfig);
        }
      } catch {
        // Keep default values if backend config is unavailable.
      }
    };

    loadConfig();

    return () => {
      mounted = false;
    };
  }, []);

  const dirty = useMemo(
    () => JSON.stringify(config) !== JSON.stringify(savedConfig),
    [config, savedConfig]
  );

  const handleSave = async () => {
    setStatus('saving');

    try {
      const result = await saveConfig(config);
      const persisted = result?.config ?? config;
      setSavedConfig(persisted);
      setConfig(persisted);
      setStatus('saved');
    } catch {
      setStatus('failed');
    }
  };

  const handleReset = () => {
    setConfig(savedConfig);
    setStatus('idle');
  };

  return (
    <div className="config-page fade-in">
      <div className="config-page__header">
        <div>
          <h2>
            <Settings2 size={18} /> System Configuration
          </h2>
          <p className="text-secondary">
            Tune detector behavior, stream processing, notifications, and governance policies.
          </p>
        </div>

        <div className="config-page__actions">
          <button className="action-btn action-btn--ghost" onClick={handleReset}>
            <RotateCcw size={14} /> Reset
          </button>
          <button
            className="action-btn action-btn--primary"
            onClick={handleSave}
            disabled={status === 'saving' || !dirty}
          >
            <Save size={14} /> {status === 'saving' ? 'Saving...' : 'Save Changes'}
          </button>
        </div>
      </div>

      <div className="config-status-row">
        <span className={`config-status-chip ${dirty ? 'config-status-chip--warning' : ''}`}>
          {dirty ? 'Unsaved changes' : 'Saved state'}
        </span>
        {status === 'saved' && <span className="config-status-chip config-status-chip--success">Saved</span>}
        {status === 'failed' && <span className="config-status-chip config-status-chip--danger">Save failed</span>}
      </div>

      <div className="config-grid">
        <section className="config-card card">
          <h3>
            <SlidersHorizontal size={15} /> Detection Thresholds
          </h3>

          <RangeRow
            label="Crowd Density Trigger"
            value={config.thresholds.crowdDensity}
            min={55}
            max={95}
            suffix="%"
            onChange={(value) =>
              setConfig((prev) => ({
                ...prev,
                thresholds: { ...prev.thresholds, crowdDensity: value },
              }))
            }
          />

          <RangeRow
            label="Loitering Duration"
            value={config.thresholds.loiteringSeconds}
            min={30}
            max={180}
            suffix="s"
            onChange={(value) =>
              setConfig((prev) => ({
                ...prev,
                thresholds: { ...prev.thresholds, loiteringSeconds: value },
              }))
            }
          />

          <RangeRow
            label="Counter-Flow Confidence"
            value={config.thresholds.counterFlow}
            min={45}
            max={95}
            suffix="%"
            onChange={(value) =>
              setConfig((prev) => ({
                ...prev,
                thresholds: { ...prev.thresholds, counterFlow: value },
              }))
            }
          />

          <RangeRow
            label="Fire / Smoke Confidence"
            value={config.thresholds.fireSmokeConfidence}
            min={40}
            max={95}
            suffix="%"
            onChange={(value) =>
              setConfig((prev) => ({
                ...prev,
                thresholds: { ...prev.thresholds, fireSmokeConfidence: value },
              }))
            }
          />
        </section>

        <section className="config-card card">
          <h3>
            <Camera size={15} /> Stream Processing
          </h3>

          <div className="config-field-row">
            <label>Primary Source</label>
            <select
              value={config.stream.source}
              onChange={(event) =>
                setConfig((prev) => ({
                  ...prev,
                  stream: { ...prev.stream, source: event.target.value },
                }))
              }
            >
              <option value="camera_01">Camera 01 - Main Gate</option>
              <option value="camera_02">Camera 02 - Concourse</option>
              <option value="camera_03">Camera 03 - Exit Lobby</option>
              <option value="rtsp_custom">Custom RTSP stream</option>
            </select>
          </div>

          <div className="config-field-row">
            <label>Ingest FPS</label>
            <select
              value={config.stream.ingestFps}
              onChange={(event) =>
                setConfig((prev) => ({
                  ...prev,
                  stream: { ...prev.stream, ingestFps: Number(event.target.value) },
                }))
              }
            >
              <option value={15}>15 FPS</option>
              <option value={20}>20 FPS</option>
              <option value={25}>25 FPS</option>
              <option value={30}>30 FPS</option>
            </select>
          </div>

          <div className="config-field-row">
            <label>Detection Stride</label>
            <select
              value={config.stream.detectionStride}
              onChange={(event) =>
                setConfig((prev) => ({
                  ...prev,
                  stream: { ...prev.stream, detectionStride: event.target.value },
                }))
              }
            >
              <option value="1">Every frame</option>
              <option value="2">Every 2nd frame</option>
              <option value="3">Every 3rd frame</option>
              <option value="4">Every 4th frame</option>
            </select>
          </div>

          <RangeRow
            label="Retention Window"
            value={config.stream.retentionDays}
            min={1}
            max={30}
            suffix="d"
            onChange={(value) =>
              setConfig((prev) => ({
                ...prev,
                stream: { ...prev.stream, retentionDays: value },
              }))
            }
          />
        </section>

        <section className="config-card card">
          <h3>
            <BellRing size={15} /> Notification Channels
          </h3>

          <ToggleRow
            label="In-App Notifications"
            value={config.notifications.inApp}
            onChange={(value) =>
              setConfig((prev) => ({
                ...prev,
                notifications: { ...prev.notifications, inApp: value },
              }))
            }
          />

          <ToggleRow
            label="Email Alerts"
            value={config.notifications.email}
            onChange={(value) =>
              setConfig((prev) => ({
                ...prev,
                notifications: { ...prev.notifications, email: value },
              }))
            }
          />

          <ToggleRow
            label="SMS Alerts"
            value={config.notifications.sms}
            onChange={(value) =>
              setConfig((prev) => ({
                ...prev,
                notifications: { ...prev.notifications, sms: value },
              }))
            }
          />

          <ToggleRow
            label="Webhook Alerts"
            value={config.notifications.webhook}
            onChange={(value) =>
              setConfig((prev) => ({
                ...prev,
                notifications: { ...prev.notifications, webhook: value },
              }))
            }
          />

          {config.notifications.webhook && (
            <div className="config-field-row">
              <label>Webhook URL</label>
              <input
                type="text"
                value={config.notifications.webhookUrl}
                placeholder="https://example.com/hook"
                onChange={(event) =>
                  setConfig((prev) => ({
                    ...prev,
                    notifications: { ...prev.notifications, webhookUrl: event.target.value },
                  }))
                }
              />
            </div>
          )}
        </section>

        <section className="config-card card">
          <h3>
            <ShieldCheck size={15} /> Governance and Safety
          </h3>

          <ToggleRow
            label="Auto-escalate critical incidents"
            value={config.governance.autoEscalateCritical}
            onChange={(value) =>
              setConfig((prev) => ({
                ...prev,
                governance: { ...prev.governance, autoEscalateCritical: value },
              }))
            }
          />

          <ToggleRow
            label="Blur faces in report exports"
            value={config.governance.blurFacesInExports}
            onChange={(value) =>
              setConfig((prev) => ({
                ...prev,
                governance: { ...prev.governance, blurFacesInExports: value },
              }))
            }
          />

          <ToggleRow
            label="Save incident snapshots"
            value={config.governance.saveSnapshots}
            onChange={(value) =>
              setConfig((prev) => ({
                ...prev,
                governance: { ...prev.governance, saveSnapshots: value },
              }))
            }
          />

          <div className="config-info-box">
            <Database size={14} />
            <p>
              Retention and snapshot policies apply to both live incidents and uploaded video runs.
            </p>
          </div>
        </section>
      </div>
    </div>
  );
}
