import axios from 'axios';

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL ?? '/api',
  timeout: 15_000,
});

export const normalizeSeverity = (value) => {
  const text = String(value || 'YELLOW').toUpperCase();
  if (text.includes('RED') || text.includes('CRITICAL')) return 'RED';
  if (text.includes('GREEN') || text.includes('SAFE')) return 'GREEN';
  return 'YELLOW';
};

export const normalizeConfidence = (value) => {
  const number = Number(value ?? 0);
  if (!Number.isFinite(number)) return 0;
  if (number > 1) return Math.min(1, number / 100);
  if (number < 0) return 0;
  return number;
};

export const humanizeDetectorName = (value) => {
  const base = String(value || 'Unknown Detector').replace(/Detector$/i, '');
  return base
    .replace(/([a-z])([A-Z])/g, '$1 $2')
    .replace(/\s+/g, ' ')
    .trim();
};

export const normalizeEvent = (raw, index = 0) => {
  const detector = humanizeDetectorName(raw.detector || raw.detector_name || 'Unknown Detector');
  const severity = normalizeSeverity(raw.severity);
  const confidence = normalizeConfidence(raw.confidence);
  const frame = Number(raw.frame ?? raw.frame_idx ?? index);
  const personIds = Array.isArray(raw.personIds)
    ? raw.personIds
    : Array.isArray(raw.person_ids)
    ? raw.person_ids
    : [];

  const timestamp = raw.timestamp
    || raw.time
    || (() => {
      const ts = Number(raw.timestamp_s);
      if (Number.isFinite(ts)) {
        const m = Math.floor(ts / 60);
        const s = Math.floor(ts % 60);
        return `00:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
      }
      return new Date().toLocaleTimeString('en-US', { hour12: false });
    })();

  const timestampSeconds = (() => {
    const ts = Number(raw.timestamp_s);
    if (Number.isFinite(ts)) return Math.max(0, ts);
    const parts = String(timestamp).split(':').map(Number);
    if (parts.length === 3 && parts.every((part) => Number.isFinite(part))) {
      return parts[0] * 3600 + parts[1] * 60 + parts[2];
    }
    return index;
  })();

  return {
    id: raw.id || `${frame}-${index}`,
    severity,
    detector,
    description: raw.description || 'Anomaly detected.',
    confidence,
    personIds,
    timestamp,
    timestampSeconds,
    frame: Number.isFinite(frame) ? frame : index,
  };
};

export const listLiveEvents = async () => {
  const response = await api.get('/events/live');
  const rows = Array.isArray(response.data) ? response.data : [];
  return rows.map((row, index) => normalizeEvent(row, index));
};

export const getLatestReport = async () => {
  const response = await api.get('/report/latest');
  return response.data;
};

export const getProgress = async () => {
  const response = await api.get('/progress');
  return response.data;
};

export const getLiveStatus = async () => {
  const response = await api.get('/live/status');
  return response.data;
};

export const startLive = async (source = '0', clearEvents = true) => {
  const response = await api.post('/live/start', {
    source,
    clear_events: clearEvents,
  });
  return response.data;
};

export const stopLive = async () => {
  const response = await api.post('/live/stop');
  return response.data;
};

export const uploadAndAnalyze = async (file, detectors = [], options = {}) => {
  const payload = new FormData();
  payload.append('file', file);
  payload.append('detectors', JSON.stringify(detectors));
  payload.append('options', JSON.stringify(options));

  const response = await api.post('/analyze', payload, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return response.data;
};

export const getConfig = async () => {
  const response = await api.get('/config');
  return response.data;
};

export const getHealth = async () => {
  const response = await api.get('/health');
  return response.data;
};

export const saveConfig = async (config) => {
  const response = await api.post('/config', config);
  return response.data;
};

export default api;
