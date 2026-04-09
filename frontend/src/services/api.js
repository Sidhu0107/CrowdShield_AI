import axios from 'axios';

/**
 * Pre-configured Axios instance for the API Gateway.
 *
 * Base URL resolves to /api in development, which Vite proxies to
 * http://localhost:8000 (see vite.config.js). In production, set
 * VITE_API_BASE_URL to the deployed gateway URL.
 */
const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL ?? '/api',
  timeout: 10_000,
  headers: { 'Content-Type': 'application/json' },
});

/* ── Health & stats ── */

/** GET /health — aggregated service health map */
export const getHealth = () => api.get('/health');

/** GET /stats — system statistics */
export const getStats = () => api.get('/stats');

/* ── Alerts ── */

/** GET /alerts — list of recent alert events */
export const getAlerts = () => api.get('/alerts');

/* ── Analysis ── */

/**
 * POST /analyze — submit a video URL for processing.
 * @param {string} videoUrl
 */
export const analyzeVideo = (videoUrl) =>
  api.post('/analyze', { video_url: videoUrl });

export default api;
