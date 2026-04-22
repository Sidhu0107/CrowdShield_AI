/* ═══════════════════════════════════════════════════════════════════
   CrowdShield AI — Shared Mock Event Data
   Used by AlertsPage, ReportsPage, and future components.
   ═══════════════════════════════════════════════════════════════════ */

export const DETECTOR_NAMES = [
  'Loitering', 'Falling', 'Sudden Dispersal', 'Crowd Density',
  'Counter Flow', 'Exit Blocking', 'Fire / Smoke',
  'Abandoned Object', 'Vandalism',
];

const TEMPLATES = {
  'Loitering': [
    'Person ID-{id} stationary in Zone A for {t} seconds.',
    'Persistent presence detected near restricted area — ID-{id}.',
  ],
  'Falling': [
    'Person ID-{id} vertical velocity spike: {v}x avg. Hip-to-shoulder collapsed.',
    'Medical emergency possible — ID-{id} on ground for >5 seconds.',
  ],
  'Sudden Dispersal': [
    'Rapid outward divergence from Zone {z} — {n} individuals fleeing.',
    'Angular variance σ²={s} in cluster near Gate {z}.',
  ],
  'Crowd Density': [
    'Cell ({cx},{cy}) density at {d} p/m² — approaching critical threshold.',
    'Zone {z} density peaked at {d} p/m² for 12 consecutive frames.',
  ],
  'Counter Flow': [
    'Person ID-{id} moving at {a}° against dominant crowd flow.',
    'Counter-flow cluster of {n} individuals detected in corridor B.',
  ],
  'Exit Blocking': [
    'Stationary group near Exit {z} obstructing egress for {t}s.',
    'Person ID-{id} blocking emergency exit for {t} seconds.',
  ],
  'Fire / Smoke': [
    'HSV heuristic triggered — fire-pixel ratio {r}% in NE quadrant.',
    'Thermal anomaly detected at ({cx},{cy}). Confidence escalating.',
  ],
  'Abandoned Object': [
    'Unattended backpack at ({cx},{cy}) — owner ID-{id} moved {m}px away.',
    'Stationary suitcase detected for {t}s. No associated person ID.',
  ],
  'Vandalism': [
    'Motion anomaly + arm extension pose correlated near wall section {z}.',
    'Pixel color shift >30 HSV units detected in spray zone.',
  ],
};

function fillTemplate(detector) {
  const pool = TEMPLATES[detector] || ['Anomaly detected.'];
  const tpl = pool[Math.floor(Math.random() * pool.length)];
  return tpl
    .replace('{id}', Math.floor(Math.random() * 80 + 1))
    .replace('{t}', Math.floor(Math.random() * 120 + 15))
    .replace('{v}', (2 + Math.random() * 3).toFixed(1))
    .replace('{n}', Math.floor(Math.random() * 10 + 3))
    .replace('{z}', String.fromCharCode(65 + Math.floor(Math.random() * 6)))
    .replace('{s}', (0.7 + Math.random() * 0.25).toFixed(2))
    .replace('{cx}', Math.floor(Math.random() * 4))
    .replace('{cy}', Math.floor(Math.random() * 4))
    .replace('{d}', (3.5 + Math.random() * 4).toFixed(1))
    .replace('{a}', Math.floor(130 + Math.random() * 30))
    .replace('{r}', (0.3 + Math.random() * 1.2).toFixed(1))
    .replace('{m}', Math.floor(150 + Math.random() * 200));
}

function pickSeverity(detector) {
  if (detector === 'Fire / Smoke' || detector === 'Falling')
    return Math.random() > 0.3 ? 'RED' : 'YELLOW';
  const r = Math.random();
  if (r < 0.2) return 'RED';
  if (r < 0.75) return 'YELLOW';
  return 'GREEN';
}

export const MOCK_EVENTS = Array.from({ length: 50 }, (_, i) => {
  const detector = DETECTOR_NAMES[Math.floor(Math.random() * DETECTOR_NAMES.length)];
  const severity = pickSeverity(detector);
  const minute = Math.floor(Math.random() * 30);
  const second = Math.floor(Math.random() * 60);
  const personCount = Math.floor(Math.random() * 4);
  const personIds = Array.from({ length: personCount }, () => Math.floor(Math.random() * 80 + 1));
  return {
    id: i + 1,
    severity,
    detector,
    description: fillTemplate(detector),
    confidence: +(0.55 + Math.random() * 0.44).toFixed(2),
    personIds,
    timestamp: `17:${minute.toString().padStart(2, '0')}:${second.toString().padStart(2, '0')}`,
    frame: Math.floor(Math.random() * 15000 + 500),
    timestampSeconds: minute * 60 + second,
  };
}).sort((a, b) => a.timestampSeconds - b.timestampSeconds);

export const SESSION_META = {
  sessionId: 'CS-20260422-A7F3',
  date: '2026-04-22',
  duration: '00:30:00',
  totalFrames: 15420,
  fps: 25,
  source: 'stampede.mp4',
  detectorsActive: 9,
};
