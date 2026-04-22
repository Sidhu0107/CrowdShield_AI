import { createContext, useContext, useEffect, useMemo, useState } from 'react';
import { getAlerts, getHealth } from './api.js';
import { createWebSocketClient } from './websocket.js';

const AlertsContext = createContext(null);

function normalizeAlert(raw) {
  return {
    alert_id: raw.alert_id,
    type: raw.type,
    severity: raw.severity ?? 'low',
    confidence: Number(raw.confidence ?? 0),
    timestamp: raw.timestamp ?? new Date().toISOString(),
    camera_id: raw.camera_id ?? 'cam_1',
  };
}

export function AlertsProvider({ children }) {
  const [alerts, setAlerts] = useState([]);
  const [services, setServices] = useState({});
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    let mounted = true;

    const loadInitial = async () => {
      try {
        const [alertsRes, healthRes] = await Promise.all([getAlerts(), getHealth()]);
        if (!mounted) return;

        const initialAlerts = Array.isArray(alertsRes.data)
          ? alertsRes.data.map(normalizeAlert)
          : [];
        setAlerts(initialAlerts);

        // Keep a shallow health snapshot for UI status cards.
        setServices({
          'api-gateway': healthRes.data?.status === 'ok' ? 'up' : 'down',
          redis: healthRes.data?.redis ?? 'unknown',
          database: healthRes.data?.database ?? 'unknown',
        });
      } catch {
        // Non-fatal. Real-time ws updates can still populate state.
      }
    };

    loadInitial();

    const ws = createWebSocketClient();
    ws.onOpen = () => {
      if (mounted) setConnected(true);
    };
    ws.onClose = () => {
      if (mounted) setConnected(false);
    };
    ws.onAlert = (event) => {
      if (!mounted) return;
      const normalized = normalizeAlert(event);
      setAlerts((prev) => [normalized, ...prev].slice(0, 500));
    };
    ws.onStatus = (event) => {
      if (!mounted) return;
      if (event?.services) {
        setServices(event.services);
      }
    };

    ws.connect();

    return () => {
      mounted = false;
      ws.disconnect();
    };
  }, []);

  const value = useMemo(
    () => ({ alerts, services, connected }),
    [alerts, services, connected]
  );

  return <AlertsContext.Provider value={value}>{children}</AlertsContext.Provider>;
}

export function useAlertsData() {
  const context = useContext(AlertsContext);
  if (!context) {
    throw new Error('useAlertsData must be used within AlertsProvider');
  }
  return context;
}
