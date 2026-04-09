/**
 * WebSocket client for the API Gateway real-time stream.
 *
 * Connects to WS /ws and delivers live alert and service-status events
 * to registered callback handlers.
 *
 * Usage:
 *   const client = createWebSocketClient();
 *   client.onAlert = (event) => console.log(event);
 *   client.connect();
 *   // later:
 *   client.disconnect();
 */

const WS_URL =
  import.meta.env.VITE_WS_URL ??
  `${window.location.protocol === 'https:' ? 'wss' : 'ws'}://${window.location.host}/api/ws`;

/**
 * @typedef {Object} AlertEvent
 * @property {string} alert_id
 * @property {string} type
 * @property {'low'|'medium'|'high'} severity
 * @property {number} confidence
 * @property {string} timestamp
 * @property {string} camera_id
 */

/**
 * @typedef {Object} StatusEvent
 * @property {Record<string, 'up'|'down'|'degraded'>} services
 */

export function createWebSocketClient() {
  let socket = null;
  let reconnectTimer = null;
  const RECONNECT_DELAY_MS = 3_000;

  const client = {
    /** Called with each AlertEvent message received from the server */
    onAlert: null,
    /** Called with each StatusEvent message received from the server */
    onStatus: null,
    /** Called when the connection is established */
    onOpen: null,
    /** Called when the connection is closed */
    onClose: null,

    connect() {
      if (socket) return;

      socket = new WebSocket(WS_URL);

      socket.addEventListener('open', () => {
        client.onOpen?.();
      });

      socket.addEventListener('message', (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.alert_id) {
            client.onAlert?.(data);
          } else if (data.services) {
            client.onStatus?.(data);
          }
        } catch {
          // Silently discard unparseable frames
        }
      });

      socket.addEventListener('close', () => {
        socket = null;
        client.onClose?.();
        // Auto-reconnect after delay
        reconnectTimer = setTimeout(() => client.connect(), RECONNECT_DELAY_MS);
      });

      socket.addEventListener('error', () => {
        socket?.close();
      });
    },

    disconnect() {
      clearTimeout(reconnectTimer);
      socket?.close();
      socket = null;
    },
  };

  return client;
}
