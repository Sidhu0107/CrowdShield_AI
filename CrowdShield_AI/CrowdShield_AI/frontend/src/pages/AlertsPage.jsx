import { useState, useMemo, useCallback, useEffect } from 'react';
import {
  Search,
  Download,
  RefreshCw,
  Eye,
  ChevronUp,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  SlidersHorizontal,
} from 'lucide-react';
import './AlertsPage.css';
import { listLiveEvents } from '../services/api.js';

const ROWS_PER_PAGE = 20;

export default function AlertsPage() {
  const [events, setEvents] = useState([]);
  const [loading, setLoading] = useState(true);

  const [search, setSearch] = useState('');
  const [sevFilter, setSevFilter] = useState('ALL');
  const [detectorFilter, setDetectorFilter] = useState([]);
  const [showDetectorDropdown, setShowDetectorDropdown] = useState(false);

  const [sortKey, setSortKey] = useState('timestampSeconds');
  const [sortDir, setSortDir] = useState('desc');

  const [page, setPage] = useState(1);
  const [spinning, setSpinning] = useState(false);

  const loadEvents = useCallback(async () => {
    try {
      const rows = await listLiveEvents();
      setEvents(rows);
    } catch {
      // Keep previous state on transient failures.
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadEvents();
    const interval = setInterval(loadEvents, 2000);
    return () => clearInterval(interval);
  }, [loadEvents]);

  const detectorNames = useMemo(() => {
    const set = new Set(events.map((event) => event.detector));
    return Array.from(set).sort((a, b) => a.localeCompare(b));
  }, [events]);

  const filtered = useMemo(() => {
    let data = [...events];

    if (search) {
      const query = search.toLowerCase();
      data = data.filter(
        (event) =>
          event.description.toLowerCase().includes(query) ||
          event.detector.toLowerCase().includes(query)
      );
    }

    if (sevFilter !== 'ALL') {
      data = data.filter((event) => event.severity === sevFilter);
    }

    if (detectorFilter.length > 0) {
      data = data.filter((event) => detectorFilter.includes(event.detector));
    }

    return data;
  }, [events, search, sevFilter, detectorFilter]);

  const sorted = useMemo(() => {
    const data = [...filtered];
    data.sort((a, b) => {
      const aVal = a[sortKey];
      const bVal = b[sortKey];

      if (aVal < bVal) return sortDir === 'asc' ? -1 : 1;
      if (aVal > bVal) return sortDir === 'asc' ? 1 : -1;
      return 0;
    });
    return data;
  }, [filtered, sortKey, sortDir]);

  const totalPages = Math.max(1, Math.ceil(sorted.length / ROWS_PER_PAGE));

  const pageData = useMemo(() => {
    const start = (page - 1) * ROWS_PER_PAGE;
    return sorted.slice(start, start + ROWS_PER_PAGE);
  }, [sorted, page]);

  useEffect(() => {
    if (page > totalPages) {
      setPage(totalPages);
    }
  }, [page, totalPages]);

  const redCount = filtered.filter((event) => event.severity === 'RED').length;
  const yellowCount = filtered.filter((event) => event.severity === 'YELLOW').length;

  const handleSort = useCallback((key) => {
    setSortKey((previous) => {
      if (previous === key) {
        setSortDir((direction) => (direction === 'asc' ? 'desc' : 'asc'));
        return key;
      }
      setSortDir('asc');
      return key;
    });
    setPage(1);
  }, []);

  const handleRefresh = useCallback(async () => {
    setSpinning(true);
    await loadEvents();
    setTimeout(() => setSpinning(false), 300);
  }, [loadEvents]);

  const handleExportCSV = useCallback(() => {
    const header = 'ID,Severity,Detector,Description,Confidence,PersonIDs,Timestamp,Frame\n';
    const rows = filtered
      .map(
        (event) =>
          `${event.id},${event.severity},"${event.detector}","${event.description}",${event.confidence},"${event.personIds.join(';')}",${event.timestamp},${event.frame}`
      )
      .join('\n');

    const blob = new Blob([header + rows], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = `crowdshield_alerts_${Date.now()}.csv`;
    anchor.click();
    URL.revokeObjectURL(url);
  }, [filtered]);

  const toggleDetector = useCallback((name) => {
    setDetectorFilter((prev) =>
      prev.includes(name) ? prev.filter((item) => item !== name) : [...prev, name]
    );
    setPage(1);
  }, []);

  const SortArrow = ({ col }) => {
    if (sortKey !== col) {
      return <ChevronUp size={12} className="sort-arrow sort-arrow--inactive" />;
    }
    return sortDir === 'asc' ? (
      <ChevronUp size={12} className="sort-arrow" />
    ) : (
      <ChevronDown size={12} className="sort-arrow" />
    );
  };

  return (
    <div className="alerts-page fade-in">
      <div className="alerts-filters card">
        <div className="alerts-search">
          <Search size={14} className="alerts-search__icon" />
          <input
            type="text"
            placeholder="Search description or detector..."
            value={search}
            onChange={(event) => {
              setSearch(event.target.value);
              setPage(1);
            }}
          />
        </div>

        <div className="alerts-sev-pills">
          {['ALL', 'RED', 'YELLOW', 'GREEN'].map((severity) => (
            <button
              key={severity}
              className={`sev-pill ${sevFilter === severity ? 'sev-pill--active' : ''} ${
                severity === 'RED'
                  ? 'sev-pill--red'
                  : severity === 'YELLOW'
                  ? 'sev-pill--yellow'
                  : severity === 'GREEN'
                  ? 'sev-pill--green'
                  : ''
              }`}
              onClick={() => {
                setSevFilter(severity);
                setPage(1);
              }}
            >
              {severity}
            </button>
          ))}
        </div>

        <div className="detector-dropdown">
          <button
            className="detector-dropdown__btn"
            onClick={() => setShowDetectorDropdown((prev) => !prev)}
          >
            <SlidersHorizontal size={13} />
            Detectors {detectorFilter.length > 0 && `(${detectorFilter.length})`}
          </button>
          {showDetectorDropdown && (
            <div className="detector-dropdown__menu card">
              {detectorNames.map((name) => (
                <label key={name} className="detector-dropdown__item">
                  <input
                    type="checkbox"
                    checked={detectorFilter.includes(name)}
                    onChange={() => toggleDetector(name)}
                  />
                  <span>{name}</span>
                </label>
              ))}
            </div>
          )}
        </div>

        <button className="action-btn action-btn--ghost" onClick={handleExportCSV}>
          <Download size={13} /> Export CSV
        </button>
        <button
          className={`alerts-refresh ${spinning ? 'alerts-refresh--spin' : ''}`}
          onClick={handleRefresh}
          title="Refresh data"
        >
          <RefreshCw size={15} />
        </button>
      </div>

      <div className="alerts-stats">
        <span className="alerts-stats__chip">
          Showing <strong>{filtered.length}</strong> of <strong>{events.length}</strong> events
        </span>
        <span className="alerts-stats__chip alerts-stats__chip--red">
          <span className="alerts-stats__dot alerts-stats__dot--red" />
          {redCount} RED
        </span>
        <span className="alerts-stats__chip alerts-stats__chip--yellow">
          <span className="alerts-stats__dot alerts-stats__dot--yellow" />
          {yellowCount} YELLOW
        </span>
      </div>

      <div className="alerts-table-wrap card">
        <table className="alerts-table">
          <thead>
            <tr>
              <th onClick={() => handleSort('id')}># <SortArrow col="id" /></th>
              <th onClick={() => handleSort('severity')}>Severity <SortArrow col="severity" /></th>
              <th onClick={() => handleSort('detector')}>Detector <SortArrow col="detector" /></th>
              <th>Description</th>
              <th onClick={() => handleSort('confidence')}>Confidence <SortArrow col="confidence" /></th>
              <th>Person IDs</th>
              <th onClick={() => handleSort('timestampSeconds')}>Timestamp <SortArrow col="timestampSeconds" /></th>
              <th onClick={() => handleSort('frame')}>Frame <SortArrow col="frame" /></th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {pageData.map((event) => {
              const sevClass =
                event.severity === 'RED' ? 'red' : event.severity === 'YELLOW' ? 'yellow' : 'green';

              return (
                <tr key={event.id} className={`alerts-row alerts-row--${sevClass}`}>
                  <td className="alerts-row__id">{event.id}</td>
                  <td>
                    <span className={`badge-${sevClass} ${event.severity === 'RED' ? 'glow-red' : ''}`}>
                      {event.severity}
                    </span>
                  </td>
                  <td className="alerts-row__detector">{event.detector}</td>
                  <td className="alerts-row__desc">{event.description}</td>
                  <td>
                    <div className="conf-cell">
                      <div className="conf-bar">
                        <div
                          className="conf-bar__fill"
                          style={{
                            width: `${event.confidence * 100}%`,
                            background:
                              event.confidence > 0.85
                                ? 'var(--color-red)'
                                : event.confidence > 0.7
                                ? 'var(--color-yellow)'
                                : 'var(--color-green)',
                          }}
                        />
                      </div>
                      <span>{Math.round(event.confidence * 100)}%</span>
                    </div>
                  </td>
                  <td className="alerts-row__ids">
                    {event.personIds.length > 0 ? (
                      event.personIds.map((id) => (
                        <span key={`${event.id}-${id}`} className="id-chip">
                          {id}
                        </span>
                      ))
                    ) : (
                      <span className="text-secondary">-</span>
                    )}
                  </td>
                  <td className="alerts-row__time">{event.timestamp}</td>
                  <td className="alerts-row__frame">{event.frame.toLocaleString()}</td>
                  <td>
                    <div className="alerts-row__actions">
                      <button className="row-action" title="View Frame">
                        <Eye size={14} />
                      </button>
                      <button className="row-action" title="Export Event" onClick={handleExportCSV}>
                        <Download size={14} />
                      </button>
                    </div>
                  </td>
                </tr>
              );
            })}
            {pageData.length === 0 && (
              <tr>
                <td colSpan={9} className="alerts-empty">
                  {loading ? 'Loading events...' : 'No events match the current filters.'}
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      <div className="alerts-pagination">
        <button className="page-btn" disabled={page <= 1} onClick={() => setPage((p) => p - 1)}>
          <ChevronLeft size={14} /> Prev
        </button>
        <div className="page-numbers">
          {Array.from({ length: totalPages }, (_, i) => i + 1).map((number) => (
            <button
              key={number}
              className={`page-num ${number === page ? 'page-num--active' : ''}`}
              onClick={() => setPage(number)}
            >
              {number}
            </button>
          ))}
        </div>
        <button className="page-btn" disabled={page >= totalPages} onClick={() => setPage((p) => p + 1)}>
          Next <ChevronRight size={14} />
        </button>
      </div>
    </div>
  );
}
