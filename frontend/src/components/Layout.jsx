import { Outlet } from 'react-router-dom';
import Sidebar from './Sidebar.jsx';
import Topbar from './Topbar.jsx';
import styles from './Layout.module.css';

/**
 * Shell layout composed of Sidebar + Topbar + page content area.
 * <Outlet /> is replaced by the matched child route component.
 */
export default function Layout() {
  return (
    <div className={styles.shell}>
      <Sidebar />
      <div className={styles.main}>
        <Topbar />
        <main className={styles.content}>
          <Outlet />
        </main>
      </div>
    </div>
  );
}
