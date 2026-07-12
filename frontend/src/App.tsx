/* Root shell — persistent chrome (cold-start banner, mobile topbar, sidebar)
   around the routed page body. Pages are real routes now, so browser back/forward
   and deep links work; the sidebar drawer state is the only local UI state left. */
import { useEffect, useState } from "react";
import { Routes, Route, Navigate, useLocation } from "react-router-dom";
import Sidebar from "./components/Sidebar";
import ColdStartBanner from "./components/ColdStartBanner";
import MainPage from "./pages/MainPage";
import HistoryPage from "./pages/HistoryPage";
import DivesPage from "./pages/DivesPage";
import SettingsPage from "./pages/SettingsPage";
import SpeciesLibrary from "./pages/LibraryPage";
import LoginPage from "./pages/LoginPage";
import AccountPage from "./pages/AccountPage";
import { ROUTES } from "./routes";

export default function App() {
  // Mobile-only drawer state. On desktop the sidebar is always in-flow and this
  // flag is inert; on narrow screens it slides the off-canvas drawer in/out.
  const [navOpen, setNavOpen] = useState(false);
  const location = useLocation();

  // Any navigation (sidebar tap, back/forward, deep link) closes the drawer so
  // the chosen page is visible on mobile.
  useEffect(() => {
    setNavOpen(false);
  }, [location.pathname]);

  return (
    <div className="app">
      <ColdStartBanner />
      <div className="mobile-topbar">
        <button
          className="mobile-topbar__burger"
          onClick={() => setNavOpen(true)}
          aria-label="Open navigation"
          aria-expanded={navOpen}
        >
          <span /><span /><span />
        </button>
        <div className="mobile-topbar__brand">
          What's <em>that</em> Fish?
        </div>
      </div>
      <Sidebar open={navOpen} onClose={() => setNavOpen(false)} />
      {navOpen && (
        <div
          className="sidebar-backdrop"
          onClick={() => setNavOpen(false)}
          aria-hidden="true"
        />
      )}
      <Routes>
        <Route path={ROUTES.identify} element={<MainPage />} />
        <Route path={ROUTES.fieldLog} element={<HistoryPage />} />
        <Route path={`${ROUTES.fieldLog}/:taxonId`} element={<HistoryPage />} />
        <Route path={ROUTES.dives} element={<DivesPage />} />
        <Route path={ROUTES.library} element={<SpeciesLibrary />} />
        <Route path={`${ROUTES.library}/:speciesId`} element={<SpeciesLibrary />} />
        <Route path={ROUTES.settings} element={<SettingsPage />} />
        <Route path={ROUTES.login} element={<LoginPage />} />
        <Route path={ROUTES.account} element={<AccountPage />} />
        {/* Unknown paths fall back to the identify page. */}
        <Route path="*" element={<Navigate to={ROUTES.identify} replace />} />
      </Routes>
    </div>
  );
}
