/* Root — routes between the workspace pages by sidebar selection. */
import { useState } from "react";
import Sidebar, { type PageId } from "./components/Sidebar";
import ColdStartBanner from "./components/ColdStartBanner";
import MainPage from "./pages/MainPage";
import HistoryPage from "./pages/HistoryPage";
import DivesPage from "./pages/DivesPage";
import SettingsPage from "./pages/SettingsPage";
import SpeciesLibrary from "./pages/LibraryPage";
import LoginPage from "./pages/LoginPage";
import AccountPage from "./pages/AccountPage";

export default function App() {
  const [page, setPage] = useState<PageId>("whats-that-fish");
  // Mobile-only drawer state. On desktop the sidebar is always in-flow and this
  // flag is inert; on narrow screens it slides the off-canvas drawer in/out.
  const [navOpen, setNavOpen] = useState(false);

  // Navigating always closes the drawer so the chosen page is visible on mobile.
  const navigate = (id: PageId) => {
    setPage(id);
    setNavOpen(false);
  };

  let body;
  if (page === "whats-that-fish") body = <MainPage />;
  else if (page === "library")    body = <SpeciesLibrary />;
  else if (page === "history")    body = <HistoryPage />;
  else if (page === "dives")      body = <DivesPage />;
  else if (page === "settings")   body = <SettingsPage />;
  else if (page === "login")      body = <LoginPage onNavigate={navigate} />;
  else                            body = <AccountPage onNavigate={navigate} />;

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
      <Sidebar
        active={page}
        onNavigate={navigate}
        open={navOpen}
        onClose={() => setNavOpen(false)}
      />
      {navOpen && (
        <div
          className="sidebar-backdrop"
          onClick={() => setNavOpen(false)}
          aria-hidden="true"
        />
      )}
      {body}
    </div>
  );
}
