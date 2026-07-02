/* Root — routes between the workspace pages by sidebar selection. */
import { useState } from "react";
import Sidebar, { type PageId } from "./components/Sidebar";
import MainPage from "./pages/MainPage";
import HistoryPage from "./pages/HistoryPage";
import DivesPage from "./pages/DivesPage";
import SettingsPage from "./pages/SettingsPage";
import SpeciesLibrary from "./pages/LibraryPage";
import LoginPage from "./pages/LoginPage";
import AccountPage from "./pages/AccountPage";

export default function App() {
  const [page, setPage] = useState<PageId>("whats-that-fish");

  let body;
  if (page === "whats-that-fish") body = <MainPage />;
  else if (page === "library")    body = <SpeciesLibrary />;
  else if (page === "history")    body = <HistoryPage />;
  else if (page === "dives")      body = <DivesPage />;
  else if (page === "settings")   body = <SettingsPage />;
  else if (page === "login")      body = <LoginPage onNavigate={setPage} />;
  else                            body = <AccountPage onNavigate={setPage} />;

  return (
    <div className="app">
      <Sidebar active={page} onNavigate={setPage} />
      {body}
    </div>
  );
}
