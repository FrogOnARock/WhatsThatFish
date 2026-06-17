/* Root — routes between What's That Fish? and History. Other pages stub to a
   gentle "coming soon" view so the sidebar still functions. */
import { useState } from "react";
import Sidebar, { type PageId } from "./components/Sidebar";
import MainPage from "./pages/MainPage";
import HistoryPage from "./pages/HistoryPage";
import ComingSoon from "./pages/ComingSoon";
import SpeciesLibrary from "./pages/LibraryPage"

const STUBS: Partial<Record<PageId, { title: string; hint: string }>> = {
  saved:    { title: "Saved",           hint: "Pin notable encounters and study them later." },
  settings: { title: "Settings",        hint: "Model, units, sync and account preferences." },
  about:    { title: "About",           hint: "Model card, dataset notes, credits." },
};

export default function App() {
  const [page, setPage] = useState<PageId>("whats-that-fish");

  let body;
  if (page === "whats-that-fish") body = <MainPage />;
  else if (page === "library")    body = <SpeciesLibrary />;
  else if (page === "history")    body = <HistoryPage />;
  else                            body = <ComingSoon {...STUBS[page]!} />;

  return (
    <div className="app">
      <Sidebar active={page} onNavigate={setPage} />
      {body}
    </div>
  );
}
