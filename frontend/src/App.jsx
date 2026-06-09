/* Root — routes between What's That Fish? and History. Other pages stub to a
   gentle "coming soon" view so the sidebar still functions. */
import { useState } from "react";
import Sidebar from "./components/Sidebar.jsx";
import MainPage from "./pages/MainPage.jsx";
import HistoryPage from "./pages/HistoryPage.jsx";
import ComingSoon from "./pages/ComingSoon.jsx";

const STUBS = {
  saved:    { title: "Saved",           hint: "Pin notable encounters and study them later." },
  library:  { title: "Species Library", hint: "Browse the full catalogue the model can recognise." },
  settings: { title: "Settings",        hint: "Model, units, sync and account preferences." },
  about:    { title: "About",           hint: "Model card, dataset notes, credits." },
};

export default function App() {
  const [page, setPage] = useState("whats-that-fish");

  let body;
  if (page === "whats-that-fish") body = <MainPage />;
  else if (page === "history")    body = <HistoryPage />;
  else                            body = <ComingSoon {...STUBS[page]} />;

  return (
    <div className="app">
      <Sidebar active={page} onNavigate={setPage} />
      {body}
    </div>
  );
}
