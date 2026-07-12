import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import App from "./App";
import { AuthProvider } from "./auth/AuthContext";
import { BackendStatusProvider } from "./api/backendStatus";
import { IdentifyProvider } from "./pages/IdentifyContext";
import "./styles/tokens.css";
import "./styles/app.css";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <BrowserRouter>
      <AuthProvider>
        <BackendStatusProvider>
          {/* Above <Routes>, so the current identification survives navigation. */}
          <IdentifyProvider>
            <App />
          </IdentifyProvider>
        </BackendStatusProvider>
      </AuthProvider>
    </BrowserRouter>
  </React.StrictMode>,
);
