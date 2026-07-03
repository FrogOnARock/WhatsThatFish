/* Slim app-wide strip that explains a cold start (or an unreachable backend) so
   a 30s first-load wait reads as intentional, not broken. Renders nothing when
   the backend is online/unknown — no flash on the warm path. */
import { useBackendStatus } from "../api/backendStatus";

export default function ColdStartBanner() {
  const status = useBackendStatus();
  if (status !== "warming" && status !== "down") return null;

  const down = status === "down";
  return (
    <div className={`cold-banner ${down ? "cold-banner--down" : ""}`} role="status" aria-live="polite">
      <span className="analyzing__pulse" />
      {down
        ? "Can't reach the model service — it may be starting up. Retrying…"
        : "Waking the model — the first request after a period of inactivity can take ~30s."}
    </div>
  );
}
