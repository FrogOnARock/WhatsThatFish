/* Shared cold-start signal for the whole app.

   Cloud Run scales to zero, so the first request after idle waits ~20–60s while
   the instance boots. This context turns that into one legible piece of state
   (`warming`) that the banner, the model pill, and the analyzing overlay all read
   — instead of each feature rediscovering the cold start on its own.

   Two inputs feed it:
   1. A warm-up `GET /health` fired on mount — kicks the instance awake BEFORE the
      user acts and gives us an authoritative online/down verdict.
   2. BACKEND_OK / BACKEND_SLOW events from the http layer (apiFetch), so any
      in-flight call keeps the signal fresh without importing React there.

   `warming` is only ENTERED from a non-online state: a confirmed-online backend
   won't be flipped back by a merely-slow call (a real 3–30s warm inference must
   not cry "waking up"). The on-load ping is the reliable cold-start trigger. */
import {
  createContext,
  useContext,
  useEffect,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { API_BASE } from "./config";
import { apiFetch, TIMEOUT } from "./http";
import { BACKEND_OK_EVENT, BACKEND_SLOW_EVENT } from "./events";

export type BackendStatus = "unknown" | "warming" | "online" | "down";

const BackendStatusContext = createContext<BackendStatus>("unknown");

export function useBackendStatus(): BackendStatus {
  return useContext(BackendStatusContext);
}

const PING_RETRIES = 4; // re-ping a cold/unreachable instance a few times before "down"

export function BackendStatusProvider({ children }: { children: ReactNode }) {
  // Start "warming": we ping immediately, and the honest state on first paint
  // after idle IS waking. A successful ping flips it to online within one RTT.
  const [status, setStatus] = useState<BackendStatus>("warming");
  const statusRef = useRef<BackendStatus>(status);
  statusRef.current = status;

  // Warm-up ping loop: succeed → online; keep failing → down (still recoverable
  // via a later OK event from a user action).
  useEffect(() => {
    let cancelled = false;
    const ping = async (attempt: number): Promise<void> => {
      try {
        const res = await apiFetch(`${API_BASE}/health`, {}, { timeoutMs: TIMEOUT.HEALTH, retries: 0 });
        if (cancelled) return;
        if (res.ok) { setStatus("online"); return; }
        throw new Error(String(res.status));
      } catch {
        if (cancelled) return;
        if (attempt >= PING_RETRIES) { setStatus("down"); return; }
        setStatus("warming");
        await new Promise((r) => setTimeout(r, Math.min(1000 * 2 ** attempt, 8000)));
        if (!cancelled) return ping(attempt + 1);
      }
    };
    void ping(0);
    return () => { cancelled = true; };
  }, []);

  // Any successful call confirms the instance is up; a slow/failed one nudges us
  // to warming — but only if we haven't already confirmed online (see file note).
  useEffect(() => {
    const onOk = () => setStatus("online");
    const onSlow = () => {
      if (statusRef.current !== "online") setStatus("warming");
    };
    window.addEventListener(BACKEND_OK_EVENT, onOk);
    window.addEventListener(BACKEND_SLOW_EVENT, onSlow);
    return () => {
      window.removeEventListener(BACKEND_OK_EVENT, onOk);
      window.removeEventListener(BACKEND_SLOW_EVENT, onSlow);
    };
  }, []);

  return (
    <BackendStatusContext.Provider value={status}>
      {children}
    </BackendStatusContext.Provider>
  );
}
