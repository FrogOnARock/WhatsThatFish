/* Shared fetch helpers so every call handles cold-start + failure the same way:
   - a per-call TIMEOUT (AbortController) so a Cloud-Run cold start that HANGS the
     socket becomes a catchable BackendTimeoutError instead of an endless spinner.
   - auto-retry x1 on transient 5xx / timeout (the scale-from-zero window) with a
     short backoff, so a booting instance self-heals without a user click.
   - BACKEND_OK / BACKEND_SLOW events so the BackendStatus context reflects
     warming/online WITHOUT this module importing React (same trick as 401 below).
   - 401 → clear, human "session expired" + an app-wide event AuthContext listens
     for to sign the user out.
   - other errors → surface the backend's { detail } message, not a bare status. */
import { authToken } from "../auth/AuthContext";
import { AUTH_EXPIRED_EVENT, BACKEND_OK_EVENT, BACKEND_SLOW_EVENT } from "./events";

export class AuthExpiredError extends Error {
  constructor() {
    super("Your session has expired — please sign in again.");
    this.name = "AuthExpiredError";
  }
}

/** Thrown when a request exceeds its timeout budget — during a cold start this
    means the instance is still booting; the UI can message "waking up" + retry. */
export class BackendTimeoutError extends Error {
  constructor() {
    super("The service is taking longer than usual — it may be waking up. Please retry.");
    this.name = "BackendTimeoutError";
  }
}

/** Per-call timeout budgets (ms). A real cold inference is slow, so /predict gets
    a much longer leash than metadata calls. */
export const TIMEOUT = {
  META: 15_000,
  PREDICT: 90_000,
  HEALTH: 10_000,
} as const;

const RETRYABLE = [502, 503, 504];
const SLOW_AFTER_MS = 3_000; // flag "warming" if a call is still pending past this

interface ApiFetchOpts {
  timeoutMs?: number;
  retries?: number;
  retryOn?: number[];
}

const emit = (name: string) => window.dispatchEvent(new Event(name));
const sleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));

/** fetch with a timeout, transient-5xx/timeout retry, and cold-start signalling.
    Callers pick a timeout preset + retry count; everything else (events, abort,
    backoff) is uniform. Throws BackendTimeoutError on expiry; network/HTTP errors
    propagate as usual (raiseForStatus turns non-ok bodies into friendly errors). */
export async function apiFetch(
  input: string,
  init: RequestInit = {},
  opts: ApiFetchOpts = {},
): Promise<Response> {
  const { timeoutMs = TIMEOUT.META, retries = 0, retryOn = RETRYABLE } = opts;

  for (let attempt = 0; ; attempt++) {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
    // Independent "still waiting" nudge — flips the app to `warming` well before
    // the hard timeout so the banner appears during the boot, not only after.
    const slowId = setTimeout(() => emit(BACKEND_SLOW_EVENT), SLOW_AFTER_MS);
    try {
      const res = await fetch(input, { ...init, signal: controller.signal });
      clearTimeout(timeoutId);
      clearTimeout(slowId);

      if (retryOn.includes(res.status) && attempt < retries) {
        emit(BACKEND_SLOW_EVENT);
        await sleep(400 * 2 ** attempt); // 400ms → 800ms → …
        continue;
      }

      // A response < 500 means the instance is up and serving (even a 401/404).
      if (res.status < 500) emit(BACKEND_OK_EVENT);
      else emit(BACKEND_SLOW_EVENT);

      if (res.status === 401) {
        window.dispatchEvent(new Event(AUTH_EXPIRED_EVENT));
        throw new AuthExpiredError();
      }
      return res;
    } catch (err) {
      clearTimeout(timeoutId);
      clearTimeout(slowId);
      if (err instanceof AuthExpiredError) throw err;

      // AbortError = our timeout fired; otherwise a network-level failure
      // (connection refused / DNS / offline). Both are "instance not answering".
      emit(BACKEND_SLOW_EVENT);
      const timedOut = err instanceof DOMException && err.name === "AbortError";
      if (attempt < retries) {
        await sleep(400 * 2 ** attempt);
        continue;
      }
      throw timedOut ? new BackendTimeoutError() : err;
    }
  }
}

/** fetch + attach the Bearer token, routed through apiFetch so authed calls
    inherit timeout/retry/cold-start events. On 401, apiFetch already notified the
    app and threw AuthExpiredError. */
export async function authedFetch(
  input: string,
  init: RequestInit = {},
  opts: ApiFetchOpts = {},
): Promise<Response> {
  const token = authToken();
  return apiFetch(
    input,
    {
      ...init,
      headers: {
        ...(init.headers ?? {}),
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
      },
    },
    opts,
  );
}

/** Throw an Error carrying the backend's `detail` (string or validation list)
    when the response isn't ok. A transient 5xx (the cold-start tail) gets a
    friendly "waking up" message instead of a raw status. */
export async function raiseForStatus(res: Response, fallback: string): Promise<void> {
  if (res.ok) return;
  let detail: string | null = null;
  try {
    const body = await res.json();
    if (body?.detail) {
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    }
  } catch {
    /* non-JSON body */
  }
  if (!detail && RETRYABLE.includes(res.status)) {
    detail = "The service is waking up — please retry in a moment.";
  }
  throw new Error(detail ?? `${fallback} (${res.status})`);
}
