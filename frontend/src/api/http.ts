/* Shared fetch helpers so every authed call handles failure the same way:
   - 401 → clear, human "session expired" message + an app-wide event that
     AuthContext listens for to sign the user out (so the UI reflects it).
   - other errors → surface the backend's { detail } message, not a bare status. */
import { authToken } from "../auth/AuthContext";
import { AUTH_EXPIRED_EVENT } from "./events";

export class AuthExpiredError extends Error {
  constructor() {
    super("Your session has expired — please sign in again.");
    this.name = "AuthExpiredError";
  }
}

/** fetch + attach the Bearer token. On 401, notify the app and throw a friendly
    error instead of letting a raw "401" bubble up to the user. */
export async function authedFetch(input: string, init: RequestInit = {}): Promise<Response> {
  const token = authToken();
  const res = await fetch(input, {
    ...init,
    headers: {
      ...(init.headers ?? {}),
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
  });
  if (res.status === 401) {
    window.dispatchEvent(new Event(AUTH_EXPIRED_EVENT));
    throw new AuthExpiredError();
  }
  return res;
}

/** Throw an Error carrying the backend's `detail` (string or validation list)
    when the response isn't ok. */
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
  throw new Error(detail ?? `${fallback} (${res.status})`);
}
