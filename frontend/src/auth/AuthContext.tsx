/* Auth state for the SPA. Holds the Google ID token + resolved user, hydrates
   on load by calling /auth/me, and owns the GIS script lifecycle so any
   component can drop a sign-in button without re-initialising Google.

   Token model: the Google ID token (~1h TTL) is stored in localStorage and sent
   as Bearer. When it expires, the next /auth/me returns 401 and we fall back to
   signed-out — there's no silent refresh (GIS ID tokens aren't refreshable
   without a re-prompt). Good enough until we trade it for a backend session. */
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useState,
  type ReactNode,
} from "react";
import type { UserProfile } from "../api/types";
import { getMe } from "../api/auth";
import { GOOGLE_CLIENT_ID } from "../api/config";
import { AUTH_EXPIRED_EVENT } from "../api/events";

const TOKEN_KEY = "wtf_token";

type AuthStatus = "loading" | "signed-in" | "signed-out";

interface AuthValue {
  user: UserProfile | null;
  status: AuthStatus;
  /** True once the GIS script is loaded + initialised (gate sign-in button render). */
  gisReady: boolean;
  signOut: () => void;
  /** Re-hydrate the user from /auth/me (e.g. after a settings change). */
  refreshUser: () => Promise<void>;
}

const AuthContext = createContext<AuthValue | null>(null);

export function useAuth(): AuthValue {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within <AuthProvider>");
  return ctx;
}

/** Read the stored token for authed API calls (e.g. future /observations). */
export function authToken(): string | null {
  return localStorage.getItem(TOKEN_KEY);
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<UserProfile | null>(null);
  const [status, setStatus] = useState<AuthStatus>("loading");
  const [gisReady, setGisReady] = useState(false);

  // Exchange a fresh Google credential for our user profile.
  const handleCredential = useCallback(async (idToken: string) => {
    localStorage.setItem(TOKEN_KEY, idToken);
    try {
      setUser(await getMe(idToken));
      setStatus("signed-in");
    } catch {
      localStorage.removeItem(TOKEN_KEY);
      setUser(null);
      setStatus("signed-out");
    }
  }, []);

  // Pull the latest profile (preferred name / units may have changed). Silent
  // on failure — a stale-but-present user is better than flicking signed-out.
  const refreshUser = useCallback(async () => {
    const token = localStorage.getItem(TOKEN_KEY);
    if (!token) return;
    try {
      setUser(await getMe(token));
    } catch {
      /* keep current user */
    }
  }, []);

  const signOut = useCallback(() => {
    localStorage.removeItem(TOKEN_KEY);
    setUser(null);
    setStatus("signed-out");
    window.google?.accounts.id.disableAutoSelect();
  }, []);

  // Any authed call that hits 401 (expired token) dispatches this — flip the
  // whole app to signed-out so the sidebar + pages reflect it immediately.
  useEffect(() => {
    const onExpired = () => signOut();
    window.addEventListener(AUTH_EXPIRED_EVENT, onExpired);
    return () => window.removeEventListener(AUTH_EXPIRED_EVENT, onExpired);
  }, [signOut]);

  // Hydrate from a stored token on first load.
  useEffect(() => {
    const token = localStorage.getItem(TOKEN_KEY);
    if (!token) {
      setStatus("signed-out");
      return;
    }
    getMe(token)
      .then((me) => {
        setUser(me);
        setStatus("signed-in");
      })
      .catch(() => {
        localStorage.removeItem(TOKEN_KEY);
        setStatus("signed-out");
      });
  }, []);

  // Load + initialise GIS exactly once. The callback funnels every sign-in
  // (button click or One Tap) through handleCredential.
  useEffect(() => {
    const init = () => {
      if (!window.google) return;
      if (!GOOGLE_CLIENT_ID) {
        // Loud, actionable failure instead of a silently-blank button.
        console.error(
          "[auth] VITE_GOOGLE_CLIENT_ID is unset — create frontend/.env.local and " +
            "restart `npm run dev`. The Google sign-in button cannot render without it.",
        );
        return;
      }
      window.google.accounts.id.initialize({
        client_id: GOOGLE_CLIENT_ID,
        callback: (resp) => handleCredential(resp.credential),
      });
      setGisReady(true);
    };
    if (window.google) {
      init();
      return;
    }
    const script = document.createElement("script");
    script.src = "https://accounts.google.com/gsi/client";
    script.async = true;
    script.onload = init;
    document.head.appendChild(script);
  }, [handleCredential]);

  return (
    <AuthContext.Provider value={{ user, status, gisReady, signOut, refreshUser }}>
      {children}
    </AuthContext.Provider>
  );
}
