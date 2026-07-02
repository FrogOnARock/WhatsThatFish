/* Settings — edit the app-owned profile (preferred name + units) and show the
   diver's summary stats. The Google-sourced name/email are read-only (the login
   sync overwrites them), so editing lives in preferred_name instead. */
import { useEffect, useState } from "react";
import { useAuth } from "../auth/AuthContext";
import { updateSettings } from "../api/auth";
import { getUserStats, type UserStats } from "../api/userStats";
import StatPill from "../components/history/StatPill";
import type { UnitSystem } from "../api/types";

export default function SettingsPage() {
  const { user, status, refreshUser } = useAuth();

  const [preferredName, setPreferredName] = useState("");
  const [unitSystem, setUnitSystem] = useState<UnitSystem>("metric");
  const [stats, setStats] = useState<UserStats | null>(null);
  const [busy, setBusy] = useState(false);
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Seed the form from the loaded profile.
  useEffect(() => {
    if (user) {
      setPreferredName(user.preferredName ?? "");
      setUnitSystem(user.unitSystem);
    }
  }, [user]);

  useEffect(() => {
    if (status !== "signed-in") {
      setStats(null);
      return;
    }
    let cancelled = false;
    getUserStats()
      .then((s) => !cancelled && setStats(s))
      .catch(() => !cancelled && setStats(null));
    return () => {
      cancelled = true;
    };
  }, [status]);

  async function handleSave() {
    setBusy(true);
    setError(null);
    setSaved(false);
    try {
      await updateSettings({ preferredName, unitSystem });
      await refreshUser();
      setSaved(true);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  if (status === "loading") return <main className="main" />;

  if (status !== "signed-in" || !user) {
    return (
      <main className="main">
        <div className="main__inner">
          <div className="auth-card">
            <h1 className="page-header__title">
              Your <em>settings</em>
            </h1>
            <p className="page-header__subtitle">
              Sign in to manage your profile and see your diving stats.
            </p>
          </div>
        </div>
      </main>
    );
  }

  const dirty =
    preferredName !== (user.preferredName ?? "") || unitSystem !== user.unitSystem;

  return (
    <main className="main">
      <div className="main__inner">
        <header className="page-header">
          <div>
            <div className="page-header__crumb">Workspace · Settings</div>
            <h1 className="page-header__title">
              Your <em>settings</em>
            </h1>
            <p className="page-header__subtitle">
              Personalise how you appear and which units we show — your diving
              totals at a glance.
            </p>
          </div>
        </header>

        <div className="log-stats">
          <StatPill label="dives" value={stats?.dives ?? "—"} />
          <StatPill label="observations" value={stats?.observations ?? "—"} />
          <StatPill label="unique species" value={stats?.uniqueSpecies ?? "—"} />
        </div>

        <div className="settings-card">
          <div className="modal__field">
            <label className="modal__label">Display name</label>
            <input
              className="modal__input"
              placeholder={user.displayName ?? "Your name"}
              value={preferredName}
              onChange={(e) => {
                setPreferredName(e.target.value);
                setSaved(false);
              }}
            />
            <p className="settings-hint">
              Overrides your Google name across the app. Leave blank to use{" "}
              <strong>{user.displayName ?? "your Google name"}</strong>.
            </p>
          </div>

          <div className="modal__field">
            <label className="modal__label">Email</label>
            <input className="modal__input" value={user.email ?? ""} disabled />
            <p className="settings-hint">Managed by your Google account.</p>
          </div>

          <div className="modal__field">
            <label className="modal__label">Units</label>
            <select
              className="modal__select"
              value={unitSystem}
              onChange={(e) => {
                setUnitSystem(e.target.value as UnitSystem);
                setSaved(false);
              }}
            >
              <option value="metric">Metric (meters)</option>
              <option value="imperial">Imperial (feet)</option>
            </select>
          </div>

          {error && <div className="modal__error">{error}</div>}
          <div className="settings-actions">
            {saved && !dirty && <span className="settings-saved">Saved ✓</span>}
            <button
              className="btn btn--foam btn--sm"
              onClick={handleSave}
              disabled={busy || !dirty}
            >
              {busy ? "Saving…" : "Save changes"}
            </button>
          </div>
        </div>
      </div>
    </main>
  );
}
