/* Edit a logged sighting. Two scopes in one window:
   - sighting-level (this row only): re-label via the species picker, depth.
   - dive-level (shared): site + date — a disclaimer notes this affects every
     sighting on the dive, since site/time live on the dive, not the observation. */
import { useEffect, useState } from "react";
import type { FieldSighting } from "../api/history";
import { updateObservation, updateDive } from "../api/observations";
import { searchSpecies, type TaxonOption } from "../api/taxa";
import SiteAutocomplete from "./SiteAutocomplete";
import { useAuth } from "../auth/AuthContext";
import { toDisplayDepth, toMeters, unitLabel } from "../lib/units";

interface Props {
  sighting: FieldSighting;
  currentLabel: string;
  onClose: () => void;
  onSaved: () => void;
}

export default function EditSightingModal({ sighting, currentLabel, onClose, onSaved }: Props) {
  const { user } = useAuth();
  const units = user?.unitSystem ?? "metric";
  const [depth, setDepth] = useState(
    sighting.depthM != null ? String(toDisplayDepth(sighting.depthM, units)) : "",
  );
  const [siteName, setSiteName] = useState(sighting.siteName ?? "");
  const [divedAt, setDivedAt] = useState(sighting.divedAt ? sighting.divedAt.slice(0, 10) : "");
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<TaxonOption[]>([]);
  const [correction, setCorrection] = useState<TaxonOption | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (query.trim().length < 2 || correction) {
      setResults([]);
      return;
    }
    let cancelled = false;
    searchSpecies(query)
      .then((r) => !cancelled && setResults(r))
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [query, correction]);

  async function handleSubmit() {
    setBusy(true);
    setError(null);
    try {
      await updateObservation(sighting.observationId, {
        depthM: depth ? toMeters(Number(depth), units) : null,
        ...(correction
          ? { correctedTaxonId: correction.taxonId, labelStatus: "corrected" as const }
          : {}),
      });
      await updateDive(sighting.diveId, { siteName: siteName || undefined, divedAt });
      onSaved();
      onClose();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="modal__backdrop" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <h3 className="modal__title">Edit sighting</h3>

        <div className="modal__field">
          <label className="modal__label">Species (currently {currentLabel})</label>
          <input
            className="modal__input"
            placeholder="Search to re-label…"
            value={correction ? correction.name : query}
            onChange={(e) => {
              setQuery(e.target.value);
              setCorrection(null);
            }}
          />
          {!correction && results.length > 0 && (
            <ul className="modal__results">
              {results.map((t) => (
                <li key={t.taxonId}>
                  <button
                    type="button"
                    className="modal__result"
                    onClick={() => {
                      setCorrection(t);
                      setQuery(t.name);
                    }}
                  >
                    {t.name}
                    {t.commonName ? ` (${t.commonName})` : ""}
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>

        <div className="modal__field">
          <label className="modal__label">Depth ({unitLabel(units)})</label>
          <input
            className="modal__input"
            type="number"
            min="0"
            value={depth}
            onChange={(e) => setDepth(e.target.value)}
          />
        </div>

        <p className="modal__disclaimer">
          Site &amp; date are dive-level — changing them updates every sighting on this dive.
        </p>
        <div className="modal__grid">
          <SiteAutocomplete
            value={siteName}
            onChange={setSiteName}
            placeholder="Dive site"
          />
          <input
            className="modal__input"
            type="date"
            value={divedAt}
            onChange={(e) => setDivedAt(e.target.value)}
          />
        </div>

        {error && <div className="modal__error">{error}</div>}
        <div className="modal__actions">
          <button className="btn btn--ghost btn--sm" onClick={onClose} disabled={busy}>Cancel</button>
          <button className="btn btn--foam btn--sm" onClick={handleSubmit} disabled={busy}>
            {busy ? "Saving…" : "Save changes"}
          </button>
        </div>
      </div>
    </div>
  );
}
