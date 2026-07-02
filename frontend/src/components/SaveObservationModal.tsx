/* Save-to-history / Report-wrong-ID modal. Same window, two modes:
   - save:   saves the ACTIVE selected species (chosen in the result cards). A
             disclaimer names exactly what's being saved (scientific + common).
   - report: search the FULL fish/shark species list and pick the correct taxon.
   Then orchestrates: (create or pick) dive → observation → upload photo. */
import { useEffect, useState } from "react";
import type { ImageState } from "./ResultsView";
import type { Prediction, Candidate } from "../api/types";
import { useAuth } from "../auth/AuthContext";
import { createDive, createObservation, listDives, uploadPhoto, type Dive } from "../api/observations";
import { searchSpecies, type TaxonOption } from "../api/taxa";
import SiteAutocomplete from "./SiteAutocomplete";
import { toMeters, unitLabel } from "../lib/units";

interface Props {
  mode: "save" | "report";
  image: ImageState;
  prediction: Prediction;
  /** The active species selected in the result cards — what "save" persists. */
  selectedSpecies: Candidate;
  onClose: () => void;
  onSaved: () => void;
}

/** "Scientific name (Common name)" — common only when we have one. */
function speciesLabel(name: string, common?: string | null): string {
  return common ? `${name} (${common})` : name;
}

export default function SaveObservationModal({
  mode,
  image,
  prediction,
  selectedSpecies,
  onClose,
  onSaved,
}: Props) {
  const { status, user } = useAuth();
  const units = user?.unitSystem ?? "metric";
  const top = prediction.species[0];

  const [dives, setDives] = useState<Dive[]>([]);
  const [diveMode, setDiveMode] = useState<"new" | "existing">("new");
  const [diveId, setDiveId] = useState("");
  const [siteName, setSiteName] = useState("");
  const [divedAt, setDivedAt] = useState("");
  const [depth, setDepth] = useState("");
  const [validated, setValidated] = useState(false);

  // report-mode: search the full species list
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<TaxonOption[]>([]);
  const [correction, setCorrection] = useState<TaxonOption | null>(null);

  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (status !== "signed-in") return;
    listDives()
      .then((d) => {
        setDives(d);
        if (d.length) {
          setDiveMode("existing");
          setDiveId(d[0].id);
        }
      })
      .catch(() => {});
  }, [status]);

  useEffect(() => {
    if (mode !== "report" || query.trim().length < 2) {
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
  }, [query, mode]);

  if (status !== "signed-in") {
    return (
      <Backdrop onClose={onClose}>
        <h3 className="modal__title">Sign in to save</h3>
        <p className="modal__body">Saving to your field log requires signing in with Google (sidebar).</p>
        <div className="modal__actions">
          <button className="btn btn--ghost btn--sm" onClick={onClose}>Close</button>
        </div>
      </Backdrop>
    );
  }

  async function handleSubmit() {
    setBusy(true);
    setError(null);
    try {
      if (mode === "report" && !correction) throw new Error("Select the correct species first.");

      // 1. Resolve the dive.
      let targetDiveId = diveId;
      if (diveMode === "new") {
        const dive = await createDive({
          siteName: siteName || undefined,
          divedAt: divedAt || undefined,
        });
        targetDiveId = dive.id;
      }
      if (!targetDiveId) throw new Error("Pick or create a dive first.");

      // 2. Decide the effective label + provenance.
      let labelStatus: "predicted" | "confirmed" | "corrected";
      let correctedSpeciesIndex: number | null = null;
      let correctedTaxonId: number | null = null;
      if (mode === "report") {
        labelStatus = "corrected";
        correctedTaxonId = correction!.taxonId;
      } else if (selectedSpecies.index !== top.index) {
        // User picked a non-top candidate in the result cards.
        labelStatus = "corrected";
        correctedSpeciesIndex = selectedSpecies.index;
      } else {
        labelStatus = validated ? "confirmed" : "predicted";
      }

      const obs = await createObservation({
        diveId: targetDiveId,
        predictedSpeciesIndex: top.index,
        correctedSpeciesIndex,
        correctedTaxonId,
        labelStatus,
        confidence: selectedSpecies.conf,
        depthM: depth ? toMeters(Number(depth), units) : null,
      });

      // 3. Upload the photo (bytes behind image.url — upload or sample alike).
      const blob = await (await fetch(image.url!)).blob();
      await uploadPhoto({
        observationId: obs.id,
        image: blob,
        bbox: prediction.bbox[0] ?? null,
        predictedSpeciesIndex: top.index,
        confidence: top.conf,
      });

      onSaved();
      onClose();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <Backdrop onClose={onClose}>
      <h3 className="modal__title">
        {mode === "report" ? "Report wrong ID" : "Save to field log"}
      </h3>

      {mode === "save" ? (
        <>
          <p className="modal__disclaimer">
            Saving <strong>{speciesLabel(selectedSpecies.name, selectedSpecies.common)}</strong> to
            your field log. Please make sure you've validated this identification — accurate
            labels may be used to improve the model.
          </p>
          <div className="modal__field">
            <label className="modal__label">Species being saved</label>
            <div className="modal__species">
              <span className="modal__species-name">{speciesLabel(selectedSpecies.name, selectedSpecies.common)}</span>
              <span className="modal__species-conf">{(selectedSpecies.conf * 100).toFixed(1)}%</span>
            </div>
          </div>
          <label className="modal__check">
            <input type="checkbox" checked={validated} onChange={(e) => setValidated(e.target.checked)} />
            I've validated this identification is correct
          </label>
        </>
      ) : (
        <>
          <p className="modal__disclaimer">
            Reporting a wrong ID. Search the full fish &amp; shark species list and pick the
            correct one — it's saved as the corrected label.
          </p>
          <div className="modal__field">
            <label className="modal__label">Correct species</label>
            <input
              className="modal__input"
              placeholder="Search scientific or common name…"
              value={query}
              onChange={(e) => {
                setQuery(e.target.value);
                setCorrection(null);
              }}
            />
            {correction ? (
              <div className="modal__picked">
                Selected: <strong>{speciesLabel(correction.name, correction.commonName)}</strong>
              </div>
            ) : (
              results.length > 0 && (
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
                        {speciesLabel(t.name, t.commonName)}
                      </button>
                    </li>
                  ))}
                </ul>
              )
            )}
          </div>
        </>
      )}

      {/* Dive */}
      <div className="modal__field">
        <label className="modal__label">Dive</label>
        <div className="modal__radio-row">
          <label><input type="radio" checked={diveMode === "new"} onChange={() => setDiveMode("new")} /> New dive</label>
          <label>
            <input type="radio" checked={diveMode === "existing"} disabled={!dives.length} onChange={() => setDiveMode("existing")} /> Existing
          </label>
        </div>
        {diveMode === "existing" ? (
          <select className="modal__select" value={diveId} onChange={(e) => setDiveId(e.target.value)}>
            {dives.map((d) => (
              <option key={d.id} value={d.id}>
                {d.siteName ?? "Untitled dive"} · {d.divedAt ? d.divedAt.slice(0, 10) : "no date"}
              </option>
            ))}
          </select>
        ) : (
          <div className="modal__grid">
            <SiteAutocomplete
              value={siteName}
              onChange={setSiteName}
              placeholder="Dive site (e.g. Tulamben, Bali)"
            />
            <input className="modal__input" type="date" value={divedAt} onChange={(e) => setDivedAt(e.target.value)} />
          </div>
        )}
      </div>

      {/* Depth */}
      <div className="modal__field">
        <label className="modal__label">Depth ({unitLabel(units)})</label>
        <input className="modal__input" type="number" min="0" placeholder={units === "imperial" ? "e.g. 40" : "e.g. 12"} value={depth} onChange={(e) => setDepth(e.target.value)} />
      </div>

      {error && <div className="modal__error">{error}</div>}

      <div className="modal__actions">
        <button className="btn btn--ghost btn--sm" onClick={onClose} disabled={busy}>Cancel</button>
        <button className="btn btn--foam btn--sm" onClick={handleSubmit} disabled={busy}>
          {busy ? "Saving…" : mode === "report" ? "Submit correction" : "Save to history"}
        </button>
      </div>
    </Backdrop>
  );
}

function Backdrop({ children, onClose }: { children: React.ReactNode; onClose: () => void }) {
  return (
    <div className="modal__backdrop" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        {children}
      </div>
    </div>
  );
}
