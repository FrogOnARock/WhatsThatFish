/* Edit a dive's dive-level details: site (with existing-site autocomplete),
   date, and the detailed notes. All three are shared by every sighting on the
   dive, so a disclaimer mirrors the one in EditSightingModal. */
import { useState } from "react";
import type { Dive } from "../api/observations";
import { updateDive } from "../api/observations";
import SiteAutocomplete from "./SiteAutocomplete";

interface Props {
  dive: Dive;
  onClose: () => void;
  onSaved: () => void;
}

export default function DiveEditModal({ dive, onClose, onSaved }: Props) {
  const [siteName, setSiteName] = useState(dive.siteName ?? "");
  const [divedAt, setDivedAt] = useState(dive.divedAt ? dive.divedAt.slice(0, 10) : "");
  const [notes, setNotes] = useState(dive.notes ?? "");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit() {
    setBusy(true);
    setError(null);
    try {
      await updateDive(dive.id, {
        siteName: siteName || undefined,
        divedAt,
        notes,
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
    <div className="modal__backdrop" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <h3 className="modal__title">Edit dive</h3>

        <p className="modal__disclaimer">
          These details are dive-level — changing them updates every sighting logged
          on this dive.
        </p>

        <div className="modal__field">
          <label className="modal__label">Dive site</label>
          <SiteAutocomplete
            value={siteName}
            onChange={setSiteName}
            placeholder="Dive site (e.g. Tulamben, Bali)"
          />
        </div>

        <div className="modal__field">
          <label className="modal__label">Date</label>
          <input
            className="modal__input"
            type="date"
            value={divedAt}
            onChange={(e) => setDivedAt(e.target.value)}
          />
        </div>

        <div className="modal__field">
          <label className="modal__label">Notes</label>
          <textarea
            className="modal__input modal__textarea"
            rows={4}
            placeholder="Conditions, buddies, anything memorable…"
            value={notes}
            onChange={(e) => setNotes(e.target.value)}
          />
        </div>

        {error && <div className="modal__error">{error}</div>}
        <div className="modal__actions">
          <button className="btn btn--ghost btn--sm" onClick={onClose} disabled={busy}>
            Cancel
          </button>
          <button className="btn btn--foam btn--sm" onClick={handleSubmit} disabled={busy}>
            {busy ? "Saving…" : "Save changes"}
          </button>
        </div>
      </div>
    </div>
  );
}
