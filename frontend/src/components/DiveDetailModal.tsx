/* Expanded view of a single dive: site, date, GPS, observation count, the
   detailed notes, and the species logged on that dive. Read-only — the Edit
   button hands off to DiveEditModal. */
import type { Dive } from "../api/observations";

function fmtDate(iso: string | null): string {
  if (!iso) return "No date";
  return new Date(iso).toLocaleDateString("en-US", {
    weekday: "short",
    day: "2-digit",
    month: "short",
    year: "numeric",
  });
}

function speciesLabel(name: string | null, common: string | null): string {
  if (common && name) return `${name} (${common})`;
  return common ?? name ?? "Unknown species";
}

interface Props {
  dive: Dive;
  onClose: () => void;
  onEdit: () => void;
}

export default function DiveDetailModal({ dive, onClose, onEdit }: Props) {
  const hasGps = dive.gpsLat != null && dive.gpsLng != null;
  return (
    <div className="modal__backdrop" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <h3 className="modal__title">{dive.siteName ?? "Untitled dive"}</h3>

        <div className="dive-detail__meta">
          <div className="dive-detail__meta-item">
            <span className="dive-detail__meta-label">Date</span>
            <span>{fmtDate(dive.divedAt)}</span>
          </div>
          <div className="dive-detail__meta-item">
            <span className="dive-detail__meta-label">Sightings</span>
            <span>{dive.observationCount}</span>
          </div>
          <div className="dive-detail__meta-item">
            <span className="dive-detail__meta-label">Location</span>
            <span>
              {hasGps
                ? `${dive.gpsLat!.toFixed(4)}, ${dive.gpsLng!.toFixed(4)}`
                : "—"}
            </span>
          </div>
        </div>

        <div className="modal__field">
          <label className="modal__label">Notes</label>
          <p className="dive-detail__notes">
            {dive.notes?.trim() ? dive.notes : "No notes for this dive."}
          </p>
        </div>

        <div className="modal__field">
          <label className="modal__label">
            Species logged ({dive.species.length})
          </label>
          {dive.species.length > 0 ? (
            <ul className="dive-detail__species">
              {dive.species.map((s) => (
                <li key={s.taxonId} className="dive-detail__species-chip">
                  {speciesLabel(s.name, s.commonName)}
                </li>
              ))}
            </ul>
          ) : (
            <p className="dive-detail__notes">No species logged on this dive yet.</p>
          )}
        </div>

        <div className="modal__actions">
          <button className="btn btn--ghost btn--sm" onClick={onClose}>
            Close
          </button>
          <button className="btn btn--foam btn--sm" onClick={onEdit}>
            Edit dive
          </button>
        </div>
      </div>
    </div>
  );
}
