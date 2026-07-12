/* Sticky right-hand panel: hero photo, taxonomy crumb, stats, sightings log,
   and a gallery of every photo. Each photo can be enlarged, set as the species'
   card image, re-run through inference, or deleted; each sighting can be edited
   or deleted. Driven by the real field-log shape. */
import { useState } from "react";
import { useNavigate } from "react-router-dom";
import FishPlaceholder from "../FishPlaceholder";
import AuthedImage from "../AuthedImage";
import StatPill from "./StatPill";
import EditSightingModal from "../EditSightingModal";
import ConfirmModal from "../ConfirmModal";
import type { FieldSpecies, FieldSighting, LogPhoto } from "../../api/history";
import { photoImageEndpoint } from "../../api/history";
import { deleteObservation, deletePhoto, setHeroPhoto } from "../../api/observations";
import { authedFetch } from "../../api/http";
import { getPredictionBlob } from "../../api/client";
import { useIdentifySession } from "../../pages/IdentifyContext";
import { ROUTES } from "../../routes";
import { useAuth } from "../../auth/AuthContext";
import { formatDepth, toDisplayDepth, unitLabel } from "../../lib/units";

function fmtDate(iso: string | null): string {
  if (!iso) return "—";
  return new Date(iso).toLocaleDateString("en-US", {
    day: "2-digit",
    month: "short",
    year: "numeric",
  });
}

/** The species' card image: the user-chosen hero, else the first photo. */
function heroPhotoId(photos: LogPhoto[]): string | null {
  return (photos.find((p) => p.isHero) ?? photos[0])?.id ?? null;
}

export default function DetailPanel({
  sp,
  onChanged,
}: {
  sp: FieldSpecies;
  onChanged: () => void;
}) {
  const { user } = useAuth();
  const identify = useIdentifySession();
  const navigate = useNavigate();
  const units = user?.unitSystem ?? "metric";
  const [zoom, setZoom] = useState<string | null>(null);
  const [editing, setEditing] = useState<FieldSighting | null>(null);
  const [deletingSighting, setDeletingSighting] = useState<FieldSighting | null>(null);
  const [deletingPhoto, setDeletingPhoto] = useState<LogPhoto | null>(null);
  const [reinferId, setReinferId] = useState<string | null>(null);
  // The photo whose action toolbar (View / Set Display / Re-run / Delete) is
  // shown in the section head. Clicking a photo selects it.
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const currentLabel = sp.commonName ?? sp.species ?? "this species";
  const sightings = sp.sightings;
  const sites = new Set(sightings.map((s) => s.siteName).filter(Boolean)).size;
  const depths = sightings
    .map((s) => s.depthM)
    .filter((d): d is number => d != null);
  const depthLabel = depths.length
    ? `${toDisplayDepth(Math.min(...depths), units)}–${toDisplayDepth(
        Math.max(...depths),
        units,
      )} ${unitLabel(units)}`
    : "—";
  const photos = sightings.flatMap((s) => s.photos);
  const hero = heroPhotoId(photos);
  // Resolve the selected photo fresh each render so a reload (delete / re-label)
  // that removes it collapses the toolbar rather than acting on a stale id.
  const selected = photos.find((p) => p.id === selectedId) ?? null;

  async function handleSetHero(photoId: string) {
    setError(null);
    try {
      await setHeroPhoto(photoId);
      onChanged();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  // Re-run inference: fetch the stored bytes back, seed the identify session, and
  // send the user to the main page where the analyzing → result flow plays out.
  async function handleReinfer(photoId: string) {
    setError(null);
    setReinferId(photoId);
    try {
      const res = await authedFetch(photoImageEndpoint(photoId));
      if (!res.ok) throw new Error("Couldn't load the photo for re-inference");
      const blob = await res.blob();
      identify.reset();
      const url = URL.createObjectURL(blob);
      identify.setImage({
        kind: "file",
        filename: `${currentLabel} · field-log photo`,
        size: `${(blob.size / (1024 * 1024)).toFixed(1)} MB`,
        url,
      });
      navigate(ROUTES.identify);
      // Kick the inference after navigation; the context lives above the router
      // so MainPage renders straight into the analyzing state.
      void identify.runInference(() => getPredictionBlob(blob));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setReinferId(null);
    }
  }

  return (
    <aside className="pdx-detail">
      <div className="pdx-detail__hero">
        {hero ? (
          <AuthedImage photoId={hero} className="pdx-detail__hero-img" />
        ) : (
          <FishPlaceholder hue={sp.taxonId % 360} caption={sp.species ?? ""} large />
        )}
      </div>

      <div className="pdx-detail__title-block">
        <div className="pdx-detail__crumb">
          <span>{sp.family}</span>
          <span className="pdx-detail__crumb-sep">›</span>
          <span>{sp.genus}</span>
          <span className="pdx-detail__crumb-sep">›</span>
          <span className="pdx-detail__crumb-sp">{sp.species?.split(" ")[1] ?? ""}</span>
        </div>
        <h2 className="pdx-detail__common">{sp.commonName ?? sp.species}</h2>
        <div className="pdx-detail__sci">{sp.species}</div>
      </div>

      <div className="pdx-detail__stats">
        <StatPill label="sightings" value={sp.sightingCount} />
        <StatPill label="dive sites" value={sites} />
        <StatPill label="depth" value={depthLabel} />
        <StatPill label="photos" value={photos.length} />
      </div>

      {error && <div className="modal__error">{error}</div>}

      <div className="pdx-detail__section">
        <div className="pdx-detail__section-head pdx-detail__section-head--center">
          <h4>Sightings log</h4>
        </div>
        <table className="pdx-table">
          <thead>
            <tr>
              <th>Date</th>
              <th>Site</th>
              <th className="pdx-table__num">Depth</th>
              <th>Label</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {sightings.map((s) => (
              <tr key={s.observationId}>
                <td className="pdx-table__date">{fmtDate(s.divedAt)}</td>
                <td>
                  <div className="pdx-table__site">{s.siteName ?? "—"}</div>
                </td>
                <td className="pdx-table__num">{formatDepth(s.depthM, units)}</td>
                <td>
                  <span className={`label-tag label-tag--${s.labelStatus}`}>
                    {s.labelStatus}
                  </span>
                </td>
                <td className="pdx-table__num">
                  <div className="pdx-table__actions">
                    <button className="pdx-table__edit" onClick={() => setEditing(s)}>
                      Edit
                    </button>
                    <button
                      className="pdx-table__edit pdx-table__edit--danger"
                      onClick={() => setDeletingSighting(s)}
                    >
                      Delete
                    </button>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="pdx-detail__section">
        <div className="pdx-detail__section-head">
          <h4>Photos</h4>
          {selected ? (
            <div className="pdx-photo-actions">
              <button
                className="pdx-photo-actions__btn"
                onClick={() => setZoom(selected.id)}
                title="Enlarge this photo"
              >
                View
              </button>
              <button
                className="pdx-photo-actions__btn"
                disabled={selected.isHero}
                onClick={() => handleSetHero(selected.id)}
                title="Use as the card image for this species"
              >
                {selected.isHero ? "★ Display" : "Set Display"}
              </button>
              <button
                className="pdx-photo-actions__btn"
                disabled={reinferId === selected.id}
                onClick={() => handleReinfer(selected.id)}
                title="Re-run the model on this photo"
              >
                {reinferId === selected.id ? "…" : "Re-run"}
              </button>
              <button
                className="pdx-photo-actions__btn pdx-photo-actions__btn--danger"
                onClick={() => setDeletingPhoto(selected)}
                title="Delete this photo"
              >
                Delete
              </button>
            </div>
          ) : (
            <span className="pdx-detail__section-meta">
              {photos.length} photo{photos.length === 1 ? "" : "s"}
              {photos.length > 0 && " · tap to select"}
            </span>
          )}
        </div>
        <div className="pdx-gallery">
          {photos.map((p) => (
            <button
              key={p.id}
              type="button"
              className={`pdx-gallery__cell ${p.isHero ? "pdx-gallery__cell--hero" : ""} ${
                p.id === selectedId ? "pdx-gallery__cell--selected" : ""
              }`}
              onClick={() => setSelectedId((cur) => (cur === p.id ? null : p.id))}
              aria-pressed={p.id === selectedId}
              aria-label="Select photo"
            >
              <AuthedImage photoId={p.id} className="pdx-gallery__img" />
              {p.isHero && <span className="pdx-gallery__badge">Display</span>}
            </button>
          ))}
          {photos.length === 0 && (
            <div className="pdx-empty">No photos for this species.</div>
          )}
        </div>
      </div>

      {zoom && (
        <div className="lightbox" onClick={() => setZoom(null)} role="dialog" aria-modal="true">
          <AuthedImage photoId={zoom} className="lightbox__img" />
        </div>
      )}

      {editing && (
        <EditSightingModal
          sighting={editing}
          currentLabel={currentLabel}
          onClose={() => setEditing(null)}
          onSaved={onChanged}
        />
      )}

      {deletingSighting && (
        <ConfirmModal
          title="Delete sighting?"
          body="This permanently removes the sighting and every photo attached to it, including the image files. This can't be undone."
          onConfirm={async () => {
            await deleteObservation(deletingSighting.observationId);
            onChanged();
          }}
          onClose={() => setDeletingSighting(null)}
        />
      )}

      {deletingPhoto && (
        <ConfirmModal
          title="Delete photo?"
          body="This permanently removes the photo and its image file. This can't be undone."
          onConfirm={async () => {
            await deletePhoto(deletingPhoto.id);
            setSelectedId(null);
            onChanged();
          }}
          onClose={() => setDeletingPhoto(null)}
        />
      )}
    </aside>
  );
}
