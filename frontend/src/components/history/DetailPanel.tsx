/* Sticky right-hand panel: hero photo, taxonomy crumb, stats, sightings log,
   and a gallery of every photo (click to enlarge). Driven by the real field-log shape. */
import { useState } from "react";
import FishPlaceholder from "../FishPlaceholder";
import AuthedImage from "../AuthedImage";
import StatPill from "./StatPill";
import EditSightingModal from "../EditSightingModal";
import type { FieldSpecies, FieldSighting } from "../../api/history";
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

export default function DetailPanel({
  sp,
  onChanged,
}: {
  sp: FieldSpecies;
  onChanged: () => void;
}) {
  const { user } = useAuth();
  const units = user?.unitSystem ?? "metric";
  const [zoom, setZoom] = useState<string | null>(null);
  const [editing, setEditing] = useState<FieldSighting | null>(null);
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
  const hero = photos[0]?.id ?? null;

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
                  <button className="pdx-table__edit" onClick={() => setEditing(s)}>
                    Edit
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="pdx-detail__section">
        <div className="pdx-detail__section-head">
          <h4>Photos</h4>
          <span className="pdx-detail__section-meta">
            {photos.length} photo{photos.length === 1 ? "" : "s"}
          </span>
        </div>
        <div className="pdx-gallery">
          {photos.map((p) => (
            <button
              key={p.id}
              type="button"
              className="pdx-gallery__cell"
              onClick={() => setZoom(p.id)}
              aria-label="Enlarge photo"
            >
              <AuthedImage photoId={p.id} className="pdx-gallery__img" />
            </button>
          ))}
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
    </aside>
  );
}
