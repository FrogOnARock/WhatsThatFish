/* Sticky right-hand panel: hero, taxonomy crumb, stats, captures, sightings log. */
import FishPlaceholder from "../FishPlaceholder";
import type { SpeciesEntry } from "../../api/types";
import { useState } from "react";
import { API_BASE } from "../../api/config";

export function fmtDate(iso: string): string {
  const d = new Date(iso + "T12:00:00");
  return d.toLocaleDateString("en-US", { day: "2-digit", month: "short", year: "numeric" });
}

interface DetailPanelProps {
  sp: SpeciesEntry;
}

export default function LibraryPanel({ sp }: DetailPanelProps) {
  const [broken, setBroken] = useState(false);

  // Shared across both hero states (real image vs. placeholder fallback) so the
  // fields are written once and can't drift between branches.
  const body = (
    <div className="pdx-detail__title-block">
      <div className="pdx-detail__crumb">
        <span>{sp.family}</span>
        <span className="pdx-detail__crumb-sep">›</span>
        <span>{sp.genus}</span>
        <span className="pdx-detail__crumb-sep">›</span>
        <span className="pdx-detail__crumb-sp">{sp.name.split(" ")[1]}</span>
      </div>
      <h2 className="pdx-detail__common">{sp.common}</h2>
      <div className="pdx-detail__sci">{sp.name}</div>

      {sp.description && (
        <div className="pdx-detail__section">{sp.description}</div>
      )}

      {sp.location.length > 0 && (
        <div className="pdx-detail__section">
          <div className="pdx-detail__section-head">
            <h4>Range</h4>
          </div>
          <div className="pdx-detail__chips">
            {sp.location.map((loc) => (
              <span key={loc} className="pdx-detail__chip">{loc}</span>
            ))}
          </div>
        </div>
      )}

      {sp.depth && (
        <div className="pdx-detail__section">
          <div className="pdx-detail__section-head">
            <h4>Depth</h4>
          </div>
          <div className="pdx-detail__depth">{sp.depth}</div>
        </div>
      )}
    </div>
  );

  const hero = broken ? (
    <div className="pdx-detail__hero">
      <FishPlaceholder large />
      <div className="pdx-detail__hero-no">{String(sp.speciesId).padStart(3, "0")}</div>
    </div>
  ) : (
    <div className="pdx-detail__hero">
      <img
        src={`${API_BASE}/image/${sp.filename}`}
        alt={sp.name}
        loading="lazy"
        onError={() => setBroken(true)}
      />
      <div className="pdx-detail__hero-no">{String(sp.speciesId).padStart(3, "0")}</div>
    </div>
  );

  return (
    <aside className="pdx-detail">
      {hero}
      {body}
    </aside>
  );
}
