/* Sticky right-hand panel: hero, taxonomy crumb, stats, captures, sightings log. */
import FishPlaceholder from "../FishPlaceholder";
import type { SpeciesEntry } from "../../api/types";

export function fmtDate(iso: string): string {
  const d = new Date(iso + "T12:00:00");
  return d.toLocaleDateString("en-US", { day: "2-digit", month: "short", year: "numeric" });
}

interface DetailPanelProps {
  sp: SpeciesEntry;
}

export default function LibraryPanel({ sp }: DetailPanelProps) {

  return (
    <aside className="pdx-detail">
      <div className="pdx-detail__hero">
        <FishPlaceholder large />
        <div className="pdx-detail__hero-no">{String(sp.speciesId).padStart(3, "0")}</div>
      </div>

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
        <div className=".pdx-detail__section">{sp.description}</div>
        </div>
    </aside>
  );
}
