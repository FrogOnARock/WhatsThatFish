import FishPlaceholder from "../FishPlaceholder";
import type { Ghost, LogSpecies } from "../../api/types";

interface SpeciesCardProps {
  sp: LogSpecies;
  active: boolean;
  onSelect: (id: string) => void;
}

export function SpeciesCard({ sp, active, onSelect }: SpeciesCardProps) {
  return (
    <button
      className={`pdx-card ${active ? "pdx-card--active" : ""}`}
      onClick={() => onSelect(sp.id)}
    >
      <div className="pdx-card__no">№ {String(sp.no).padStart(3, "0")}</div>
      <div className="pdx-card__thumb">
        <FishPlaceholder hue={sp.hue} caption={sp.caption} />
      </div>
      <div className="pdx-card__body">
        <div className="pdx-card__common">{sp.common}</div>
        <div className="pdx-card__sci">{sp.species}</div>
      </div>
      <div className="pdx-card__count">
        <span className="pdx-card__count-num">{sp.sightings.length}</span>
        <span className="pdx-card__count-lbl">seen</span>
      </div>
    </button>
  );
}

interface GhostCardProps {
  ghost: Ghost;
}

export function GhostCard({ ghost }: GhostCardProps) {
  return (
    <div className="pdx-card pdx-card--ghost" aria-disabled>
      <div className="pdx-card__no">№ {String(ghost.no).padStart(3, "0")}</div>
      <div className="pdx-card__thumb pdx-card__thumb--ghost">
        <span className="pdx-card__qmark">?</span>
      </div>
      <div className="pdx-card__body">
        <div className="pdx-card__common">Undiscovered</div>
        <div className="pdx-card__sci">{ghost.hint}</div>
      </div>
    </div>
  );
}
