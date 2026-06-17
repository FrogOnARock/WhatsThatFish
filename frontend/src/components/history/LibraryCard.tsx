import type { SpeciesEntry } from "../../api/types";
import FishPlaceholder from "../FishPlaceholder";

interface LibraryCardProps {
    sp: SpeciesEntry;
    onSelect: (id: string) => void;
}

export function LibraryCard({ sp, onSelect }: LibraryCardProps) {
    return (
    <button
      className={`pdx-card`}
      onClick={() => onSelect(`${sp.speciesId}`)}
    >
      <div className="pdx-card__no">{String(sp.speciesId).padStart(3, "0")}</div>
      <div className="pdx-card__thumb">
        <FishPlaceholder/>
      </div>
      <div className="pdx-card__body">
        <div className="pdx-card__common">{sp.common}</div>
        <div className="pdx-card__sci">{sp.name}</div>
      </div>
    </button>
  );
}
