import type { SpeciesEntry  } from "../../api/types";
import FishPlaceholder from "../FishPlaceholder";
import { useState } from "react";
import { API_BASE } from "../../api/config";

interface LibraryCardProps {
    sp: SpeciesEntry;
    onSelect: (id: string) => void;
}

export function SampleCard({ sp, onSelect }: LibraryCardProps) {
    const [broken, setBroken] = useState(false);

    if (broken) return (
        <button
          className={`pdx-card`}
          onClick={() => onSelect(`${sp.speciesId}`)}
        >
          <div className="pdx-card__thumb">
              < FishPlaceholder />
          </div>
          <div className="pdx-card__body">
            <div className="pdx-card__common">{sp.common}</div>
            <div className="pdx-card__sci">{sp.name}</div>
          </div>
        </button>
    )

    return (
        <button
          className={`pdx-card`}
          onClick={() => onSelect(`${sp.speciesId}`)}
        >
          <div className="pdx-card__thumb">
              <img src={ `${API_BASE}/image/${sp.filename}` } alt={sp.name}
              loading="lazy" onError={() => setBroken(true)}>
              </img>
          </div>
          <div className="pdx-card__body">
            <div className="pdx-card__common">{sp.common}</div>
            <div className="pdx-card__sci">{sp.name}</div>
          </div>
        </button>
  )
}
