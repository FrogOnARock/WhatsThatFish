/* One taxonomy head's top-3: emphasised top row + two minor rows.
   The species head renders rows as selectable buttons so the page's
   description/habitat can follow the active guess; genus/family are display-only
   (their candidates carry no summary/habitat, and their indices would collide
   with species in the shared `active` value). */
import type { Taxon } from "./ResultsView";
import type { Candidate } from "../api/types";

interface ClassificationCardProps {
  taxon: Taxon;
  prediction: Candidate[];
  /** Currently-selected species index (as string), or null for "show the top guess". */
  active: string | null;
  onSelect: (id: string) => void;
  common: string;
}

export default function ClassificationCard({
  taxon,
  prediction,
  active,
  onSelect,
  common,
}: ClassificationCardProps) {
  const selectable = taxon.key === "species";
  // Mirror ResultsView's `?? speciesTop`: with nothing selected, the top guess is
  // the active one — so it expands and shows the common name by default.
  const topId = prediction[0] ? `${prediction[0].index}` : null;
  const effectiveActive = active ?? topId;

  return (
    <div className="cls-card">
      <div className="cls-card__head">
        <span className="cls-card__rank">{taxon.label}</span>
        <span className="cls-card__hint">{taxon.hint}</span>
      </div>
      <div className="cls-card__rows">
        {prediction.map((p, i) => {
          const modifier = i === 0 ? "cls-row--top" : "cls-row--minor";
          const isActive = selectable && `${p.index}` === effectiveActive;

          // The three grid cells: rank · name (+ reveal) · conf. The common-name
          // reveal animates open only for the active species (grid-rows 0fr→1fr).
          const cells = (
            <>
              <span className="cls-row__rank">{i + 1}</span>
              <div>
                <div className="cls-row__name">{p.name}</div>
                {selectable && (
                  <div className={`cls-row__reveal${isActive ? " cls-row__reveal--open" : ""}`}>
                    <div className="cls-row__common">{isActive ? common : ""}</div>
                  </div>
                )}
              </div>
              <span className="cls-row__conf">{(p.conf * 100).toFixed(1)}%</span>
            </>
          );

          return selectable ? (
            <button
              key={p.index}
              type="button"
              className={`cls-row ${modifier}${isActive ? " cls-row--active" : ""}`}
              aria-pressed={isActive}
              onClick={() => onSelect(`${p.index}`)}
            >
              {cells}
            </button>
          ) : (
            <div key={p.index} className={`cls-row ${modifier}`}>
              {cells}
            </div>
          );
        })}
      </div>
    </div>
  );
}
