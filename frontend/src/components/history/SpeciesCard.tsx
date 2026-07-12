import FishPlaceholder from "../FishPlaceholder";
import AuthedImage from "../AuthedImage";
import type { FieldSpecies } from "../../api/history";

/** The species' card image: the user-chosen hero across any sighting, else the
    first available photo. */
function cardPhotoId(sp: FieldSpecies): string | null {
  const all = sp.sightings.flatMap((s) => s.photos);
  return (all.find((p) => p.isHero) ?? all[0])?.id ?? null;
}

interface SpeciesCardProps {
  sp: FieldSpecies;
  no: number;
  active: boolean;
  onSelect: (taxonId: number) => void;
}

export function SpeciesCard({ sp, no, active, onSelect }: SpeciesCardProps) {
  const photo = cardPhotoId(sp);
  return (
    <button
      className={`pdx-card ${active ? "pdx-card--active" : ""}`}
      onClick={() => onSelect(sp.taxonId)}
    >
      <div className="pdx-card__no">№ {String(no).padStart(3, "0")}</div>
      <div className="pdx-card__thumb">
        {photo ? (
          <AuthedImage photoId={photo} className="pdx-card__img" />
        ) : (
          <FishPlaceholder hue={sp.taxonId % 360} caption={sp.species ?? ""} />
        )}
      </div>
      <div className="pdx-card__body">
        <div className="pdx-card__common">{sp.commonName ?? sp.species ?? "Unknown"}</div>
        <div className="pdx-card__sci">{sp.species}</div>
      </div>
      <div className="pdx-card__count">
        <span className="pdx-card__count-num">{sp.sightingCount}</span>
        <span className="pdx-card__count-lbl">seen</span>
      </div>
    </button>
  );
}
