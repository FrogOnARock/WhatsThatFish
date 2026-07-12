/* Sticky right-hand panel: taxonomy crumb, names, description, range, depth, and
   a link out to Wikipedia. Images were dropped from the library in favour of the
   tabular listing, so this is text-only now. */
import type { SpeciesEntry } from "../../api/types";
import WikipediaButton from "../WikipediaButton";

interface DetailPanelProps {
  sp: SpeciesEntry;
}

export default function LibraryPanel({ sp }: DetailPanelProps) {
  return (
    <aside className="pdx-detail">
      <div className="pdx-detail__title-block">
        <div className="pdx-detail__crumb">
          <span>{sp.family}</span>
          <span className="pdx-detail__crumb-sep">›</span>
          <span>{sp.genus}</span>
          <span className="pdx-detail__crumb-sep">›</span>
          <span className="pdx-detail__crumb-sp">{sp.name.split(" ")[1]}</span>
        </div>
        <h2 className="pdx-detail__common">{sp.common || sp.name}</h2>
        <div className="pdx-detail__sci">{sp.name}</div>

        <WikipediaButton name={sp.name} className="pdx-detail__wiki" />

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

        <div className="pdx-detail__section">
          <div className="pdx-detail__section-head">
            <h4>Training images</h4>
          </div>
          <div className="pdx-detail__depth">{sp.imageCount.toLocaleString()}</div>
        </div>
      </div>
    </aside>
  );
}
