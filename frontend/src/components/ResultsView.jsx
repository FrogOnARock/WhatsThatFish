/* Results: image (with toggleable bbox) + 3 stacked top-3 cards + meta + actions. */
import { useState } from "react";
import FishPlaceholder from "./FishPlaceholder.jsx";
import ClassificationCard from "./ClassificationCard.jsx";

const TAXA = [
  { key: "species",   label: "Species",   hint: "TOP-3 · italic = scientific" },
  { key: "genus",     label: "Genus",     hint: "TOP-3" },
  // CustomResnet's third head is subfamily (fc_subfamily), not family.
  { key: "subfamily", label: "Subfamily", hint: "TOP-3" },
];

function Toggle({ on, onChange, children }) {
  return (
    <button className={`toggle ${on ? "toggle--on" : ""}`} onClick={() => onChange(!on)}>
      <span className="toggle__track"><span className="toggle__thumb" /></span>
      <span>{children}</span>
    </button>
  );
}

export default function ResultsView({ image, prediction, onReset, analyzing }) {
  const [showBox, setShowBox] = useState(true);
  const [location, setLocation] = useState("");
  const [saved, setSaved] = useState(false);
  const [reported, setReported] = useState(false);

  const bbox = prediction?.bbox;
  const speciesTop = prediction?.species[0];

  return (
    <div className="results">
      {/* LEFT — image + meta + actions */}
      <div className="results__left">
        <div className="image-card">
          <div className="image-card__viewport">
            {image.kind === "sample" ? (
              <FishPlaceholder hue={image.hue} caption={image.caption} large />
            ) : (
              <img src={image.url} alt="uploaded" />
            )}
            {showBox && bbox && !analyzing && (
              <div
                className="bbox"
                style={{
                  left:   `${bbox.x}%`,
                  top:    `${bbox.y}%`,
                  width:  `${bbox.w}%`,
                  height: `${bbox.h}%`,
                }}
              >
                <span className="bbox__tag">
                  {speciesTop.name} · {speciesTop.conf.toFixed(1)}%
                </span>
              </div>
            )}
            {analyzing && (
              <div className="analyzing">
                <span className="analyzing__pulse" />
                Running YOLOv11 · classifying
              </div>
            )}
          </div>
          <div className="image-card__bar">
            <div className="image-card__bar-left">
              <span className="image-card__filename">{image.filename}</span>
              <span>·</span>
              <span>{image.size}</span>
            </div>
            <Toggle on={showBox} onChange={setShowBox}>Bounding box</Toggle>
          </div>
        </div>

        {prediction && !analyzing && (
          <>
            <div className="meta-strip">
              <div className="meta-card">
                <div className="meta-card__label">Habitat & range</div>
                <h4 className="meta-card__title">{prediction.summary}</h4>
                <p className="meta-card__body">{prediction.habitat}</p>
              </div>
              <div className="meta-card">
                <div className="meta-card__label">Where did you see it?</div>
                <input
                  type="text"
                  placeholder="e.g. Tulamben, Bali · 12 m · house reef"
                  value={location}
                  onChange={(e) => setLocation(e.target.value)}
                />
                <div className="meta-card__actions">
                  <button className="btn btn--ghost btn--sm">Use GPS</button>
                  <button className="btn btn--ghost btn--sm">Drop pin</button>
                </div>
              </div>
            </div>

            <div className="actions-row">
              <div className="actions-row__meta">
                inference · 184 ms · YOLOv11 + CustomResnet · img 640×480
              </div>
              <div className="actions-row__buttons">
                <button className="btn btn--ghost btn--sm" onClick={onReset}>
                  ← Try another
                </button>
                <button
                  className="btn btn--coral-ghost btn--sm"
                  onClick={() => setReported(true)}
                  disabled={reported}
                >
                  {reported ? "Reported · thanks" : "Report wrong ID"}
                </button>
                <button
                  className="btn btn--foam btn--sm"
                  onClick={() => setSaved(true)}
                  disabled={saved}
                >
                  {saved ? "Saved to history ✓" : "Save to history"}
                </button>
              </div>
            </div>
          </>
        )}
      </div>

      {/* RIGHT — three taxonomy cards */}
      {prediction && !analyzing && (
        <div className="results__right">
          <div className="results__hed">
            <h3 className="results__hed-title">Three best guesses</h3>
            <span className="results__hed-sub">multi-head · 3 taxa</span>
          </div>
          {TAXA.map((t) => (
            <ClassificationCard
              key={t.key}
              taxon={t}
              predictions={prediction[t.key]}
              common={prediction.common}
            />
          ))}
        </div>
      )}
    </div>
  );
}
