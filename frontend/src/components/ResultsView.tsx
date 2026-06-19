/* Results: image (with toggleable bbox) + 3 stacked top-3 cards + meta + actions. */
import { useState } from "react";
import type { ReactNode } from "react";
import ClassificationCard from "./ClassificationCard";
import type {TaxonKey} from "../api/types";
import type { RequestState } from "../pages/MainPage";
import ImageCard from "./ImageCard";
import { SAMPLE_FISH } from "../api/prediction";

/** A taxonomy head as presented in the results column. */
export interface Taxon {
  key: TaxonKey;
  label: string;
  hint: string;
}

/** The image the user is identifying — a hosted sample or an uploaded file. */
export interface ImageState {
  kind: "sample" | "file";
  filename: string;
  size: string;
  hue?: number;
  caption?: string;
  url?: string;
}

const TAXA: Taxon[] = [
  { key: "species",   label: "Species",   hint: "TOP-3 · italic = scientific" },
  { key: "genus",     label: "Genus",     hint: "TOP-3" },
  // CustomResnet's third (coarsest) head is family (fc_family).
  { key: "family", label: "Family", hint: "TOP-3" },
];

interface ToggleProps {
  on: boolean;
  onChange: (on: boolean) => void;
  children: ReactNode;
}

function Toggle({ on, onChange, children }: ToggleProps) {
  return (
    <button className={`toggle ${on ? "toggle--on" : ""}`} onClick={() => onChange(!on)}>
      <span className="toggle__track"><span className="toggle__thumb" /></span>
      <span>{children}</span>
    </button>
  );
}

interface ResultsViewProps {
  image: ImageState;
  request: RequestState | null;
  onReset: () => void;
  onRetry: () => void;
}

export default function ResultsView({ image, request, onReset, onRetry }: ResultsViewProps) {
  const [showBox, setShowBox] = useState(true);
  const [location, setLocation] = useState("");
  const [saved, setSaved] = useState(false);
  const [reported, setReported] = useState(false);

  if (!request) {
    return null
  }

  if (request.status === "analyzing") {
    return (
        <div className="results">
          <div className="results__left">
            <ImageCard image={image} overlay={
              <div className="analyzing">
                <span className="analyzing__pulse"/>
                Running YOLOv11 · classifying
              </div>
            }/>
          </div>
        </div>
    );
  }

  if (request.status === "no-fish") {
    return (
        <div className="results">
          <div className="results__left">
            <ImageCard image={image} overlay={
              <div className="standard_msg">
                No fish found. Please retry with another photo.
              </div>
            } barAction={
              <button onClick={onRetry}>
                 ← Retry
              </button>
            }/>
          </div>
          <div className="actions-row">
            <div className="actions-row__meta">
              inference · 184 ms · YOLOv11 + CustomResnet · img 640×480
            </div>
            <div className="actions-row__buttons">
              <button className="btn btn--ghost btn--sm" onClick={onReset}>
                ← Try another
              </button>
             </div>
          </div>
        </div>
    );
  }

  if (request.status === "error") {
    return (
        <div className="results">
          <div className="results__left">
            <ImageCard image={image} overlay={
              <div className="standard_msg">
                Error on detection/classification: {request.message}
              </div>
            }
            barAction={
              <button onClick={onRetry}>
                 ← Retry
              </button>
            }/>
          </div>
        <div className="actions-row">
            <div className="actions-row__meta">
              inference · 184 ms · YOLOv11 + CustomResnet · img 640×480
            </div>
            <div className="actions-row__buttons">
              <button className="btn btn--ghost btn--sm" onClick={onReset}>
                ← Try another
              </button>
             </div>
          </div>
        </div>
    );
  }


  const bbox = request.prediction.bbox[0];
  const speciesTop = request.prediction.species[0];

  return (
      <div className="results">
        <div className="results__left">
          <ImageCard image={image} overlay={
            showBox && bbox && speciesTop && (
                <div
                className="bbox"
                style={{
                  left: `${bbox.x}%`,
                  top: `${bbox.y}%`,
                  width: `${bbox.w}%`,
                  height: `${bbox.h}%`,
                }}
            >
              <span className="bbox__tag">
                {speciesTop.name} · {speciesTop.conf.toFixed(1)}%
                </span>
            </div>)
          } barAction={
            <Toggle on={showBox} onChange={setShowBox}>Bounding box</Toggle>
          }/>

          <div className="meta-strip">
            <div className="meta-card">
              <div className="meta-card__label">Habitat & range</div>
              <h4 className="meta-card__title">{speciesTop.summary}</h4>
              <p className="meta-card__body">{speciesTop.habitat}</p>
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
        </div>

          <div className="results__right">
            <div className="results__hed">
              <h3 className="results__hed-title">Three best guesses</h3>
              <span className="results__hed-sub">multi-head · 3 taxa</span>
            </div>
            {TAXA.map((t) => (
                <ClassificationCard
                    key={t.key}
                    taxon={t}
                    predictions={speciesTop.name}
                    common={speciesTop.common}
                />
            ))}
          </div>
      </div>
  );
}