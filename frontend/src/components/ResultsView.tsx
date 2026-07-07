/* Results: image (with toggleable bbox) + 3 stacked top-3 cards + meta + actions. */
import { useEffect, useState } from "react";
import type { ReactNode } from "react";
import ClassificationCard from "./ClassificationCard";
import type {TaxonKey} from "../api/types";
import type { RequestState } from "../pages/MainPage";
import ImageCard from "./ImageCard";
import SaveObservationModal from "./SaveObservationModal";
import { useBackendStatus } from "../api/backendStatus";

/** Analyzing overlay with an elapsed timer. For the first few seconds it reads as
    a normal inference; once it's clearly slow (or the backend is warming) it
    explains the cold start so a 30s wait doesn't look like a hang. Its own
    component so the interval hook lives outside ResultsView's early-return
    branches. */
const COLD_HINT_AFTER = 5; // seconds

function AnalyzingOverlay() {
  const [elapsed, setElapsed] = useState(0);
  const status = useBackendStatus();

  useEffect(() => {
    const id = setInterval(() => setElapsed((s) => s + 1), 1000);
    return () => clearInterval(id);
  }, []);

  const cold = elapsed >= COLD_HINT_AFTER || status === "warming" || status === "down";
  return (
    <div className="analyzing">
      <span className="analyzing__pulse" />
      {cold
        ? `Waking the model — first inference after idle can take ~30s · ${elapsed}s`
        : "Running YOLOv11 · classifying"}
    </div>
  );
}

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
  const [saved, setSaved] = useState(false);
  const [reported, setReported] = useState(false);
  const [active, setActive] = useState<string | null>(null);
  const [modal, setModal] = useState<null | "save" | "report">(null);

  if (!request) {
    return null
  }

  if (request.status === "analyzing") {
    return (
        <div className="results">
          <div className="results__left">
            <ImageCard image={image} overlay={<AnalyzingOverlay />}/>
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
  const detected = request.prediction.detected;
  // The selected guess, or the top guess when nothing is selected. The `?? speciesTop`
  // fallback makes this ALWAYS a Candidate — so every field below is a plain
  // string/string[], never `Candidate | undefined` (and never a stray boolean from
  // a per-field ternary). Read summary/habitat/common straight off it.
  const speciesActive =
      request.prediction.species.find((s) => `${s.index}` === active) ?? speciesTop;

  return (
      <div className="results">
        <div className="results__left">
          <ImageCard image={image} overlay={
            <>
              {showBox && bbox && speciesTop && (
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
                  {speciesActive.name} · {(speciesActive.conf * 100).toFixed(1)}%
                  </span>
              </div>)}
              {!detected && (
                  <div className="standard_msg">
                    No fish detected — species identified from the full image. Treat as a low-confidence guess.
                  </div>)}
            </>
          } barAction={
            // No box to toggle when nothing was detected.
            detected && <Toggle on={showBox} onChange={setShowBox}>Bounding box</Toggle>
          }/>

          <div className="meta-strip">
            <div className="meta-card">
              <div className="meta-card__label">Description & Habitat</div>
              <h4 className="meta-card__body">{speciesActive.summary}</h4>
              <p className="meta-card__chips">{speciesActive.habitat.map((loc) => (
                <span key={loc} className="meta-card__chip">{loc}</span>
              ))}</p>
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
                    prediction={
                        t.key === "species" ? request.prediction.species :
                        t.key === "genus" ? request.prediction.genus :
                        request.prediction.family
                      }
                    active={active}
                    onSelect={setActive}
                    common={speciesActive.common}
                />
            ))}
            <div className="actions-row">
              <div className="actions-row__meta">
                inference · 184 ms · YOLOv11 + CustomResnet · img 640×480
              </div>
              <div className="actions-row__buttons">
                <button className="btn btn--ghost btn--sm" onClick={onReset}>
                  ← Try another
                </button>
                {/* Samples are demo images — they must never enter a user's field
                    log, so only real uploads get the save/report actions. */}
                {image.kind === "sample" ? (
                  <span className="actions-row__note">
                    Sample image — upload your own dive photo to log it.
                  </span>
                ) : (
                  <>
                    <button
                        className="btn btn--coral-ghost btn--sm"
                        onClick={() => setModal("report")}
                        disabled={reported}
                    >
                      {reported ? "Reported · thanks" : "Report wrong ID"}
                    </button>
                    <button
                        className="btn btn--foam btn--sm"
                        onClick={() => setModal("save")}
                        disabled={saved}
                    >
                      {saved ? "Saved to history ✓" : "Save to history"}
                    </button>
                  </>
                )}
              </div>
            </div>
          </div>

        {modal && (
          <SaveObservationModal
            mode={modal}
            image={image}
            prediction={request.prediction}
            selectedSpecies={speciesActive}
            onClose={() => setModal(null)}
            onSaved={() => (modal === "report" ? setReported(true) : setSaved(true))}
          />
        )}
      </div>
  );
}