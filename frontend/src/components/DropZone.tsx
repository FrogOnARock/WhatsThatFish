/* Drop zone + sample fish row — the idle/empty state of the main page. */
import { useEffect, useRef, useState } from "react";
import FishPlaceholder from "./FishPlaceholder";
import { SAMPLE_FISH, SAMPLE, DEFAULT_PREDICTION_KEY } from "../api/prediction";
import {API_BASE} from "../api/config";

/** One sample thumbnail. Owns a `broken` flag so a failed load (e.g. the fast
    cold-start 503 tail) falls back to the striped placeholder instead of the
    browser's broken-image glyph — the inline `onError` here previously no-op'd. */
function SampleTile({ sample, onSelect }: { sample: SAMPLE; onSelect: (id: string) => void }) {
  const [broken, setBroken] = useState(false);
  return (
    <button className="sample" onClick={() => onSelect(sample.id)}>
      <div className="sample__thumb">
        {broken ? (
          <FishPlaceholder caption={sample.label} />
        ) : (
          <img
            src={`${API_BASE}/image/${sample.filename}`}
            alt={sample.label}
            loading="lazy"
            onError={() => setBroken(true)}
          />
        )}
      </div>
      <div className="sample__meta">
        <span className="sample__name">{sample.label}</span>
        <span className="sample__chev">→</span>
      </div>
    </button>
  );
}

interface DropZoneProps {
  onUpload: (file: File) => void;
  onSample: (id: string) => void;
  speciesCount?: number;
  // False while the sample thumbnails are still warming (notably during a Cloud
  // Run cold start). We show skeleton tiles instead of blank/broken <img>s until
  // the whole strip can paint at once. The upload CTA above is unaffected.
  samplesReady?: boolean;
}

export default function DropZone({ onUpload, onSample, speciesCount, samplesReady = true }: DropZoneProps) {
  const [hot, setHot] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  // Paste-from-clipboard handler
  useEffect(() => {
    const handler = (e: ClipboardEvent) => {
      // DataTransferItemList is array-like but not iterable — materialise first.
      const items = Array.from(e.clipboardData?.items ?? []);
      for (const item of items) {
        if (item.kind === "file" && item.type.startsWith("image/")) {
          const file = item.getAsFile();
          if (file) onUpload(file);
          return;
        }
      }
    };
    window.addEventListener("paste", handler);
    return () => window.removeEventListener("paste", handler);
  }, [onUpload]);

  const handleFiles = (files: FileList | null) => {
    if (!files || !files.length) return;
    const file = files[0];
    if (file.type.startsWith("image/")) onUpload(file);
  };

  return (
    <div className="idle">
      <div
        className={`dropzone ${hot ? "dropzone--hot" : ""}`}
        onDragOver={(e) => { e.preventDefault(); setHot(true); }}
        onDragLeave={() => setHot(false)}
        onDrop={(e) => {
          e.preventDefault();
          setHot(false);
          handleFiles(e.dataTransfer.files);
        }}
      >
        <div className="dropzone__icon" aria-hidden />
        <h2 className="dropzone__title">
          Drag a photo of <em>that fish</em> here
        </h2>
        <p className="dropzone__sub">
          The model will return top-3 candidates for species, genus and family.
        </p>

        <div className="dropzone__actions">
          <button className="btn" onClick={() => inputRef.current?.click()}>
            Browse files
          </button>
          <button className="btn btn--ghost" onClick={() => onSample(DEFAULT_PREDICTION_KEY)}>
            Try a sample
          </button>
          <input
            ref={inputRef}
            type="file"
            accept="image/*"
            hidden
            onChange={(e) => handleFiles(e.target.files)}
          />
        </div>

        <div className="dropzone__hint">
          or paste an image — <kbd>⌘</kbd>+<kbd>V</kbd>
        </div>
      </div>

      <section>
        <header className="samples__head">
          <h3 className="samples__title">— or try one of these</h3>
          <span className="samples__hint">
            {SAMPLE_FISH.length} of {speciesCount ? speciesCount.toLocaleString() : "…"} known species
          </span>
        </header>
        {samplesReady ? (
          <div className="samples">
            {SAMPLE_FISH.map((s: SAMPLE) => (
              <SampleTile key={s.id} sample={s} onSelect={onSample} />
            ))}
          </div>
        ) : (
          <>
            {/* Cold-start stop-gap: one skeleton tile per known sample so the grid
                keeps its shape (no layout shift when the real strip swaps in). */}
            <div className="samples samples--loading">
              {SAMPLE_FISH.map((s: SAMPLE) => (
                <div key={s.id} className="sample sample--skeleton" aria-hidden>
                  <div className="sample__thumb" />
                  <div className="sample__meta">
                    <span className="sample__name">{s.label}</span>
                  </div>
                </div>
              ))}
            </div>
            <p className="samples__warming">
              <span className="analyzing__pulse" />
              waking the model — the first load after a period of inactivity can take ~30s
            </p>
          </>
        )}
      </section>
    </div>
  );
}
