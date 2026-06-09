/* Drop zone + sample fish row — the idle/empty state of the main page. */
import { useEffect, useRef, useState } from "react";
import FishPlaceholder from "./FishPlaceholder.jsx";
import { SAMPLE_FISH, DEFAULT_PREDICTION_KEY } from "../api/client.js";

export default function DropZone({ onUpload, onSample }) {
  const [hot, setHot] = useState(false);
  const inputRef = useRef(null);

  // Paste-from-clipboard handler
  useEffect(() => {
    const handler = (e) => {
      const items = e.clipboardData?.items || [];
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

  const handleFiles = (files) => {
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
          The model will return top-3 candidates for species, genus and subfamily.
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
          <span className="samples__hint">5 of 1,247 known species</span>
        </header>
        <div className="samples">
          {SAMPLE_FISH.map((s) => (
            <button key={s.id} className="sample" onClick={() => onSample(s.id)}>
              <div className="sample__thumb">
                <FishPlaceholder hue={s.hue} caption={s.caption} />
              </div>
              <div className="sample__meta">
                <span className="sample__name">{s.label}</span>
                <span className="sample__chev">→</span>
              </div>
            </button>
          ))}
        </div>
      </section>
    </div>
  );
}
