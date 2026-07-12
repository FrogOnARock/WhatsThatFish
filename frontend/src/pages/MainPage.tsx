/* Main page — orchestrates idle / analyzing / results states.
   Unlike the prototype (setTimeout), the analyzing state is driven by the
   API promise resolving, so it behaves identically with the real backend. */
import { useCallback, useEffect, useMemo, useState } from "react";
import DropZone from "../components/DropZone";
import ResultsView from "../components/ResultsView";
import { getPrediction, getPredictionSample } from "../api/client";
import { SAMPLE_FISH } from "../api/prediction";
import { API_BASE } from "../api/config";
import { prefetchImages, imagesReady } from "../api/imagePrefetch";
import { getStats, type ModelStats } from "../api/stats";
import { useBackendStatus } from "../api/backendStatus";
import { useIdentifySession } from "./IdentifyContext";

// Model pill label reflects real reachability instead of a hardcoded "online".
const PILL_LABEL: Record<string, string> = {
  online: "model online",
  warming: "waking model…",
  down: "model offline",
  unknown: "model online",
};

export default function MainPage() {
  // The current identification lives in a provider ABOVE the router, so it
  // survives navigating away to the field log / dive log and back.
  const { image, request, lastTask, setImage, runInference, reset } =
    useIdentifySession();
  const [stats, setStats] = useState<ModelStats | null>(null);
  const backendStatus = useBackendStatus();
  // The prefetch cache (imagePrefetch's `settled` set) is module-level, not
  // reactive — bump this once a batch finishes so `samplesReady` re-evaluates.
  const [readyTick, setReadyTick] = useState(0);

  useEffect(() => {
    // apiFetch already retries a transient cold-start 5xx; a hard failure leaves
    // the subtitle on its "loading classes…" fallback. Log rather than swallow.
    getStats().then(setStats).catch((err) => console.error("stats load failed", err));
  }, []);

  // Stable identity so the prefetch effect and the readiness check share one array.
  const sampleUrls = useMemo(
    () => SAMPLE_FISH.map((s) => `${API_BASE}/image/${s.filename}`),
    [],
  );

  useEffect(() => {
    // Warm the sample thumbnails up front so the landing strip appears together
    // rather than trickling in. On a cold Cloud Run start these requests are held
    // open until the instance wakes, so we GATE the samples strip on this batch
    // (skeleton until warm) — but only the strip: the upload CTA stays instant
    // because it needs no backend. Immutable Cache-Control keeps them warm after.
    prefetchImages(sampleUrls).then(() => setReadyTick((t) => t + 1));
  }, [sampleUrls]);

  const samplesReady = useMemo(
    () => imagesReady(sampleUrls),
    [sampleUrls, readyTick],
  );

  const handleSample = useCallback((id: string) => {
    const sample = SAMPLE_FISH.find((s) => s.id === id);
    if (!sample) return;
    // Replacing whatever was there — clear a live file URL first.
    reset();
    setImage({
      kind: "sample",
      filename: sample.filename,
      size: "2.3 MB",
      caption: sample.label,
      // Serve the real sample frame from the backend so the bbox overlays the
      // actual fish, not the abstract placeholder.
      url: `${API_BASE}/image/${sample.filename}`,
    });
    runInference(() => getPredictionSample(sample.filename));
  }, [reset, setImage, runInference]);

  const handleUpload = useCallback((file: File) => {
    reset();
    const url = URL.createObjectURL(file);
    setImage({
      kind: "file",
      filename: file.name,
      size: `${(file.size / (1024 * 1024)).toFixed(1)} MB`,
      url,
    });
    runInference(() => getPrediction(file));
  }, [reset, setImage, runInference]);

  const handleRetry = useCallback(() => {
    if (lastTask) runInference(lastTask);
  }, [lastTask, runInference]);

  const handleReset = useCallback(() => {
    reset();
  }, [reset]);

  return (
    <main className="main">
      <div className="main__inner">
        <header className="page-header">
          <div>
            <div className="page-header__crumb">Workspace · Identify</div>
            <h1 className="page-header__title">
              What's <em>that</em> fish?
            </h1>
            <p className="page-header__subtitle">
              Drop a dive photo and the model will return its three best guesses for
              species, genus and family — plus where you're most likely to see it.
            </p>
          </div>
          <div className="page-header__model">
            <span className="page-header__model-pill">{PILL_LABEL[backendStatus]} · YOLO11 + CustomResnet</span>
            <span>
              {stats
                ? `${stats.species.toLocaleString()} species · ${stats.genera} genera · ${stats.families} families`
                : "loading classes…"}
            </span>
          </div>
        </header>

        {!image && <DropZone onUpload={handleUpload} onSample={handleSample} speciesCount={stats?.species} samplesReady={samplesReady} />}
        {image && (
          <ResultsView
            image={image}
            request={request}
            onReset={handleReset}
            onRetry={handleRetry}
          />
        )}
      </div>
    </main>
  );
}
